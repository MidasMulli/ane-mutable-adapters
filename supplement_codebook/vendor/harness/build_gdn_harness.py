#!/usr/bin/env python3
"""BUILD THE GDN SWAP/GEN HARNESS (`gdn_kvswap.mm`) FOR A GIVEN MODEL + CTX.

WHY THIS EXISTS. The recipe lived in one line of `BOOT_2026-08-13_op_form_solved.md`
("q35_swap_4b = clang++ WITHOUT -DGDN_Q35_08B") and one cell of `TOOLS.md` ("Build +
codesign --entitlements"). Neither is runnable, and neither carries the four things that
actually bite. Rebuilt from scratch on 2026-08-15 while raising CTX 512 -> 4096; every
trap below cost a real cycle that day.

THE FOUR TRAPS, ALL HANDLED HERE:

  1. OUT_* NAMES SHIFT WITH THE GRAPH. The monolithic 0.8B export returns an UNNAMED
     tuple, so coremltools numbers the outputs `var_NNNN` and RENUMBERS them whenever the
     graph changes (linear var_5295.., conv var_5967.., and the CTX=4096 re-export came
     back NAMED: out_hidden/out_conv_states/...). A stale name does NOT produce a wrong
     answer -- it inserts a nil into stepChunks' input dict and THROWS.
     ⇒ THIS SCRIPT READS THE NAMES OUT OF THE MODEL'S OWN metadata.json AND MAPS THEM BY
       SHAPE. Never type them, never carry them forward from a previous build.

  2. CTX IS THE HARNESS'S CEILING, NOT THE MODEL'S (card: 262,144 native). It sizes three
     things that must all agree with the .mlmodelc: the kv_caches MLMultiArray, the
     causal_mask, and the RoPE table read length. Mismatch fails as a SHAPE error:
       "MultiArray shape (6 x 2 x 2 x 512 x 256) does not match ... (6 x 2 x 2 x 4096 x 256)"
     ⇒ --ctx sets -DGDN_CTX and (with --rope) regenerates matching cos/sin caches.

  3. THE RoPE CACHES ARE SIZED CTX*ROT AND ARE READ WITH NO LENGTH CHECK (`rf32`, which
     fread()s the count it was asked for). A 512-row table under CTX=4096 reads past the
     end of the file -- silent garbage, not an error.
     ⇒ --rope COPIES the proven rows byte-for-byte and APPENDS only the new ones, so the
       already-validated region is identical BY CONSTRUCTION rather than by reproduction.

  4. md5 OF THE WHOLE BINARY IS NOT A VALID BYTE-IDENTITY INSTRUMENT. The linker stamps a
     fresh LC_UUID per link, so the SAME SOURCE built twice gives different md5s -- signed
     AND unsigned (measured 2026-08-15; four builds, four hashes). Reading that as "my edit
     changed the binary" is an instrument artifact read as a finding.
     ⇒ THIS SCRIPT REPORTS THE md5 OF THE __TEXT,__text SECTION, which IS stable across
       rebuilds of identical source (positive control run the same day: two same-source
       builds agreed, and a -DGDN_CTX=4096 build differed -- so the instrument can see the
       change it is being asked about).

USAGE
  build_gdn_harness.py --variant 08b --ctx 512                     # the proven reference
  build_gdn_harness.py --variant 08b --ctx 4096 \
      --model ~/models/q35_08b_ctx4096/ffn_24layers.mlmodelc \
      --rope --out q35_swap_conv_4096
  build_gdn_harness.py --variant 4b                                # the 4B anchor branch

The harness ALWAYS needs the entitlement (mandatory harness-entitlement rule); an unsigned
binary fails the mutable bind with -14 and looks like a model defect.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # ane-mutable-weights/
SRC = ROOT / "tools" / "gdn_kvswap.mm"
PLIST = ROOT / "tools" / "ane_entitlements.plist"
CFLAGS = ["-fobjc-arc", "-O2", "-framework", "Foundation", "-framework", "CoreML"]
ROT = 64
# The four tuple outputs, keyed by RANK+ROLE so the mapping survives a rename. Rank is
# what actually distinguishes them; see gdn_kvswap.mm's own shape table.
ROLE_BY_RANK = {3: "OUT_H", 4: "OUT_R", 5: "OUT_K"}   # 3=[1,1,H] 4=[NS,NV,128,128] 5=kv


def sh(cmd, **kw):
    """Run and CHECK. Never suppress a build step, always read the exit code."""
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print(f"  ⛔ FAILED ({r.returncode}): {' '.join(map(str, cmd))}")
        print((r.stderr or r.stdout)[-2000:])
        sys.exit(r.returncode)
    return r


def text_md5(binary: Path) -> str:
    """md5 of __TEXT,__text -- stable across rebuilds, unlike the whole-file md5."""
    r = subprocess.run(["otool", "-s", "__TEXT", "__text", str(binary)],
                       capture_output=True, text=True)
    body = "\n".join(r.stdout.splitlines()[1:])
    return hashlib.md5(body.encode()).hexdigest()


def derive_out_names(mlmodelc: Path):
    """Read the output names out of the model itself and map them BY SHAPE.

    This is trap 1. Returns {} for the 4B branch, which uses fixed short names.
    """
    meta = mlmodelc / "metadata.json"
    if not meta.exists():
        print(f"  ⚠️  no metadata.json at {meta} -- cannot derive OUT_* names")
        return None
    m = json.load(open(meta))
    m = m[0] if isinstance(m, list) else m
    outs = m.get("outputSchema", [])
    if not outs:
        print("  ⚠️  metadata.json has no outputSchema")
        return None
    names = {}
    conv_seen = False
    for o in outs:
        nm = o.get("name")
        shape = o.get("shape")
        if isinstance(shape, str):
            shape = json.loads(shape)
        rank = len(shape)
        if rank == 3:
            # both hidden [1,1,H] and conv [NS,CONV_D,3] are rank 3; hidden has 1s leading
            key = "OUT_H" if int(shape[0]) == 1 and int(shape[1]) == 1 else "OUT_C"
            if key == "OUT_C":
                conv_seen = True
        else:
            key = ROLE_BY_RANK.get(rank)
        if key:
            names[key] = nm
    if not conv_seen:
        print("  ⚠️  no conv-states output identified")
    missing = {"OUT_H", "OUT_C", "OUT_R", "OUT_K"} - set(names)
    if missing:
        print(f"  ⚠️  could not derive {sorted(missing)} -- falling back to source defaults")
    return names


def build_rope(ctx: int, out_dir: Path, ref_dir: Path, theta: float):
    """Trap 3. COPY the proven prefix, APPEND only new rows.

    The proven region is then identical BY CONSTRUCTION. Only rows >= len(ref) are
    generated, and they carry the stated fp32 bound below.
    """
    import numpy as np, torch
    out_dir.mkdir(parents=True, exist_ok=True)
    inv = 1.0 / (theta ** (torch.arange(0, ROT, 2, dtype=torch.float32) / ROT))
    fr = torch.outer(torch.arange(ctx, dtype=torch.float32), inv)
    emb = torch.cat([fr, fr], dim=-1)
    cos = emb.cos().numpy().astype(np.float32)
    sin = emb.sin().numpy().astype(np.float32)
    for nm, gen in (("cos_cache.f32", cos), ("sin_cache.f32", sin)):
        ref_f = ref_dir / nm
        if ref_f.exists():
            ref = np.fromfile(ref_f, dtype=np.float32).reshape(-1, ROT)
            n = min(len(ref), ctx)
            delta = float(np.abs(gen[:n] - ref[:n]).max())
            gen[:n] = ref[:n]              # PROVEN ROWS COPIED, NOT REGENERATED
            print(f"    {nm}: rows 0..{n-1} copied from the proven table "
                  f"(regen would have differed by {delta:.3e} fp32; the driver casts to "
                  f"__fp16, whose spacing near 1.0 is ~4.9e-04, so it is below the "
                  f"precision actually consumed)")
        else:
            print(f"    {nm}: no reference at {ref_f} -- FULLY generated")
        gen.tofile(out_dir / nm)
    print(f"    wrote {cos.shape} ({cos.nbytes} bytes each) -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["08b", "4b"], default="08b")
    ap.add_argument("--ctx", type=int, default=512,
                    help="harness ceiling; MUST match the .mlmodelc kv/mask shape")
    ap.add_argument("--model", type=Path,
                    help=".mlmodelc to derive OUT_* names from (trap 1). Strongly advised.")
    ap.add_argument("--out", default=None, help="output binary name")
    ap.add_argument("--outdir", type=Path, default=Path.cwd())
    ap.add_argument("--rope", action="store_true",
                    help="also emit cos/sin caches sized to --ctx (trap 3)")
    ap.add_argument("--rope-ref", type=Path,
                    default=ROOT.parent / "gdn-coreml" / "_q35_08b_demo",
                    help="dir holding the PROVEN cos/sin tables to copy rows from")
    ap.add_argument("--rope-out", type=Path, default=None)
    ap.add_argument("--theta", type=float, default=1e7,
                    help="rope_theta; 1e7 verified against the on-disk 0.8B table "
                         "(every other candidate was off by ~2.0, this by 1 ULP)")
    ap.add_argument("--no-sign", action="store_true",
                    help="skip codesign. The harness NEEDS the entitlement; unsigned "
                         "fails the mutable bind with -14 and reads as a model defect.")
    a = ap.parse_args()

    if not SRC.exists():
        sys.exit(f"⛔ no source at {SRC}")
    out = a.out or (f"q35_swap_conv_{a.ctx}" if a.variant == "08b" else "q35_swap_4b")
    a.outdir.mkdir(parents=True, exist_ok=True)
    out_path = (a.outdir / out).resolve()

    defs = []
    if a.variant == "08b":
        defs.append("-DGDN_Q35_08B=1")
    if a.ctx != 512:
        defs.append(f"-DGDN_CTX={a.ctx}")

    print(f"══ build_gdn_harness  variant={a.variant}  ctx={a.ctx}")
    if a.model:
        names = derive_out_names(Path(a.model).expanduser())
        if names:
            print(f"  OUT_* derived from the model:")
            for k in ("OUT_H", "OUT_C", "OUT_R", "OUT_K"):
                if k in names:
                    print(f"    {k} = {names[k]}")
                    defs.append(f'-D{k}="{names[k]}"')
    elif a.variant == "08b":
        print("  ⚠️  no --model: using the source's compiled-in OUT_* defaults. If the "
              "export was re-emitted, this THROWS on the first step rather than "
              "returning a wrong answer.")

    sh(["clang++", *CFLAGS, *defs, str(SRC), "-o", str(out_path)])
    print(f"  compiled -> {out_path} ({out_path.stat().st_size} bytes)")

    if not a.no_sign:
        if not PLIST.exists():
            sys.exit(f"⛔ entitlements missing at {PLIST}")
        sh(["codesign", "-f", "-s", "-", "--entitlements", str(PLIST), str(out_path)])
        print(f"  signed with {PLIST.name}")

    print(f"  __text md5 = {text_md5(out_path)}")
    print("    (whole-file md5 is NOT comparable across builds -- fresh LC_UUID per link)")

    if a.rope:
        rope_out = a.rope_out or out_path.parent
        print(f"  RoPE caches at ctx={a.ctx}, theta={a.theta:g}:")
        build_rope(a.ctx, Path(rope_out), Path(a.rope_ref), a.theta)

    print(f"\n  run: sudo -n {out_path} <CDIR> <TRUSTED> "
          f"<embed.mlmodelc> <lmhead.mlmodelc> <cachedir> <units>")
    print(f"  NOTE: <CDIR> must CONTAIN gdnmut_gc0.mlmodelc -- it is the parent dir, "
          f"not the model itself.")


if __name__ == "__main__":
    main()
