"""END TO END: an image file in, a caption out, both halves on the ANE.

  PIL image
    -> shipped Qwen processor            (256x256 -> 16x16 patches -> [256,1536])
    -> VISION TOWER on the ANE           (entitled bind, chosen merger arm) -> [64,1024]
    -> spliced into the token sequence   (positions of <|image_pad|>, from the processor)
    -> 0.8B TEXT STACK on the ANE        (GENI, stop ids by name, term reported)
    -> caption

Nothing here is precomputed: the geometry, the image positions and the stop ids are all read
off the shipped processor/tokenizer per image, so a differently-shaped image fails an assert
rather than silently producing a mis-spliced sequence.
"""
import os, sys, json, subprocess, time, numpy as np, glob
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
# VENDORED 2026-08-30 into supplement_codebook/vendor/tower/. Upstream:
# ane-mutable-adapters/vision/tower/ane_vlm.py. Only the two paths below are changed.
#
# ⛔ THE HF SNAPSHOT GLOB USED TO BE `[0]` ON A HARDCODED HOME DIRECTORY. On any other machine
# that raised IndexError from an import-time line, which reads as "the module is broken" rather
# than "you have not downloaded the base model". Both inputs are now env-overridable and both
# say what is missing and how to get it.
_HF = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
_snaps = sorted(glob.glob(os.environ.get("Q35_SNAPSHOT_GLOB")
                          or os.path.join(_HF, "hub/models--Qwen--Qwen3.5-0.8B/snapshots/*")))
if not _snaps:
    raise SystemExit(
        "⛔ Qwen3.5-0.8B snapshot not found under " + _HF + "/hub/models--Qwen--Qwen3.5-0.8B/\n"
        "   Fetch it once:  huggingface-cli download Qwen/Qwen3.5-0.8B\n"
        "   Or set Q35_SNAPSHOT_GLOB=/path/to/snapshots/*")
SNAP = os.environ.get("Q35_SNAPSHOT") or _snaps[-1]

# The compiled-artifact + signed-harness directory. See README 'Bring-up': build_gdn_harness.py
# produces it. It is NOT shipped (≈1 GB of compiled .mlmodelc plus an entitlement-signed binary).
DEMO = os.environ.get("Q35_DEMO_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_demo")
if not os.path.isdir(DEMO):
    raise SystemExit(
        f"⛔ bind harness directory not found: {DEMO}\n"
        f"   This holds the compiled embed/lmhead .mlmodelc and the signed q35_swap_img_nb binary.\n"
        f"   Build it with vendor/harness/build_gdn_harness.py (README 'Bring-up'), or set\n"
        f"   Q35_DEMO_DIR= if you already have one.")
# ⛔ AN UNSHIPPED DEPENDENCY THAT ONLY RUNNING THE THING FOUND. `{HERE}/../assets/conv_swap_probe`
# is a compiled Mach-O probe that no static read of this file flagged as an artifact -- it looked
# like an ordinary relative path, and it broke the moment this module was vendored one level
# deeper. It is a build product of the same class as the bind harness, not a source file.
PROBE = os.environ.get("Q35_SWAP_PROBE") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "conv_swap_probe")
if not os.path.exists(PROBE):
    _alt = os.path.join(DEMO, "conv_swap_probe")
    if os.path.exists(_alt):
        PROBE = _alt
    else:
        raise SystemExit(
            f"⛔ conv_swap_probe not found: {PROBE}\n"
            f"   A compiled probe binary the vision tower shells out to. Build it during bring-up\n"
            f"   (README 'Bring-up') or set Q35_SWAP_PROBE=/path/to/conv_swap_probe.")
VMDL = "/Library/Caches/com.apple.aned/qvis.mlmodelc"
VTRUST = "/private/var/db/AppleIntelligencePlatform/AppModelAssets/research/qvis"
KEY = "@model_path/weights/adapter.bin"
CDIR = "/Library/Caches/com.apple.aned/q35_08b_conv"
CTRUST = "/private/var/db/AppleIntelligencePlatform/AppModelAssets/q35_08b_conv"

import warnings; warnings.filterwarnings("ignore")
from transformers import AutoProcessor, AutoConfig, AutoTokenizer
PROC = AutoProcessor.from_pretrained(SNAP)
CFG = AutoConfig.from_pretrained(SNAP)
TOK = AutoTokenizer.from_pretrained(SNAP)
STOPS = [i for i in (TOK.convert_tokens_to_ids("<|im_end|>"), TOK.convert_tokens_to_ids("<|endoftext|>"))
         if i is not None and i != TOK.unk_token_id]
assert STOPS, "no stop ids"
SIDE = int(os.environ.get('VLM_SIDE', '256'))   # processor floor; VLM_SIDE=384 -> 24x24 grid -> 144 merged tokens
#   ⛔ changing this changes the patch grid AND requires a re-exported tower (convert_vision.py);
#   default stays 256 so every banked result reproduces byte-identically.


def fit_square(img, side=None, mode="letterbox", bg=(255, 255, 255)):
    """⛔ THE OLD PATH STRETCHED TO SQUARE AND THAT WAS DESTROYING TEXT.

    A 494x195 screenshot was squashed 1.9x horizontally while being STRETCHED 1.3x vertically
    before the tower ever saw it. Measured on a real HuggingFace screenshot, same model, same
    prompt, only this line different:
        stretched   -> "mvidia models updated by hauguro000"   (header only, garbled)
        letterboxed -> "nvidia/DeepSeek-V4-Pro-...", "nvidia/DeepSeek-V4-Flash-..."  (both names)
    LANCZOS rather than BICUBIC because this is a DOWNSCALE and text is the payload.
    """
    side = side or SIDE
    if mode == "stretch":
        return img.convert("RGB").resize((side, side), Image.BICUBIC)
    img = img.convert("RGB")
    w, h = img.size
    sc = side / max(w, h)
    nw, nh = max(1, round(w * sc)), max(1, round(h * sc))
    out = Image.new("RGB", (side, side), bg)
    out.paste(img.resize((nw, nh), Image.LANCZOS), ((side - nw) // 2, (side - nh) // 2))
    return out


def tiles(img, side=None, overlap=32):
    """NATIVE-resolution windows. Kept for the dense-crop case; auto_caption uses grid_tiles."""
    side = side or SIDE
    img = img.convert("RGB"); W, H = img.size
    if max(W, H) <= side: return [fit_square(img, side)]
    step = side - overlap
    out = []
    for y in range(0, max(1, H - overlap), step):
        for x in range(0, max(1, W - overlap), step):
            out.append(fit_square(img.crop((x, y, min(x + side, W), min(y + side, H))), side))
    return out


def grid_tiles(img, n, overlap_frac=0.08, ink_thr=0.005):
    """An n x n grid, each cell letterboxed to 256 -- so effective resolution rises n-fold.

    ⛔ WHY NOT NATIVE 256-PX WINDOWS, WHICH IS WHAT CC SHIPPED FIRST AND WHICH FAILED:
    a 1172x795 document produced 24 native windows; auto_caption took the first 6 IN RASTER ORDER,
    THREE OF WHICH WERE BLANK MARGINS (0.00% ink, measured). Half the ANE passes were spent on
    white space and the body of the document was never reached -- the answer came back as "the
    ISDA logo". The need is ENOUGH resolution, not maximum resolution.

    BLANK CELLS ARE SKIPPED. They cost a 2 s ANE pass each and return "this image is completely
    blank", which then dilutes the merge.
    """
    import numpy as _np
    img = img.convert("RGB"); W, H = img.size
    ox, oy = int(W / n * overlap_frac), int(H / n * overlap_frac)
    out = []
    for r in range(n):
        for c in range(n):
            x0 = max(0, int(c * W / n) - ox); x1 = min(W, int((c + 1) * W / n) + ox)
            y0 = max(0, int(r * H / n) - oy); y1 = min(H, int((r + 1) * H / n) + oy)
            cell = img.crop((x0, y0, x1, y1))
            ink = float((_np.asarray(cell.convert("L"), _np.float32) < 200).mean())
            if ink < ink_thr: continue                  # blank: skip, do not spend a pass
            out.append(((r, c), fit_square(cell, SIDE)))   # reading order preserved by construction
    return out


def choose_grid(img, thr=0.45, cap=4):
    """Smallest n whose cells individually survive the downscale."""
    W, H = img.size
    for n in range(2, cap + 1):
        cell = img.crop((0, 0, W // n, H // n))
        if detail_loss(cell) <= thr: return n
    return cap


def _flatten_alpha(img):
    """⛔ COMPOSITE ALPHA ONTO WHITE BEFORE ANYTHING ELSE. `.convert("RGB")` maps TRANSPARENT to
    BLACK; a black glyph on a now-black field is a uniform black square and the letter is GONE.
    MEASURED: 27 of the 41 sourced tier-3 images carry alpha and 21 go near-solid-black under the
    naive call. Fixed in `eval_glyph2.py` on 2026-08-16 and NOT propagated here, so the live web
    path was feeding the model black rectangles -- the base arm dutifully described one."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, img).convert("RGB")
    return img.convert("RGB")


def load_rgb(path):
    """⛔ THE ONE WAY TO OPEN AN IMAGE FOR THIS MODEL. Use this, never `Image.open(p).convert("RGB")`.
    MEASURED on the 41 sourced tier-3 images: 25 carry alpha and 20 go NEAR-SOLID-BLACK under the
    naive call. Fixing `encode()` alone did NOT help, because seven eval/probe callers flattened the
    image BEFORE handing it over -- the damage was already done at the call site and the on-ANE
    tier-3 score came back byte-identical (0.634), which is what exposed it."""
    return _flatten_alpha(Image.open(path))


def encode(img, prompt="Describe this image in one sentence.", fit="letterbox"):
    img = fit_square(_flatten_alpha(img), SIDE, mode=fit)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = PROC.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = PROC(text=[text], images=[img], return_tensors="pt")
    grid = enc["image_grid_thw"]
    assert tuple(grid[0].tolist()) == (1, SIDE//16, SIDE//16), f"unexpected grid {grid.tolist()}"
    ids = enc["input_ids"][0].numpy()
    pos = np.nonzero(ids == CFG.image_token_id)[0]
    assert len(pos) and list(pos) == list(range(pos[0], pos[-1]+1)), "image positions not contiguous"
    return enc["pixel_values"].numpy().astype(np.float32), ids, int(pos[0]), len(pos)


def tower(px, arm="BASE"):
    """One entitled bind of the vision tower on the chosen merger arm.

    TWO TRAPS, both live:
      1. `sudo` SANITISES THE ENVIRONMENT. Passing PROBE_INPUT/PROBE_OUT via subprocess `env=`
         does NOT reach the binary -- the probe then falls back to its SYNTHETIC ramp input and
         still prints `PREDICT OK`. The vars must go through an explicit `env` argv after sudo.
      2. A STALE OUTPUT FILE READS AS A FRESH RESULT. Because of (1) the probe can succeed while
         writing nothing, so the previous image's vectors would be captioned as this one's.
         The output is DELETED before the call and its existence+size asserted after.
    """
    fin, fout = f"{HERE}/_vlm_in.f32", f"{HERE}/_vlm_out.f32"
    px.tofile(fin)
    if os.path.exists(fout): os.remove(fout)          # trap 2
    r = subprocess.run(["/usr/bin/sudo", "-n", "env", f"PROBE_INPUT={fin}", f"PROBE_OUT={fout}",
                        PROBE, VMDL, "3", "1", f"{VTRUST}/{arm}.bin", KEY],
                       capture_output=True, text=True)                       # trap 1
    if "PREDICT OK" not in r.stdout:
        raise RuntimeError("tower failed: " + (r.stdout + r.stderr)[-300:])
    assert f"INPUT {fin}" in r.stdout, "probe did NOT read our input -- env did not reach it"
    assert os.path.exists(fout), "probe wrote no output"
    v = np.fromfile(fout, np.float32)
    assert v.size == 64 * 1024, f"tower output {v.size} != 65536"
    return v


def decode_lm(vecs, ids, p0, nimg, lm_arm, ngen=160):
    """`lm_arm` IS REQUIRED AND HAS NO DEFAULT, AND THAT IS THE WHOLE POINT.

    ⛔ THIS SLOT WAS A HARDCODED `0` AND `0` MEANS `off`. MEASURED, not inferred: driving the
    binary with a bare `GENX <arm> ...` prints `GENXBEGIN start=off` for 0 and `start=on` for 1,
    and the two arms emit COMPLETELY DIFFERENT tokens. So every caption this module ever
    produced ran the TEXT DECODER ON THE OFF ARM -- the base model -- and a freshly deployed
    text adapter was inert through this path with no error anywhere.
    That is exactly how the Syriac adapter read 0/6 with garbage output (four wrong scripts, no
    abstention) while scoring 1.000 letters in torch: the arm it was graded on was never `on`.

    THE TOWER ARM WAS NEVER AFFECTED -- `tower(px, arm)` always received its arm, so the
    merger-swap dissociation results stand. It was only ever the text-decoder adapter that
    could not be selected, because nothing could express the selection.

    No default, so a caller that has not thought about which arm it wants gets a TypeError
    instead of silently getting the base model. Rule 44: a rule in a docstring is
    documentation, a required argument is a gate.
    """
    assert lm_arm in ("on", "off"), f"lm_arm must be 'on' or 'off', got {lm_arm!r}"
    f = f"{HERE}/_vlm_emb.f32"; vecs.astype(np.float32).tofile(f)
    stoparg = f"{len(STOPS)} " + " ".join(map(str, STOPS))
    cmd = (f"GENI {f} {p0} {nimg} {1 if lm_arm == 'on' else 0} {len(ids)} {ngen} {stoparg} "
           + " ".join(map(str, ids)))
    # ⛔ NEVER `stderr=DEVNULL` HERE. The bind's whole diagnostic channel comes out on stderr:
    # `Too many fvmlibs (more than 255)` (SILENT -- espresso rebuilds a NON-mutable plan and the
    # swap becomes a no-op), `-14` plan-build failures, entitlement gaps. Discarding it removes
    # the evidence rather than misplacing it, and a suppressed failure reads exactly like a
    # result. It is kept on disk so the fvmlib grep can actually run.
    errlog = open(f"{HERE}/.decode_lm_{lm_arm}.err", "w")
    p = subprocess.Popen(["/usr/bin/sudo", "-n", "./q35_swap_img", CDIR, CTRUST,
                          "./embed.mlmodelc", "./lmhead.mlmodelc", ".", "3"],
                         cwd=DEMO, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=errlog, text=True, bufsize=1)
    p.stdout.readline()
    p.stdin.write(cmd + "\n"); p.stdin.flush()
    toks, term, started = [], "?", None
    while True:
        l = p.stdout.readline()
        if not l: break
        # THE ARM THE BINARY ACTUALLY BOUND, read back from its own banner rather than assumed
        # from what we passed. This is the manipulation-took check for the swap (Rule 32): the
        # request and the bind are two different things, and only one of them is evidence.
        if l.startswith("GENIBEGIN") and "start=" in l:
            started = l.split("start=")[1].split()[0].strip()
            if "bind_ms=" in l:                    # the binary's own MUTABLE-SWAP time, in ms
                try: globals()["last_bind_ms"] = float(l.split("bind_ms=")[1].split()[0])
                except Exception: pass
        if l.startswith("TOK "): toks.append(int(l.split()[1]))
        if l.startswith("GENIDONE"):
            term = l.split("term=")[1].strip() if "term=" in l else "?"
            break
    try: p.stdin.close(); p.terminate()
    except Exception: pass
    errlog.close()
    assert started == lm_arm, \
        f"ARM DID NOT TAKE: asked for {lm_arm!r}, binary bound {started!r} -- refusing to grade"
    # The overflow is SILENT: the mutable compile fails, espresso rebuilds a NON-mutable plan,
    # the adapter is read STATICALLY and the swap becomes a no-op with NO error anywhere.
    err = open(f"{HERE}/.decode_lm_{lm_arm}.err", errors="replace").read()
    assert "Too many fvmlibs" not in err, \
        "fvmlib COUNT OVERFLOW: the mutable compile fell back to a non-mutable plan; swap is a no-op"
    return TOK.decode(toks, skip_special_tokens=True).strip(), term, len(toks)


def caption(path, arm="BASE", prompt="Describe this image in one sentence.", ngen=160,
            fit="letterbox", lm_arm="off"):
    # ngen was 48 and every free-form description hit it -- the operator's first real
    # photo came back truncated mid-clause with term="cap". The prompt is ~85 tokens
    # against a 4096 window, so the cap was the only thing binding.
    #
    # TWO INDEPENDENT ARMS, and conflating them cost a full deploy investigation:
    #   `arm`     selects the VISION TOWER merger (BASE/A/B)  -- always worked
    #   `lm_arm`  selects the TEXT DECODER adapter (on/off)   -- was UNREACHABLE, pinned off
    # `lm_arm` defaults to "off" ONLY so the tower-merger results stay bit-reproducible; it is
    # explicit at every call site that grades a text adapter.
    px, ids, p0, nimg = encode(Image.open(path), prompt, fit=fit)
    t0 = time.time(); v = tower(px, arm); t_tower = (time.time()-t0)*1000
    t0 = time.time()
    txt, term, n = decode_lm(v, ids, p0, nimg, lm_arm, ngen=ngen); t_lm = (time.time()-t0)*1000
    return {"text": txt, "term": term, "tokens": n, "arm": arm, "lm_arm": lm_arm,
            "tower_ms": round(t_tower, 1), "lm_ms": round(t_lm, 1),
            "prompt_tokens": len(ids), "image_positions": nimg}


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = caption(p)
        print(f"  {os.path.basename(p):22s} [{r['term']}, {r['tokens']} tok, "
              f"tower {r['tower_ms']:.0f}ms + lm {r['lm_ms']:.0f}ms]  {r['text']!r}")

# ---------------------------------------------------------------------------------------------
# AUTO PATH: the user drops an arbitrary image and the system decides. A manual mode flag is a
# lab control, not a product.
#
# TWO SIGNALS, GATED, BECAUSE NEITHER WORKS ALONE (both measured on real dropped images):
#   detail_loss  edge energy destroyed by the 256x256 downscale. Catches density -- but a dog
#                photo scores 0.63 on fur and sand and needs no tiling at all.
#   describes-text  one cheap whole-image pass, keyword-matched. Separates TEXT from TEXTURE,
#                which edge energy cannot. Asked as yes/no it fails on code (5/7); asked as a
#                DESCRIPTION, its natural mode, it succeeds (5/6).
# Tiling fires only when BOTH agree, so photos pay one extra pass and nothing else.
# ---------------------------------------------------------------------------------------------
import re as _re
from PIL import ImageFilter as _IF

_TEXTY = _re.compile(r"\b(text|code|word|letter|character|caption|label|number|digit|chart|graph|"
                     r"table|document|screenshot|writing|written|font|script|snippet|typed)\w*", _re.I)


def detail_loss(img):
    """Fraction of edge energy destroyed by the round trip through 256x256."""
    g = img.convert("L"); W, H = g.size
    if max(W, H) <= SIDE: return 0.0
    small = g.resize((SIDE, SIDE), Image.LANCZOS).resize((W, H), Image.LANCZOS)
    ef = np.asarray(g.filter(_IF.FIND_EDGES), np.float32).mean()
    es = np.asarray(small.filter(_IF.FIND_EDGES), np.float32).mean()
    return float(1 - es / max(ef, 1e-6))



def textiness(img, W=1024):
    """Statistical text detector: no model call, no keyword list.

    ⛔ TWO EARLIER DETECTORS FAILED AND ARE RECORDED SO THEY ARE NOT RETRIED:
      keyword-match on a description -- the 10-K described itself as "a financial statement
        showing financial data", which matches NONE of text|code|table|chart|document. A list
        built from six examples did not generalise to the seventh, and the answer came back as
        "no change in net product sales" against a real change of +1,114M.
      ask the model YES/NO -- 3/6 and 4/6 across two framings. A 0.8B is not a reliable
        single-token classifier.
    Text lays ink down in REGULAR HORIZONTAL BANDS separated by gaps; photographs do not.
    Measured on 8 real images: text 0.205-3.206, photos 0.030-0.032. A 6.4x gap.
    """
    import numpy as _np
    g = img.convert("L")
    w = W; h = max(1, int(g.size[1] * W / g.size[0]))
    a = _np.asarray(g.resize((w, h), Image.LANCZOS), _np.float32)
    ink = (a < a.mean() - 0.4 * a.std()).astype(_np.float32)   # adaptive to paper tone
    row = ink.mean(1)
    if row.mean() < 1e-4: return 0.0
    m = row.mean()
    cross = int(_np.sum(_np.diff((row > m).astype(int)) != 0))
    return round(cross * float(row.std() / (m + 1e-6)) / 100.0, 3)


def auto_caption(path, arm="BASE", prompt="Describe this image in one sentence.",
                 ngen=160, loss_thr=0.45, max_tiles=6, lm_arm="off"):
    """Decide whole-image vs native tiles, then answer. Returns the decision alongside the text.

    `lm_arm` selects the TEXT-DECODER adapter and is threaded into EVERY caption() path below,
    including the per-tile loop. A tiled answer that silently dropped back to the base model on
    the tiles would look like "the adapter helps on whole images but not on documents", which is
    a plausible-sounding conclusion about the ADAPTER drawn from a hole in the PLUMBING.
    """
    img = Image.open(path)
    dl = detail_loss(img)
    plan = {"detail_loss": round(dl, 3), "size": list(img.size)}
    if dl <= loss_thr:
        r = caption(path, arm=arm, prompt=prompt, ngen=ngen, lm_arm=lm_arm)
        return {**r, **plan, "mode": "whole", "reason": "detail survives the downscale"}

    tx = textiness(img)
    plan["textiness"] = tx
    if tx < 0.12:                       # measured separation: photos <=0.032, text >=0.205
        r = caption(path, arm=arm, prompt=prompt, ngen=ngen, lm_arm=lm_arm)
        return {**r, **plan, "mode": "whole", "reason": f"dense but not text-like ({tx})"}

    n = choose_grid(img)
    cells = grid_tiles(img, n)
    parts = []
    import tempfile, os as _os
    for (r, c), t in cells:
        f = _os.path.join(tempfile.gettempdir(), f"_aneTile{r}{c}.png"); t.save(f)
        parts.append(caption(f, arm=arm, prompt=prompt, ngen=max(60, ngen // 2),
                             lm_arm=lm_arm)["text"].strip())
    return {"text": stitch(parts), "tiles_raw": [f"[{r+1},{c+1}] {p}" for ((r, c), _), p in zip(cells, parts)],
            "term": "tiled", "tokens": sum(len(p.split()) for p in parts), "arm": arm,
            "lm_arm": lm_arm,
            "tower_ms": 0.0, "lm_ms": 0.0, "prompt_tokens": 0, "image_positions": 64,
            **plan, "mode": f"grid {n}x{n}, {len(cells)} non-blank",
            "reason": f"detail lost and text-like ({tx})"}

def stitch(parts, min_overlap=6):
    """Merge per-tile transcriptions, removing the duplication the overlap margin creates.

    WHY THIS IS OWED. Grid cells overlap by 8% so nothing is lost at a boundary -- but that means
    the same text is read twice. On a 10-K, `[1,2] 241` and `[1,3] 241,787` are one number caught
    in two cells. Worse, on the OCR chart a cell that saw only part of the axis FABRICATED "2018",
    and the assembler combined that invention with a real "2031" into "2018 to 2031" when the
    correct answer, present in two other cells, was "2026 to 2031". Duplication is not just noise:
    it gives fabricated fragments equal standing with real ones.

    Two passes, both conservative -- this DEDUPLICATES, it never invents a join:
      1. suffix/prefix overlap between consecutive cells in reading order
      2. drop a line that is a strict substring of a line already kept
    """
    kept = []
    for txt in parts:
        for line in (l.strip() for l in txt.splitlines()):
            if not line: continue
            if any(line in k and line != k for k in kept):   # strictly contained already
                continue
            merged = False
            for i, k in enumerate(kept):
                if k in line:                                # this line supersedes a fragment
                    kept[i] = line; merged = True; break
                for n in range(min(len(k), len(line)), min_overlap - 1, -1):
                    if k[-n:] == line[:n]:                   # tail of k == head of line
                        kept[i] = k + line[n:]; merged = True; break
                if merged: break
            if not merged: kept.append(line)
    return "\n".join(kept)
