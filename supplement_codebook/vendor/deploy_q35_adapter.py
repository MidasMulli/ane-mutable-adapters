#!/usr/bin/env python3
"""Deploy TRAINED LoRA factors into the 0.8B's mutable adapter.bin, by NAME not by index.

`bfmi_inject.py` packed 96 sections of random values and REPOINTED each const at
adapter.bin@offset. Those offsets are now written in the model's own `model.mil`, so this
tool reads them back rather than recomputing a packing — the model is the source of truth.

NAME-KEYED, NOT INDEX-KEYED, ON PURPOSE. The trainer stores factors by enumeration index over
its `wraps` list. Reconstructing that order here would be a second, independent ordering that
could silently drift from the first — the exact class of defect this arc keeps finding. So the
mapping is rebuilt from the SAME construction the trainer used (attn layers in order, each
q_proj then v_proj; then ssm layers in order, each in_proj_qkv then out_proj) and then
CHECKED against the const names parsed out of the MIL. A mismatch aborts.

Emits:
  gc0_on.bin   the trained factors from $PT (the label used to say "legal" regardless of what was deployed)
  gc0_off.bin  B=0 (delta exactly 0) -> the base model, the honest OFF arm
The OFF arm is NOT random. A random OFF arm would make any ON/OFF difference unreadable:
it would show "two different adapters differ", not "the trained adapter does something".
"""
import os
import re
import sys
import json
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("GDN_VARIANT", "0.8B")
# VENDORED 2026-08-30 into supplement_codebook/vendor/. Upstream: gdn-coreml/deploy_q35_adapter.py.
# Only the four absolute paths below are changed; the logic is byte-for-byte upstream.
# `gdn_4b_model.py` is vendored alongside this file, so HERE is the right import root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import struct
import torch
import gdn_4b_model as M

R = 16
ALPHA = 32
SC = ALPHA / R          # must match train_q35_legal.py

# FORM: which LoRA op form this artifact uses. `linear` is the reference build
# (consts `..._l_` / `..._lB`, rank-2, consumed by `linear`); `conv` is the CQ252
# re-emission (consts `..._lA_weight` / `..._lB_weight`, rank-4, consumed by `conv`).
# The trained factors are IDENTICAL in both -- only the const naming and declared rank
# differ, and the bytes are the same because trailing unit dims add no stride.
FORM = os.environ.get("FORM", "linear")
A_TAG, B_TAG = ("_l_", "_lB") if FORM == "linear" else ("_lA_weight", "_lB_weight")
# ⛔ THESE DEFAULTS ARE NO LONGER A DEVELOPER'S TREE, AND THEY ARE NO LONGER SILENT.
# The upstream defaults pointed at an unrelated arc's artifacts (`legal_fp32.pt`), so a caller who
# forgot an env var did not fail -- it built a DIFFERENT adapter and reported success.
_SUP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # supplement_codebook/
MLC = os.environ.get("MLC") or os.path.join(_SUP, "_mlc_shadow", "gdnmut_gc0.mlmodelc")
PT = os.environ.get("PT") or "/tmp/codebook_deploy.pt"
OUTDIR = os.environ.get("OUTDIR") or os.path.join(_SUP, "_adapters")
for _n, _p in (("MLC", MLC), ("PT", PT)):
    if not os.path.exists(_p):
        sys.exit(f"⛔ {_n} does not exist: {_p}\n"
                 f"   deploy_codebook.sh sets MLC/PT/OUTDIR explicitly; if you are calling this\n"
                 f"   script directly, set {_n}= rather than relying on a default.")
os.makedirs(OUTDIR, exist_ok=True)

# CQ165 scale sweep + CQ169 attention-only arm.
#   EPS=<float>  rescale every written factor's delta by eps (folded into A alongside SC)
#   ARM=full|attn|azero|bzero
#     attn  -> SSM sites get B=0 (delta exactly 0), attention sites unchanged
#     azero -> A written as ZEROS, B trained.  delta = (x@0)@B = 0 EXACTLY.
#     bzero -> A trained, B written as ZEROS.  delta = (x@A)@0 = 0 EXACTLY.
#   azero and bzero are BOTH mathematically identical to OFF. They are the three-arm rule
#   applied to the FACTORS rather than to the adapters: if either deviates from OFF on
#   silicon, the graph is not consuming that factor, and WHICH one deviates names the fault.
#   Motivation: the deployed full arm FLOORS at +2.96 nats as eps->0, and eps multiplies
#   ONLY A (line below), so a floor that survives eps->0 cannot come from A.
# NOTE ON CQ169's ARITHMETIC: the attention-only subset is 12 SITES, not 30.
# 6 full-attention layers x {q_proj, v_proj} = 12; 18 linear layers x {in_proj_qkv, out_proj} = 36.
# The 30 is the count of sites whose INPUT is hidden-1024 (12 attn + 18 ssm in_proj_qkv) -- it is
# the A-shape histogram (30x[16,1024] + 18x[16,2048]), not the attention count.
EPS = float(os.environ.get("EPS", "1.0"))
# EPSB scales B INDEPENDENTLY of A. With both tiny the true delta is EPS*EPSB of trained --
# 1e-6 at 0.001/0.001 -- while BOTH consts stay non-zero. That discriminates "the damage is
# value-dependent" (approaches OFF) from "any non-zero content damages" (stays broken), which
# neither azero nor bzero can separate because each leaves one factor at full strength.
EPSB = float(os.environ.get("EPSB", "1.0"))
ARM = os.environ.get("ARM", "full")

CONST_RE = re.compile(
    r'tensor<(\w+),\s*\[([0-9,\s]+)\]>\s+([A-Za-z0-9_]+)\s*=\s*const\(\)\[name\s*=\s*string\("\3"\),'
    r'\s*val\s*=\s*tensor<[^>]+>\(BLOBFILE\(path\s*=\s*string\("([^"]+)"\),\s*offset\s*=\s*uint64\((\d+)\)')


def main():
    mil = open(os.path.join(MLC, "model.mil"), errors="replace").read()
    by_name = {}
    for m in CONST_RE.finditer(mil):
        dt, shape, name, path, off = m.group(1), m.group(2), m.group(3), m.group(4), int(m.group(5))
        if "adapter.bin" not in path:
            continue
        by_name[name] = {"dims": [int(x) for x in shape.split(",")], "offset": off, "dtype": dt}
    print(f"  adapter-backed consts in model: {len(by_name)}")

    # Rebuild the trainer's construction order, then CHECK it against the MIL names.
    order = []
    for i in sorted({int(k) for k in re.findall(r"attn_layers_(\d+)_q_proj" + re.escape(B_TAG), mil)}):
        order += [f"attn_layers_{i}_q_proj", f"attn_layers_{i}_v_proj"]
    for i in sorted({int(k) for k in re.findall(r"ssm_layers_(\d+)_in_proj_qkv" + re.escape(B_TAG), mil)}):
        order += [f"ssm_layers_{i}_in_proj_qkv", f"ssm_layers_{i}_out_proj"]
    print(f"  reconstructed site order: {len(order)} sites "
          f"({M.NUM_ATTN_LAYERS} attn x2 + {M.NUM_SSM_LAYERS} ssm x2)")
    missing = [s for s in order if f"{s}{B_TAG}" not in by_name or f"{s}{A_TAG}" not in by_name]
    if missing:
        raise SystemExit(f"ABORT: {len(missing)} sites have no A/B pair in the MIL: {missing[:3]}")

    ck = torch.load(PT, map_location="cpu")
    fac = ck["A"]
    # ⛔ THE SCALE COMES FROM THE CHECKPOINT, NOT FROM A CONSTANT THAT MUST MATCH ANOTHER FILE.
    # This was `SC = ALPHA/R` with ALPHA=32, R=16 hardcoded at module scope. A checkpoint trained
    # at alpha=16, rank=16 (SC=1.0) would have been deployed at DOUBLE STRENGTH, silently -- the
    # fourth hardcoded-constant mismatch found on 2026-08-15 after the site list, the rank and the
    # op form. If the checkpoint does not carry them, we refuse rather than assume.
    if "alpha" in ck and "rank" in ck:
        SC = float(ck["alpha"]) / float(ck["rank"])
        print(f"  LoRA scale from checkpoint: alpha {ck['alpha']} / rank {ck['rank']} = {SC}")
    else:
        raise SystemExit("ABORT: checkpoint carries no alpha/rank; refusing to assume a scale")
    print(f"  loaded {PT}\n    sites={ck.get('sites')} steps={ck.get('steps')} tw={ck.get('tw')} "
          f"CE {ck.get('heldout_ce_before'):.4f} -> {ck.get('heldout_ce_after'):.4f}")
    if len(fac) != len(order):
        raise SystemExit(f"ABORT: trainer has {len(fac)} sites, model has {len(order)}")
    # SITE-ORDER ASSERT. `order` above is RECONSTRUCTED from the MIL with indices sorted as
    # INTS; the trainer's factor list follows ModuleDict iteration over STRING keys, where
    # '11' < '3'. They agree only by insertion-order luck, and 30 of the 48 sites share
    # din=1024, so a permutation inside that group passes the per-tag shape assert SILENTLY
    # and ships correct bytes to the WRONG SITE. The 96-section byte audit cannot catch it --
    # that audit indexed the checkpoint with THIS SAME reconstruction, so audit and suspect
    # shared the mapping (circular). The trainer now persists `site_order`; assert against it.
    trained_order = ck.get("site_order")
    if trained_order is None:
        raise SystemExit(
            "ABORT: checkpoint has no `site_order`. The trainer's factor-to-site mapping "
            "cannot be verified against the MIL reconstruction, and a within-din=1024 "
            "permutation is silent. Re-train (train_q35_legal.py now writes it) or backfill "
            "it from the model's ModuleDict iteration order.")
    if list(trained_order) != list(order):
        bad = [(j, a, b) for j, (a, b) in enumerate(zip(trained_order, order)) if a != b]
        raise SystemExit(f"ABORT: trainer site order != MIL reconstruction at {len(bad)} "
                         f"positions, first {bad[:3]}. Correct bytes would go to wrong sites.")
    print(f"  site-order assert PASS: {len(order)} sites, trainer order == MIL reconstruction")

    base = bytearray(open(os.path.join(MLC, "weights", "adapter.bin"), "rb").read())
    on = bytearray(base)
    off = bytearray(base)
    wrote = 0
    for j, site in enumerate(order):
        A, B = fac[j]                       # A [R,din], B [dout,R], fp16
        # FOLD THE LoRA SCALE INTO A. The TRAINER's forward is
        #     base(x) + SC*(x@A.T)@B.T      with SC = ALPHA/R = 2.0
        # the DEPLOYED graph (gdn4b_lora_convert.py:14) is
        #     base(x) + (x@lA.T)@lB.T       -- NO scalar
        # so shipping raw factors deploys a HALF-STRENGTH adapter. deploy_lora.py's own
        # docstring states the rule: "the LoRA scale alpha/r is folded into A at pack time,
        # so the forward carries no scalar -- otherwise coremltools' fuse_conv_scale absorbs
        # it into B and renames the const." Folding into A (not B) keeps that property.
        A = (A.float() * SC * EPS).half()
        B = (B.float() * EPSB).half() if EPSB != 1.0 else B
        ssm_site = site.startswith('ssm_layers_')
        mute = (ARM == 'attn' and ssm_site)
        for tag, arr in ((f"{site}{A_TAG}", A), (f"{site}{B_TAG}", B)):
            c = by_name[tag]
            want = tuple(c["dims"])
            # conv consts carry trailing unit dims ([R,din,1,1] vs [R,din]). Same bytes, same
            # order -- reshape rather than reject, but keep the element-count assert below.
            if len(want) == 4 and len(arr.shape) == 2 and want[:2] == tuple(arr.shape):
                arr = arr.reshape(want)
            got = tuple(arr.shape)
            if want != got:
                raise SystemExit(f"ABORT shape mismatch {tag}: model {want} vs trained {got}")
            # THE MIL OFFSET POINTS AT A MILBlob METADATA HEADER, NOT AT THE DATA.
            #   sentinel u32 (0xdeadbeef) | dtype u32 | sizeBytes u64 | dataOffset u64
            # gdn_deploy_mutable.py:155 states it and warns that clobbering it gives espresso
            # "Cannot retrieve file blob properties" and a -14 -- the ROOT CAUSE OF THE GDN -14.
            # v1 wrote factor data straight to the MIL offset, i.e. over all 96 headers. That,
            # not the adapter, is why the B-zeroed OFF arm read 5.78 / 16.56 / 15.06 when it is
            # mathematically pinned to the base: the blobs were structurally corrupt.
            o = c["offset"]
            sent, _dt = struct.unpack_from("<II", bytes(base), o)
            if sent != 0xDEADBEEF:
                raise SystemExit(f"ABORT {tag}: no 0xdeadbeef sentinel at {o} (got 0x{sent:08x})")
            szb, doff = struct.unpack_from("<QQ", bytes(base), o + 8)
            buf = arr.contiguous().numpy().astype(np.float16).tobytes()
            if len(buf) != szb:
                raise SystemExit(f"ABORT {tag}: header says {szb} B, factor is {len(buf)} B")
            # azero_attn / azero_ssm: A=0 EVERYWHERE and B trained in ONE family only.
            # Still pinned to delta=0 by arithmetic (A=0 => the composed product is 0), so any
            # deviation localises the phantom-B effect to a site family. This is the bisection
            # step after CQ221 refuted folding: lB changes the output while its MIL input is
            # provably zero, so WHICH lB consts do it is the next fact.
            # ARM=azonly:<substr> -- A=0 everywhere, B kept ONLY at sites whose name contains
            # <substr>. Generalises the family bisect to any subset without new ARM names.
            az_only = ARM.startswith('azonly:')
            # bzonly:<substr> -- B=0 EVERYWHERE, A trained ONLY at matching sites. The mirror
            # of azonly, localising the second phantom: bzero pins delta=0 via B=0 and still
            # deviates (+10.67), so which lA consts cause that is the other open fact.
            bz_only = ARM.startswith('bzonly:')
            # only:<substr> -- BOTH factors trained at matching sites, BOTH zeroed elsewhere.
            # This is NOT delta=0; it is the correct adapter restricted to one family. Compared
            # against azonly/bzonly on the same family it answers whether the CHAIN is ever
            # computed: if both-together == one-alone, the other const contributes NOTHING even
            # when its partner is present, and (x@A)@B is never formed.
            bo_only = ARM.startswith('only:')
            az_fam = ARM in ('azero_attn', 'azero_ssm')
            is_b = tag.endswith(B_TAG)
            is_a = tag.endswith(A_TAG)
            zero_this = (mute and is_b) \
                or ((az_only or ARM in ('azero', 'azero_attn', 'azero_ssm'))
                    and is_a) \
                or (ARM == 'bzero' and is_b) \
                or (bz_only and (is_b or
                                 ARM.split(':', 1)[1] not in site)) \
                or (az_fam and is_b and
                    (ssm_site if ARM == 'azero_attn' else not ssm_site)) \
                or (az_only and is_b and ARM.split(':', 1)[1] not in site) \
                or (bo_only and ARM.split(':', 1)[1] not in site)
            on[doff:doff + len(buf)] = (b'\x00' * szb) if zero_this else buf
            # OFF arm: zero the DATA REGION ONLY, both factors -- matching the proven
            # gdn_deploy_mutable.py construction (it zeroes BOTH A and B, not just B).
            off[doff:doff + szb] = b"\x00" * szb
            wrote += 1

    os.makedirs(OUTDIR, exist_ok=True)
    open(os.path.join(OUTDIR, "gc0_on.bin"), "wb").write(bytes(on))
    open(os.path.join(OUTDIR, "gc0_off.bin"), "wb").write(bytes(off))
    assert on != off, "ASSERT FAIL: on and off blobs identical"
    assert len(on) == len(base) == len(off), "ASSERT FAIL: blob size changed"
    print(f"  wrote {wrote} factor blocks into {len(base)/1e6:.2f} MB adapters  "
          f"[EPS={EPS} EPSB={EPSB} ARM={ARM} FORM={FORM}]")
    print(f"  gc0_on.bin  = the TRAINED factors from {os.path.basename(PT)}")
    print(f"  gc0_off.bin = B zeroed (delta 0 -> base). NOT random: an honest OFF arm.")
    json.dump({"sites": len(order), "blocks": wrote, "bytes": len(base),
               "ce_before": ck.get("heldout_ce_before"), "ce_after": ck.get("heldout_ce_after")},
              open(os.path.join(OUTDIR, "deploy_result.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
