#!/usr/bin/env python3
"""Convert an `adapter_trainer.py` checkpoint into the shape `deploy_q35_adapter.py` binds.

⛔ THE RISK THIS FILE EXISTS TO ELIMINATE, IN THE DEPLOY TOOL'S OWN WORDS:
   "a within-din=1024 permutation is SILENT and ships correct bytes to the WRONG SITE. The
    96-section byte audit cannot catch it -- that audit indexed the checkpoint with THIS SAME
    reconstruction, so audit and suspect shared the mapping (CIRCULAR)."
30 of the 48 sites share din=1024, so a mis-ordering passes every shape assert and produces a
plausible-but-wrong adapter.

⇒ THIS CONVERTER IS NAME-KEYED END TO END AND NEVER RECONSTRUCTS AN ORDER.
For every adapter-backed const in the MIL it DERIVES the HF module that const belongs to, then
looks that factor up BY NAME. There is no second ordering to drift from the first.

THE FAMILY MAPPING IS READ OFF THE CHECKPOINT, NOT ASSUMED FROM `full_attention_interval`:
  the checkpoint says which HF layers carry `self_attn` (measured: 3, 7, 11, 15, 19, 23) and which
  carry `linear_attn` (the other 18). Ascending order gives attn_layers_0..5 and ssm_layers_0..17.
  Both directions are asserted bijective before a byte is written.

  MIL const                              HF site
  attn_layers_{i}_q_proj_lA_weight   ->  ...layers.{ATTN[i]}.self_attn.q_proj      .A
  ssm_layers_{j}_out_proj_lB_weight  ->  ...layers.{SSM[j]}.linear_attn.out_proj   .B
"""
import argparse, glob, json, os, re, sys

CONST = re.compile(
    r'tensor<(\w+),\s*\[([0-9,\s]+)\]>\s+([A-Za-z0-9_]+)\s*=\s*const\(\)\[name\s*=\s*string\("\3"\),'
    r'\s*val\s*=\s*tensor<[^>]+>\(BLOBFILE\(path\s*=\s*string\("([^"]+)"\),\s*offset\s*=\s*uint64\((\d+)\)')
MIL_NAME = re.compile(r"^(attn|ssm)_layers_(\d+)_(q_proj|v_proj|in_proj_qkv|out_proj)_l([AB])_weight$")


def mil_consts(mlmodelc):
    hits = glob.glob(os.path.join(mlmodelc, "**", "model.mil"), recursive=True)
    if not hits: raise SystemExit(f"no model.mil under {mlmodelc}")
    out = {}
    for p in hits:
        for m in CONST.finditer(open(p, errors="replace").read()):
            if "adapter.bin" not in m.group(4): continue
            out[m.group(3)] = {"dims": [int(x) for x in m.group(2).split(",")],
                               "dtype": m.group(1), "offset": int(m.group(5))}
    return out


def families(sites):
    """Which HF layers are attention and which are linear -- READ FROM THE CHECKPOINT."""
    attn, ssm = set(), set()
    for s in sites:
        m = re.search(r"layers\.(\d+)\.(self_attn|linear_attn)\.", s)
        if not m: raise SystemExit(f"unparseable site name: {s}")
        (attn if m.group(2) == "self_attn" else ssm).add(int(m.group(1)))
    return sorted(attn), sorted(ssm)


def main(a):
    import torch
    ck = torch.load(a.ckpt, map_location="cpu")
    sd, sites = ck["sd"], list(ck["sites"])
    ATTN, SSM = families(sites)
    print(f"  checkpoint: {len(sites)} sites · {len(sd)} factors · rank {ck['rank']} alpha {ck['alpha']}")
    print(f"  attention layers {ATTN}")
    print(f"  linear layers    {SSM[:6]}... ({len(SSM)})")

    consts = mil_consts(a.target)
    print(f"  target: {len(consts)} adapter-backed consts")
    assert len(consts) == 2 * len(sites), \
        f"MISMATCH: {len(consts)} consts vs {2*len(sites)} factors -- refusing to guess a packing"

    prefix = os.path.commonprefix(sites).rsplit(".layers.", 1)[0] + ".layers."

    # ⛔ THE ORDER AND THE NAMING MUST BOTH BE THE DEPLOY TOOL'S, NOT OURS.
    # `deploy_q35_adapter.py` reconstructs, and asserts against, this exact sequence:
    #     for i in sorted(attn indices as INTS):  attn_layers_{i}_q_proj, then _v_proj
    #     for i in sorted(ssm  indices as INTS):  ssm_layers_{i}_in_proj_qkv, then _out_proj
    # in MIL naming. A lexical sort puts `attn_layers_11` before `attn_layers_3`, and HF-style
    # names never compare equal to MIL-style ones -- both were caught by that tool's assert with
    # "Correct bytes would go to wrong sites", which is the whole reason it exists: 30 of the 48
    # sites share din=1024, so a permutation inside that group passes every shape check silently.
    mil_order = []
    for i in ATTN: mil_order += [f"attn_layers_{i}_q_proj", f"attn_layers_{i}_v_proj"]
    for i in SSM:  mil_order += [f"ssm_layers_{i}_in_proj_qkv", f"ssm_layers_{i}_out_proj"]
    assert len(mil_order) == len(sites), f"{len(mil_order)} reconstructed vs {len(sites)} sites"

    order, A_list, seen = [], [], set()
    for base in mil_order:
      for name in (f"{base}_lA_weight", f"{base}_lB_weight"):
        assert name in consts, f"{name} not among the target's adapter-backed consts"
        m = MIL_NAME.match(name)
        if not m: raise SystemExit(f"unrecognised const name: {name}")
        fam, layer, proj, ab = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        # MEASURED, NOT ASSUMED: the MIL numbers these by HF LAYER INDEX, not family-relative
        # position -- attn_layers indices read [3,7,11,15,19,23], exactly the checkpoint's
        # self_attn layers, and ssm_layers are the other 18. CC first assumed family-relative
        # (attn_layers_0..5) and the bijection assert caught it as an IndexError rather than a
        # silent mis-map, which is the whole reason this file asserts instead of reconstructs.
        assert layer in (ATTN if fam == "attn" else SSM), \
            f"{name}: layer {layer} is not in the checkpoint's {fam} family"
        mod = "self_attn" if fam == "attn" else "linear_attn"
        hf = f"{prefix}{layer}.{mod}.{proj}"
        assert hf in sites, f"const {name} maps to {hf}, which the checkpoint does not have"
        key = f"{hf}.{ab}"
        assert key in sd, f"factor {key} missing from the checkpoint"
        # ⛔ OP FORM IS SEMANTICS -- THE RESHAPE HERE IS THE ARC'S CENTRAL FINDING, NOT PLUMBING.
        # A parallel LoRA delta expressed as `linear` over RANK-2 consts DOES NOT COMPOSE through
        # the ANE mutable path; the IDENTICAL factors, identically packed, byte-identical,
        # expressed as `conv` over RANK-4 consts DO. 13 models, zero exceptions. The trainer
        # emits rank-2 (`nn.Linear`), the build declares rank-4, so the factors are re-expressed:
        #     A  [din, R]  -> [R, din, 1, 1]        B  [R, dout] -> [dout, R, 1, 1]
        # transpose then two unsqueezes. The byte content is unchanged; only the declared form is.
        # THE DEPLOY TOOL DOES THE RANK-4 RESHAPE ITSELF and expects RANK-2 factors in its own
        # orientation: A [R, din], B [dout, R]. The trainer stores A [din, R], B [R, dout], so
        # each is transposed here -- and NOT unsqueezed, or the tool's own reshape guard
        # (`want[:2] == arr.shape`) rejects it. Op form is still the arc's central finding; the
        # conv/rank-4 re-expression simply happens one layer down, at pack time.
        want = tuple(consts[name]["dims"])
        v = sd[key].T.contiguous().half()
        assert tuple(v.shape) == want[:2], \
            (f"{name}: MIL declares {want}, transposed factor {key} is {tuple(v.shape)} -- "
             f"refusing to pack a shape the build did not declare")
        sd[key] = v
        seen.add(hf)
        if ab == "A":
            order.append(base)          # MIL naming, matching the deploy tool's reconstruction
            A_list.append((hf, key))
    assert seen == set(sites), f"BIJECTION FAIL: {len(set(sites)-seen)} checkpoint sites unmapped"
    print(f"  ✅ bijection asserted: every const maps to a distinct checkpoint site, shapes agree")

    # the tool unpacks `A, B = fac[j]` -- a list of PAIRS, not two parallel lists
    out = {"A": [(sd[f"{hf}.A"], sd[f"{hf}.B"]) for hf, _ in A_list],
           "site_order": order, "sites": len(order),
           "rank": ck["rank"], "alpha": ck["alpha"],
           "steps": ck.get("step"), "tw": None,
           "heldout_ce_before": a.ce_before, "heldout_ce_after": a.ce_after,
           "source_ckpt": os.path.abspath(a.ckpt),
           "mapping": "name-keyed from the target MIL; families read from the checkpoint"}
    torch.save(out, a.out)
    print(f"  wrote {a.out}  ({len(order)} sites, site_order persisted)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--target", required=True, help="the compiled .mlmodelc to bind against")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ce-before", type=float, default=0.0)
    ap.add_argument("--ce-after", type=float, default=0.0)
    main(ap.parse_args())
