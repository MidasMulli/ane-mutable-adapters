# Supplement: the codebook adapter, trained and bound on the Apple Neural Engine

This is the runnable build behind the paper. It trains a low-rank adapter that binds a *codebook*:
an arbitrary map from eight invented stroke-glyphs (present in no font and no corpus) to letters,
then binds that adapter onto a resident base model on the Neural Engine and decodes glyph strings the
base cannot read. It exists so the mechanism the paper characterises can be reproduced, not just read.

## ⛔ WHAT RUNS WHERE (read this first)

**The trainer runs on a stock Mac (GPU / PyTorch-MPS). Only the bind and the generation run on the
Neural Engine. On-ANE training is NOT part of this artifact.** The paper's separate, synthetic-topology
result that a whole training step can be *expressed* on the engine is a different object and is not
what this supplement builds or claims. What this supplement demonstrates is the deployment pipeline:
train (stock) → convert → bind (ANE) → generate (ANE).

## The two tiers

- **Stock Mac, no privileges:** corpus generation and training (`make_data_codebook.py`,
  `adapter_trainer.py`). This half runs anywhere.
- **Requires SIP off, AMFI off, and bring-up:** the bind and on-ANE generation. AMFI off is needed to
  adhoc-sign the four `com.apple.aned.private.*` + `com.apple.private.iokit-user-access` entitlements
  the bind harness carries; SIP off is needed to stage the adapter under the AppleIntelligence trusted
  path (`/private/var/db/AppleIntelligencePlatform/AppModelAssets/`), which the runtime trusts by
  container prefix. Neither is a way to defeat the weight seal; both are the access cost of this
  private research path, which is not an Apple-supported public adapter API.

## ⛔ WHAT THIS PACKAGE DOES NOT SHIP, AND WHY

Stated plainly so nobody discovers it three commands in. **The bind/generate half needs artifacts that
are ~1.2 GB of compiled output and one entitlement-signed binary**, which are build products, not
sources:

| Needed for bind/generate | What it is | How you get it |
|---|---|---|
| `_mlc_shadow/gdnmut_gc0.mlmodelc` | the 0.8B GDN compiled with mutable sections | build from the HF weights (see Bring-up) |
| `_demo/embed.mlmodelc`, `_demo/lmhead.mlmodelc` | ~485 MB each | same |
| `_demo/q35_swap_img_nb` | the signed bind harness | `vendor/harness/build_gdn_harness.py` from `vendor/harness/gdn_kvswap.mm` |
| Qwen3.5-0.8B weights | the base model | `huggingface-cli download Qwen/Qwen3.5-0.8B` |

Everything **else** — the trainer, the converters, the corpus generator, the decode module, the demo
server and the acceptance check — ships here, with the corpus and the paper's checkpoint.

`cleanroom_gate.sh` is the acceptance criterion and it is split along exactly this line:
**Tier A (portability)** runs anywhere and is what a reader can verify immediately; **Tier B
(privileged)** is the bind and needs the table above. A machine without bring-up cannot pass Tier B,
and the gate says so rather than reporting a green it did not earn.

## Recorded deployed result (corrected 2026-11-16)

The deployed adapter produced 213/240 correct character positions and 41/60 exact strings on 60 images of 30 combination-held-out targets. The command used “Decode the symbols.” with EVAL_HINT empty. Consequently the logged PROMPT(off+hint) condition repeated BASE(off); it is not a separate prompting method. The 30 targets are absent from both fit_train and fit_heldout. The four illustrated successful renders cover two combinations.

The verified split is fit_train / fit_heldout (render holdout for checkpoint selection) / compose (combination holdout); the evaluation takes the first 60 compose rows.

## Export helper scope

deploy_codebook.sh packs and stages an adapter for an existing convolution-form compiled graph. It hard-codes FORM=conv. The FORM option in vendor/deploy_q35_adapter.py selects matching constant names/layout in an existing graph; it is not a graph converter. A second export form requires its own correctly compiled artifact.

## Pipeline (train → convert → bind → generate)

```
1. python3 make_data_codebook.py
     # invented-glyph corpus + arbitrary map; emits fit / fit_train / fit_heldout / compose / tier3
     # fit_train|fit_heldout are a RENDER hold-out (last repetition of each combination);
     # `compose` is the separate COMBINATION hold-out the paper reports on.

2. python3 adapter_trainer.py \
     --model <path-to-Qwen3.5-0.8B-snapshot> \
     --train   data_codebook/fit_train.jsonl \
     --heldout data_codebook/fit_heldout.jsonl \
     --out     run_v1 \
     --image-field image --image-prompt "Decode the symbols." \
     --deploy-target _mlc_shadow/gdnmut_gc0.mlmodelc \
     --epochs 6 --eval-every 50 --seed 0
     # --deploy-target asserts all 48 sites are bindable BEFORE training, and derives
     # --targets and --rank from the compiled article rather than requiring you to know them.

3. sudo -v && ./deploy_codebook.sh          # ckpt -> conv mutable blob -> stage to the trusted path
     # packs/stages for an existing convolution-form compiled graph (FORM=conv hard-coded; see below)
     # CK= overrides the checkpoint; its md5 is asserted (see 'The checkpoint' below)

4. python3 codebook_web.py                  # serve the real user path on :8711

5. python3 demo_acceptance.py               # 6 gates on the served path (G1..G6), exit 0 = all pass
```

Step 2 reproduces the paper's run exactly: that invocation is recorded verbatim in
`run_v1/training_report.json` under `argv`.

## Reproducing without retraining

The corpus is **deterministic**. `make_data_codebook.py` at its default settings regenerates, byte for
byte, the corpus `run_v1` was trained on (verified: 1400/700 rows identical, 2100/2100 PNGs md5-identical).
Every corpus directory records the argv and seed that made it in its own `manifest.json`, so any
corpus here can be rebuilt from the file it sits next to.

`data_cbmix_L1/` (360 KB, shipped) is the labelled single-glyph scoring set G6 uses. It was made with:

```
python3 make_data_codebook.py --out data_cbmix_L1 --n-sym 8 --length 1 \
  --train-combos 0 --held-combos 8 --reps 3 --held-reps 3 --seed 0
```

## The checkpoint

The paper's figure is **`run_v1/checkpoints/ckpt_step0001750.pt`**, md5 `972e196c88946a5b26ed3d9172d6ea43`
— the held-out-CE-minimum checkpoint (0.299 at step 1750 of 2100; the trainer's overtraining flag fired
and the endpoint was declined).

⚠️ **Three different files have carried a `ckpt_step*.pt` basename on this arc** — including one
from an earlier, far worse run (best held-out CE 1.714) that previously shipped under the *same
basename* as the deploy script's default. Pointing the script at "the file we ship with that name"
therefore bound a ~6x worse adapter while every structural gate still passed.

Two things now prevent that. Only **one checkpoint ships** (this one), and `deploy_codebook.sh`
**asserts its md5 before it binds anything**. Deploying a different checkpoint is fine, but you have
to say so out loud: `CK=<path> CK_MD5_EXPECT=<md5>`.

## Verification gates

`demo_acceptance.py` enforces these **in the code path**, and each is named with the predicate that
implements it:

| Gate | Claim | Predicate |
|---|---|---|
| **G1** capability gap | base does not decode the glyph; the adapter does | `demo_acceptance.py` `g["G1_capability_gap"]` |
| **G2** bind-once | a base↔adapter change costs a real bind (>10 ms); a same-arm repeat costs 0.0 ms | `g["G2_bind_once"]` |
| **G3** byte-identity | that zero-bind repeat emits *exactly* the tokens the fresh bind emitted — this is what separates genuine reuse from a silent fallback to base or stale weights | `g["G3_byte_identity"]` |
| **G4** base untouched | base `weight.bin` md5 constant across every swap | `g["G4_base_untouched"]` |
| **G5** blobs distinct | the on and off adapter blobs differ | `g["G5_blobs_distinct"]` |
| **G6** accuracy floor | first-char accuracy **rate** over `DEMO_ACC_N` labelled glyphs clears `DEMO_ACC_FLOOR` | `g["G6_accuracy_rate"]` |

**G6's numbers, stated exactly:** the floor is **`DEMO_ACC_FLOOR`, default 0.375** at
**`DEMO_ACC_N`, default 8**. Chance is 0.125 — that is the *baseline the floor sits above* (3×), not the
floor itself; it is reported as its own field. It is deliberately a **rate**, never a single-glyph
match: the deployed single-glyph rate is 0.625, so any one glyph is wrong ~37% of the time and a
single-glyph gate would be flaky by construction.

⛔ **G6 fails closed.** If its corpus is absent, or loads but matches none of the served glyphs, the run
**exits non-zero**. It previously reported `SKIP` for G6 and still printed `VERDICT: PASS` with exit 0 —
a five-gate run that looked identical to a six-gate one from the outside. A gate that reports success on
its own absence is worse than no gate.

### The bind-took control, cited where it actually lives

Binding a path to a missing adapter file must be **refused** (`-14`), which proves the runtime reads the
declared path rather than silently reusing what is resident. This check is real and it fires — but it is
in **`eval_syriac_ane.py`**, invoked by `deploy_codebook.sh` at the deploy stage, **not** in
`demo_acceptance.py`. Earlier revisions of this README listed it under `demo_acceptance.py`, where
`grep bogus` returns nothing. (`key=none` is not used here: it is a qwen-class check and this article is
the 0.8B GDN.)

### fvmlib overflow

`fvmlib_check.txt` is a **static 91-byte record of one observed run**, not a live check. Nothing greps a
bind log at acceptance time. The `>255` fvmlib overflow is a real failure mode — it passes every
downstream gate while the swap is silently a no-op — but **this package does not test for it**, and the
file should be read as a note, not a gate.

## Negative controls

`negative_controls.sh` feeds each gate an input it must **reject**, and passes only when the gate fails.
It exists because this arc shipped six pieces of prose asserting enforcement the predicate did not
perform. A gate never observed to fail is indistinguishable from a comment.

It covers G6-absent, G6-present-but-unmatched, a *positive* control that the shipped corpus is accepted
(without which the two refusals would be indiscriminate), the corpus generator's composition gate in
both directions, and the checkpoint-identity assert.

## Bring-up (Tier B only)

1. `huggingface-cli download Qwen/Qwen3.5-0.8B`
2. Build and sign the bind harness: `vendor/harness/build_gdn_harness.py --variant 08b --ctx 512`
   (source: `vendor/harness/gdn_kvswap.mm`). Requires AMFI off to adhoc-sign the entitlements.
3. Compile the base to `_mlc_shadow/gdnmut_gc0.mlmodelc` with mutable sections, and `embed`/`lmhead`
   into `_demo/`. Point `SHADOW=` and `Q35_DEMO_DIR=` at them if you put them elsewhere.
4. `sudo -v`, then run the pipeline from step 3 above.

## Layout

```
make_data_codebook.py      corpus generator (deterministic; records its argv+seed)
adapter_trainer.py         the trainer (verbatim copy of toolkit/train/adapter_trainer.py)
deploy_codebook.sh         convert -> blob -> stage -> on-ANE verify
codebook_web.py            the demo server (the real user path, :8711)
demo_acceptance.py         G1..G6 on the served path
eval_syriac_ane.py         the on-ANE three-arm evaluator (carries the bogus-path took-check)
cleanroom_gate.sh          ⭐ the acceptance criterion: Tier A portability, Tier B privileged
hermetic.sb                sandbox profile that makes Tier A a real clean room
negative_controls.sh       proof the gates can fail
vendor/                    to_deploy_format.py · deploy_q35_adapter.py · gdn_4b_model.py
vendor/tower/ane_vlm.py    the on-ANE decode module
vendor/harness/            gdn_kvswap.mm + build_gdn_harness.py (build the signed harness)
data_codebook/             generated by step 1 (not shipped; deterministic)
data_cbmix_L1/             shipped, 360 KB — G6's labelled scoring set
run_v1/                    the paper's run: training_report.json + the ONE cited checkpoint
run/                       an earlier run: training_report.json only (no checkpoints ship)
```
