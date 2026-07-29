# Real pretrained LLM: swap trained LoRA adapters on Qwen3-0.6B on the ANE

The `chargpt/` demo trains a small character-level GPT of our own. This one takes a **real, released,
pretrained model**, Qwen3-0.6B, and does the same thing: keep the base frozen and resident on the Neural
Engine, and swap a trained LoRA adapter over it so the generated text changes register while the base
`weight.bin` stays byte-identical.

Same requirements banner as the top-level README: **research environment only** (SIP off +
`amfi_get_out_of_my_way=1`), and the swap harness must be entitled or the mutable plan build fails with -14.

## What it shows

- Qwen3-0.6B (GQA, RoPE, per-head q/k RMSNorm, SwiGLU, vocab 151936) ported to the ANE-friendly conv form.
  The port is faithful: logits cosine **1.000000** against Hugging Face on a reference prompt.
- The model places fully on the Neural Engine (CoreML compute plan: all compute ops on ANE, CPU 0).
- Two LoRA adapters trained on the frozen base (Shakespeare and Alice in Wonderland), deployed as a
  **two-conv low-rank parallel delta** on the q and v projections of all 28 layers
  (`proj = base(x) + B(A(x))`, base frozen in `weight.bin`, the 112 low-rank factors mutable in one
  **~4.6 MB** `adapter.bin`).
- Swapping only the mutable adapter over the resident base changes what the model writes:

```
prompt "Once upon a time" ->
  adapter A (Shakespeare): "...I have heard of the noblest of the world ... KING EDWARD ..."
  adapter B (Alice):       "...'I do not know what I did,' said Alice, ..."
```

The base `weight.bin` (one file, ~1.4 GB) is used for both adapters and is never written by the swap.

## How the deep adapter is deployed

A LoRA is trained as `delta = (alpha/r) * B @ A` (low rank). It is deployed by keeping the two factors as
two small parallel convs next to the frozen base projection: `proj = base(x) + B(A(x))`, with A `[r,D]` and
B `[out,r]` both declared mutable (CoreML `BlobFileMutabilityInfo`) and the base projection frozen. All 112
factors (A and B, q and v, across 28 layers) are packed into one `adapter.bin`, so a task is one file to
swap. Shipping the factors instead of the folded dense `[out,in]` delta makes the adapter **~38x smaller**
(~4.6 MB vs ~176 MB) and is token-identical on the ANE. (The `alpha/r` scale is folded into A at pack time
so the model graph carries no scalar; otherwise coremltools' `fuse_conv_scale` rewrites the B consts.)

## Run it

```bash
cd qwen
python3 train_lora.py                     # downloads corpora, trains adapter A + B into _out/ (MPS/CPU)
python3 gen_check.py "Once upon a time"   # optional: confirm the adapters shift generation, no ANE needed
python3 deploy_lora.py                    # builds the base+delta model + adapter_A.bin / adapter_B.bin
export ANE_TRUSTED_ADAPTER_DIR=<an AppleIntelligence AppModelAssets path-prefix, SIP-off writable>
bash run.sh                               # generates on the ANE under each adapter, swapping the one file
```

Python deps: `torch`, `transformers`, `coremltools`, `safetensors`, `huggingface_hub`. If your `python3`
does not have them, point the scripts at one that does with `export PYTHON=/path/to/python3` (used by
`run.sh`; run the `.py` steps with that interpreter directly).

Outputs go to `qwen/_out/` (override with `QWEN_OUT`); model weights and adapters are gitignored, corpora
download on first run. Knobs: `QWEN_LORA_RANK` (default 16), `QWEN_STEPS` (default 400), `QWEN_PROMPT`,
`QWEN_NGEN`.

## Scope and honesty

- Qwen3-0.6B, not a trained 8B. The training here is light (a few hundred steps per adapter): the register
  shift is clear but not sharp. More steps sharpen it.
- Swap-at-bind (per load), not editing weights mid-generation. Token embedding and sampling run on the host;
  the transformer body runs on the ANE.
- SIP off is required to write the trusted directory. Apple's mutable-weight mechanism is instrumental; the
  port, the training, and the adapters are ours.
