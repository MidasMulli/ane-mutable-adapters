# ane-mutable-adapters

Hot-swapping a LoRA adapter on the Apple Neural Engine, with the base model frozen and resident, through Apple's own on-device mutable-weight mechanism. Our own transformer, our own adapters, swapped live: the model predicts a different token, and the base is never touched.

> **This is a research characterization, not a stock-Mac feature.** It requires **SIP off** plus `amfi_get_out_of_my_way=1`, which is a fully unlocked research machine, not a normal user's Mac. The point is to characterize how the ANE's on-device adapter mechanism and its file-trust model actually work, on macOS 27, using our own model and our own adapters. Nothing here works on a stock, secured Mac.

## What it shows

The ANE serves on-device adapters by keeping one base model compiled and resident, and swapping a small adapter over it. This repo reproduces that shape end to end on a model that is entirely ours:

- A small transformer (multi-head attention, MLP, LayerNorm, an LM head to 1000-vocab logits) that places fully on the Neural Engine (CoreML compute plan: 134/134 ops on ANE, CPU 0).
- The LM head split into a **frozen base** part and a **mutable adapter** part (`y = base_head(h) + adapter(h)`), the base staying in `weight.bin`, the adapter in its own `adapter.bin` declared mutable.
- Two of our own adapters swapped over the resident base. Adapter A makes the model predict token 436, adapter B predicts 544, deterministically. The base `weight.bin` is byte-identical (md5 constant) across both swaps.

In the kernel log (`results/kernel_log_swap.txt`), the base is one frozen 6.8 MB kernel section and the adapter is a separate 512 KB mutable cluster; only the adapter is re-patched per swap. That is the efficient base-resident, cheap-adapter-swap shape, on a real transformer, driven with our own weights.

## How it works

1. **Author the model with an adapter path and declare it mutable.** The base must be compiled with the adapter present in the architecture and the adapter's weight blob declared mutable (CoreML's `BlobFileMutabilityInfo`). The adapter's shape is fixed at compile time; only its values swap later.
2. **Keep the base a distinct, frozen blob.** The base weights stay in `weight.bin` and are not declared mutable, so a swap never rewrites them.
3. **Inject only the adapter at load.** The mutable adapter is supplied through the sanctioned injection API (`e5rtMutableMILWeightURLs`). Loading with no adapter refuses at predict time; the model demands its mutable weight.
4. **The trust model.** The ANE trusts an adapter file by its **directory (a path-prefix)**, not by a content signature. Placing our own adapter under a trusted path-prefix (an AppleIntelligence AppModelAssets directory, writable only under SIP off) clears `aneVnodeTrustVerification`, so arbitrary own content is admitted. macOS 27 removed the public custom-adapter API, but this internal file path still accepts our weights.

Each swap is a **bind**: a fresh program instance over the resident base plus an adapter patch. It is per-request routing, not editing weights mid-generation.

## Reproduce

Research environment only. See the requirements banner above.

```bash
export ANE_TRUSTED_ADAPTER_DIR=<an AppleIntelligence AppModelAssets path-prefix, SIP-off writable>
bash run.sh
```

Files:
- `build/build_transformer.py`  the ANE-friendly transformer with a split LoRA head
- `build/build_adapter.py`      builds a single-tensor adapter blob (two distinct ones, A and B)
- `build/split_adapter.py`      repoints the adapter const to `adapter.bin` and declares it mutable
- `build/split_adapter_multi.py` the same for N consts at once, for a multi-layer adapter
- `harness/swap_multi.mm`       swap harness taking N key/URL pairs
- `harness/swap.mm`             loads the base, injects one adapter, reads the predicted token
- `harness/ane.entitlements`    the ANE mutable-weight entitlements (self-signed, honored only under SIP off)
- `harness/build_harness.sh`    builds + entitles both harnesses (`run.sh` calls it)
- `chargpt/`                    the **trained** follow-up: train a char-GPT, train two adapters, swap them on the ANE
- `results/`                    the S1, S4 and N1-N4 findings, plus a verbatim kernel-log excerpt
- `media/`                      a mechanism video, a terminal-replay video, an interactive explainer

> **The entitlements are mandatory.** The swap harness must be codesigned with `harness/ane.entitlements`
> (`build_harness.sh` does this). Without those four `com.apple.aned.private.*` + `iokit-user-access`
> entitlements, the e5rt mutable-weight plan build fails at load with error **-14**
> (`Failed to build the model execution plan using ... model.mil`): a tooling gate, not a model problem.
> The entitlements are grantable only under SIP off + `amfi_get_out_of_my_way=1`.

## The trained follow-up (`chargpt/`)

The demo above swaps an adapter on an **untrained** model and flips a token id. That proves the mechanism
but not that it means anything. `chargpt/` closes that gap: we train a small character-level GPT
(D=256, 6 layers, 4 heads, block 128) on tinyshakespeare, train **two LoRA adapters over the one frozen
base**, and swap them on the Neural Engine. The generated text changes register with the swap while the
base stays byte-identical.

CoreML's compute plan places every op of the transformer body on the ANE (216 of 216, CPU 0). Token
embedding and sampling run on the host; the body runs on the ANE.

```bash
cd chargpt
python3 train_chargpt.py          # downloads tinyshakespeare, trains the base
python3 train_adapters.py         # trains adapter A (Shakespeare) and B (a second register)
python3 build_chargpt_ane.py      # exports the ANE-native body
python3 build_gen_assets.py       # builds the generation assets
# then build + run gen.mm against a trusted adapter dir, as in run.sh
```

Outputs go to `chargpt/_out/` by default (override with `CHARGPT_OUT`). Model weights are gitignored;
the corpus is downloaded on first run. See
`results/n1_n4_solved_trained_chargpt_adapter_swap_changes_generation_on_ane.md` for the measured record.

## Scope and honesty

- The **S4 demo** (`build/`) is **small and untrained**. It is an architecture demonstration of the mechanism, not a trained model. The token ids (436, 544) carry no meaning; what is real is that swapping the adapter changes the model's decision while the base stays frozen.
- The **trained demo** (`chargpt/`) removes that caveat: a char-level GPT we trained, where swapping the adapter changes the text it actually writes. It is still small (NanoGPT scale, character-level), and it is not a trained 8B.
- **SIP off is required** to write the trusted directory. This is Apple's internal mechanism in a research setting.
- Apple's mutable-weight mechanism is **instrumental**. The transformer, the LM head, and the adapters are ours.

## Related work

cf. [@Mechramc](https://x.com/Mechramc)'s [Orion](https://arxiv.org/abs/2603.06728), which serves adapters a different way: the LoRA matrices are passed as IOSurface **inputs** to a fixed compiled graph (with a separate weight-file reload path used for training). This repo instead drives Apple's sanctioned mutable-weight **file** mechanism and characterizes the ANE's file-trust model that gates it.

## License

MIT. See `LICENSE`.
