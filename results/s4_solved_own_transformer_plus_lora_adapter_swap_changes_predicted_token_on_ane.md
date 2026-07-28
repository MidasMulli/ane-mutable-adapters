# S4 SOLVED — our own TRANSFORMER base (frozen, resident on the ANE) + our own LoRA adapter (mutable). Swapping the adapter changes the PREDICTED NEXT TOKEN. The literal LLM version of the mutable-weight result.

**Date:** 2026-07-27. CC own-raw, measured. Completes the S1→S4 ladder: the base+adapter mutable-weight split now runs on a real transformer (attention + MLP + LayerNorm + LM head → vocab logits), not a conv net.

## The model (our own)
- **Transformer:** 4 blocks, D=256, 4 heads, FF=1024, seq=32, vocab=1000. Real multi-head attention (softmax over the sequence, einsum QK/AV), MLP (GELU), LayerNorm, residuals. ANE-friendly `(B,C,1,S)` layout with conv2d-as-linear (Apple's ANE-transformer principle).
- **LoRA-style head:** `logits = base_head(h) + adapter(h)`. `base_head` (frozen) stays in `weight.bin`; `adapter` (a `[1000,256,1,1]` parallel projection) split into `adapter.bin`, BFMI declares ONLY `adapter.bin` mutable.
- **Placement:** MLComputePlan `ANE=134, CPU=0, GPU=0`. The whole transformer — softmax, attention einsums, LayerNorm — places on the ANE with NO CPU fallback.

## The result (measured, decisive)
Inject adapter A vs adapter B over the FROZEN transformer base; read the predicted next-token (argmax over vocab at the last position):

| adapter (own) | predicted next-token | max logit | base `weight.bin` |
|---|---|---|---|
| A | **token 436** | 17.08 | md5 `b1cc1e35…` |
| B | **token 544** | 22.56 | md5 `b1cc1e35…` (identical) |

- **Load-bearing at the LLM level:** different adapter → **different predicted token** (436 vs 544). Not just different numbers — a different model *decision*.
- **Base frozen:** `weight.bin` md5 constant across swaps; the transformer base is never touched. Only `adapter.bin` changes.
- **Deterministic:** A/B/A/B repeats exactly (436, 544, 436, 544).
- **Efficient split (kernel log):** base = one frozen kernel section (`section_size: 6811776` ≈ 6.8 MB); adapter = a SEPARATE mutable cluster (`mutableClusterIndex: 0`, `sectionSize: 0x80000` = 512 KB), fresh-patched per swap while the base stays resident. For a real 8B the base is GBs and the LoRA is MBs — the ratio only gets more favorable.
- **Trust:** `aneVnodeTrustVerification: AppleIntelligence DataVault verification succeeded` — our own adapter bytes admitted via the AppleIntelligence path-prefix.
- **Injection mandatory:** no-injection load refuses at predict ("not provided by SetMutableMILWeightPaths").

## Honest scope (do not overclaim)
- **Architecture, not a trained/large model.** This is a genuine transformer LANGUAGE-MODEL architecture (attention + LM head + next-token logits), but SMALL (4 layers, D=256, V=1000) and UNTRAINED (random weights) — so tokens 436/544 carry no meaning; the *mechanism* (adapter swap → different token decision) is what is demonstrated. Scaling to a trained 8B is a separate size/quality step (CoreML-mlprogram size + ANE-placement limits to measure), NOT required to establish the mechanism.
- SIP-off travels (own adapter via the AppleIntelligence path).
- Swap = new program instance + adapter patch (512 KB cluster) — per-request/session adapter routing, not mid-generation mutation.

## What the whole arc now supports (honest headline)
Our own model — a transformer with an LM head — resident on the Apple Neural Engine, with a swappable LoRA adapter (our own), through Apple's sanctioned mutable-weight mechanism. Swapping the adapter changes the model's predicted token. Base frozen and resident; only the small adapter is re-patched. Verified on-ANE (MLComputePlan 134/134, CPU=0) and in the kernel log (separate base section + adapter mutable cluster + AppleIntelligence-path trust + fresh patch).

## Artifacts (scratchpad)
`txf.mlmodelc`→`txfad.mlmodelc` (transformer + split LoRA head), `adT_A/B.mlmodelc` (adapter blobs), `build_txf.py`, `build_adonly_txf.py`, `halfa_txf.mm` (argmax/predicted-token harness). Ladder: `brad`/`brlad` (conv base, S1), `txfad` (transformer base, S4).
