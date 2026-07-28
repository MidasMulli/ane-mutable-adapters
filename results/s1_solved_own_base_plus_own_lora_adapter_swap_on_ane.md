# S1 SOLVED — our own base (frozen) + our own LoRA-style adapter (mutable), swapped live, load-bearing, on the ANE, EFFICIENT (adapter-only cluster, base not re-patched). "Our model AND our adapter" is now real and verified.

**Date:** 2026-07-27. CC own-raw, measured. Supersedes the premature `S1_RESULT_..._2blob_injection_unsupported` doc (the `-14` "single-blob-only" verdict is RETRACTED — see reconciliation below).

## The result (measured, decisive)
Two model shapes, both our own, both proven:
- **Serial adapter (`brad`):** trunk (40 residual conv blocks) → `fin` output head. Trunk frozen in `weight.bin`; `fin` adapter split into `adapter.bin`, BFMI declares ONLY `adapter.bin` mutable.
- **Parallel LoRA (`brlad`):** `y = base_head(h) + adapter(h)` — the TRUE LoRA residual shape. `base_head` frozen in `weight.bin`; `adapter` split into `adapter.bin`, BFMI adapter-only.

For BOTH, injecting adapter A vs adapter B over the FROZEN base:

| | base `weight.bin` | output y[0..2] |
|---|---|---|
| brad — adapter A | md5 `d0b58fed…` | −1164, −1043, −915 |
| brad — adapter B | md5 `d0b58fed…` (identical) | +1029, +815, +782 |
| brlad — adapter A | md5 `575afc38…` | −1218, −1155, −1061 |
| brlad — adapter B | md5 `575afc38…` (identical) | +975, +702, +637 |

- **Load-bearing:** A vs B flip the output sign; both finite; **deterministic** (A/B/A/B repeats byte-for-byte).
- **Base frozen:** `weight.bin` md5 is CONSTANT across swaps — the base is never touched. Only `adapter.bin` changes (distinct md5s).
- **Injection is mandatory:** with no injection, predict refuses — *"BlobFileMutabilityInfo containing mutable weights, but they are not provided by SetMutableMILWeightPaths API."* The mutable adapter MUST come through the injection API (no silent file-at-path fallback).
- **On the ANE:** MLComputePlan `ANE=123/125, CPU=0`.
- **EFFICIENT split (the whole point):** kernel log shows the base as a single frozen kernel section (`initSplitKernelSections … section_size: 2961024`) and the adapter as a SEPARATE small mutable cluster (`mutableClusterIndex: 0`, `sectionSize: 0x4000` = 16 KB). Per-swap, ONLY the 16 KB adapter cluster is fresh-patched — measured **~37–41 µs** — while the ~3 MB base stays resident. This is base-resident + cheap-adapter-swap, NOT whole-blob re-patch.
- **Trust:** `aneVnodeTrustVerification: AppleIntelligence DataVault verification succeeded` — our own adapter bytes pass via the AppleIntelligence path-prefix.

## Reconciliation — the earlier `-14` verdict was WRONG
The prior S1_RESULT concluded "2-blob mutable injection is unsupported (single-blob assumption)." **Retracted.** The 2-blob base+adapter split injects fine when: (a) `adapter.bin` is a VALID single-tensor blob (built from a real adapter-only model, tensor at offset 64), and (b) you inject the MATCHING adapter path (not `weight.bin`/`kv=0`, not both blobs). The earlier `-14` came from injecting the wrong key / a hand-built adapter blob with bad metadata on tiny CPU-placed models — an artifact, not a mechanism limit. The parallel `add()` LoRA structure — the very shape first blamed — works perfectly (`brlad`).

## What this establishes
- The **efficient base-frozen + adapter-mutable split** the earlier analysis thought required §2a (host-toolchain) is actually reachable through the ordinary CoreML/aned BFMI path: declare ONLY the adapter blob mutable, keep the base as a distinct frozen blob, inject only the adapter. aned produces a per-adapter mutable cluster distinct from the base kernel section. **No §2a mint needed.**
- "Our model AND our adapter" is now literal: our own conv base (frozen) + our own LoRA-style adapter (mutable), swapped, different output, on the ANE.

## Honest scope (what this is NOT yet)
- The base is a **convolutional** net, not a transformer/LLM. The base+LoRA-adapter SERVING MECHANISM is proven on our own model; a real LLM base is the remaining scale step (S4).
- SIP-off travels (own adapter via the AppleIntelligence path).
- Swap = new program instance + adapter patch (~tens of µs for a 16 KB adapter) — per-request/session routing, not mid-generation mutation.

## Artifacts (scratchpad)
`brad.mlmodelc` (serial), `brl.mlmodelc`→`brlad.mlmodelc` (parallel LoRA), `adonly_A/B.mlmodelc` (adapter blobs), `build_basead_resid.py`, `build_brlora.py`, `build_adapter_only.py`, `halfa_ab.mm` (cancellation-proof metric harness).
