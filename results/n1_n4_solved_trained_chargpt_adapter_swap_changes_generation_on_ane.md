# N1→N4 SOLVED — a TRAINED char-GPT (ours), resident on the ANE, where swapping a mutable LoRA-style adapter changes the GENERATED TEXT (Shakespeare ↔ Alice), live on the Neural Engine. Retires the "untrained" caveat.

**Date:** 2026-07-28. CC own-raw, measured. Executes `BUILD_SCOPE_trained_ane_native_gpt_plus_lora_swap_2026-07-27.md` end to end. This is the "no untrained caveat" version of S4: the model is TRAINED, so the adapter swap changes real generation, not an arbitrary token id.

## The model (ours, trained)
Char-level GPT in our ANE-native form (conv2d-as-linear, `(B,C,1,S)`, attention-as-matmul, mean-of-squares LayerNorm, additive causal mask, fp16). 4 blocks, D=256, 4 heads, FF=1024, block=128, vocab=65 (chars). ~4.8M params. Trained on tinyshakespeare (5000 iters, MPS), val loss ~1.56. Token embedding + sampling host-side; the transformer body runs on the ANE (standard on-device serving split).

## The four gates (measured)
- **N1 — trained base.** Generates coherent Shakespeare: play format (name-colon-dialogue), verse, real history-play characters (WARWICK, QUEEN MARGARET, KING HENRY VI, KING RICHARD III), "thee"/"thou"/"'Tis". Loose word-to-word (char-level, small) but unmistakably Shakespeare.
- **N2 — trained CAUSAL model on the ANE.** Exported fp16; `MLComputePlan` = **ANE=214, CPU=0, GPU=0**. The causal mask + generation shape place fully on the ANE (the one real unknown of the build), zero CPU fallback — same as the non-causal `txf`.
- **N3 — base-frozen + adapter-mutable split.** Parallel head: `logits = base_head(h) + adapter(h)`; base_head + trunk frozen in `weight.bin`, adapter (D→V) split into `adapter.bin`, BFMI declares only the adapter mutable. Split places **ANE=216, CPU=0, GPU=0**.
- **N4 — two adapters, GENERATION CHANGES on the ANE.** Two head-adapters trained (base frozen): A on Shakespeare (loss 1.086), B on Alice in Wonderland filtered to the base vocab (loss 1.587). Swapped over the resident base ON THE ANE (units=3, injected via `e5rtMutableMILWeightURLs` at the AppleIntelligence trusted path), autoregressive generation (host embed + ANE body forward loop + sampling). **The only thing that changed between the two runs is the injected adapter blob; the base is byte-identical.**

## The result (on-ANE generation, the proof)
Adapter A (Shakespeare), generated on the ANE:
```
RICHSTER:
I go without to me the lie.
GREGORY:
But I can form Rome beat his charge
To the Duke of English be more than encounter.'
WARWICK:
Then queen is but and mine own good for his master,
With honour set the enter of my heart to hit,
```
Adapter B (Alice), same base, swapped adapter, on the ANE:
```
a yond fineing how them sight govern an all to the
cereated the babitterious repictured face. The ontaeu
aded to one, in letter your of his tongue ten side.
And man, the danger, and he was lawful much Laughes...
```
A = dramatis-personae verse ("mine own", "honour", character headers). B = lowercase running prose, no character names, Alice's register bleeding through the frozen Shakespeare base. Swap the adapter → the model writes differently, live on the Neural Engine.

## Significance
Retires the **"untrained"** caveat that rode the shipped post/repo/videos. No longer "arbitrary token 436→544 on an untrained toy": a TRAINED model that writes Shakespeare, where swapping the mutable adapter makes it write like Alice, on the ANE, through the exact sanctioned mutable-weight mechanism (S1→S4). Real capability behind the mechanism. Natural centerpiece for a stronger post / paper figure, and it stands up the working substrate the bridge→adapter distillation lead (`vault/ane-reverse/bridge_to_adapter_distillation_lead.md`) would run on.

## Honest scope (unchanged discipline)
- Char-level GPT (NanoGPT scale), NOT a frontier LLM. What HONESTLY remains is "small trained char-GPT" — exactly NanoGPT's scope.
- The adapter here is a **head-only** adapter (parallel LM head, D→V): the swap re-maps the output projection given the frozen base's hidden state (a visible style shift). A deeper multi-layer LoRA is a natural next step; the head adapter is the single-blob form matching the proven mechanism.
- SIP-off travels. Swap-at-bind (per-request), not mid-generation. Embedding + sampling host-side; transformer body on the ANE.

## Artifacts (scratchpad + ~/Desktop/ane-coreai-debug)
`train_chargpt.py` (base), `train_adapters.py` (A/B), `build_chargpt_ane.py` (N2 placement), `build_gen_assets.py` (split + adapter blobs + emb/vocab), `gen.mm` (on-ANE generation harness). Checkpoint `chargpt.pt`, adapters `adapters.pt`. Base+adapter split `cgpt.mlmodelc`; adapterA/B.bin at the trusted path.
