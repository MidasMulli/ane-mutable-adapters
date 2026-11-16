# Paper 4 provenance

Every figure and every quoted measurement in the paper is keyed to a row here.
Each row records the instrument, the article, the value, and the raw or tool that
produced it. Rows whose full raw-file record is enumerated separately are listed in
`goldensource/`, which travels with this file.

76 keys.

## Nstar.crossover
- value: ORDER, NOT A FIGURE; the characterization has since LANDED (`stage1.fedchar.fullarticle`) and the RANKING IS WITHHELD BY RULING (2026-08-24: entitlement asymmetry), not deferred pending data. `84.4 / 0.300 ≈ 281` splices THREE articles , `T_bind` GDN 4-chunk / `ΔT_forward` ANEForge synthetic 8 MiB / bound-per-forward qwen , never shape-matched. The `0.300` denominator is substantially an `astype` copy artifact , RUNTIME-PINNED 2026-08-24: zero-copy fed is ~0.065 ms/5.24 MB at the wall ON THE E5RT ROUTE ONLY (substrate stream ~80 GB/s steady); on the CoreML runtime fed is NOT ~0 under any staging (+1.55/+7.26 ms/forward, `stage1.fedchar` row). Do NOT quote 281 as a crossover. Superseded on landing by the single-article characterization
- tier: cross-article ORDER
- n: ,
- tool: `cc_to_ca/2026-08-09T08-00-00`, `d_zerocopy_*`, `PREREG_2026-08-23_fedform_characterization`
- source: prose §4.4 only

## bindfit.synthetic_conv
- value: `T_bind = 5.257 ms + 67.1 µs/section` , SHAPE fit on a synthetic conv article (flat in bytes, linear in sections); intercept does not transfer across articles; companion `zzero.per_forward_bound`: bound-vs-baked per-forward delta under the replicate floor at every rung of a 112-section/528 MB ladder ⇒ < 1.65 µs/section, a resolution floor not a measurement; aggregate at deployed 96–112 sections = 0.16–0.19 ms upper bound
- tier: measured (fit) + BOUND
- n: ladder to 112 sections
- tool: z_zero rig
- source: cited §4.2; raws in `_z_zero/`

## cb.base.instruction
- value: 0.0150 pp / 0.0000 exact (key-2 corpus; key-1 corpus also 0.0150)
- tier: PyTorch
- n: 150

## cb.key1.compose.ce
- value: 0.8083 pp / 0.6067 exact (held-out-CE-selected, step 850 of 900)
- tier: PyTorch
- n: 150
- tool: `eval_codebook.py`
- source: `RESULT_codebook_v1_step850_data_codebook.json`

## cb.key1.compose.gate
- value: 0.8417 pp / 0.6267 exact (gate-selected, earliest max step 800; circularity pin)
- tier: PyTorch
- n: 150
- source: `RESULT_codebook_v1_step800...`

## cb.key2.compose.ce
- value: 0.9083 pp / 0.7667 exact (held-out-CE-selected step 800)
- tier: PyTorch
- n: 150
- source: `RESULT_codebook_key2_cold_step800...`

## cb.key2.compose.gate
- value: 0.9250 pp / 0.8133 exact (gate-selected step 700 of 850; selected on the reported split , optimistically biased by an unmeasured amount)
- tier: PyTorch
- n: 150
- tool: `eval_codebook_key2.py`
- source: `RESULT_codebook_key2_cold_step700...`

## cb.key2.compose.splitB
- value: 0.9633 pp / 0.8800 exact , NON-CIRCULAR by construction (select on half A = the 75 historically-evaluated combos, argmax pp earliest-on-tie → step 700; report ONCE on half B = 75 virgin combos). BASE on B: 0.0067 / 0.0000. Distinct emissions 91 (top appears 3×), best-constant 0.2000. the reweight gloss ("0.9600 ⇒ composition-neutral") is STRUCK 2026-08-24 -- 0.9600 is a verified PHANTOM (no reweight field, no 0.9600 in the named raw). The 0.9633 non-circular result STANDS ON ITS CONSTRUCTION; only the composition-neutral gloss rested on the phantom
- tier: PyTorch
- n: 150 rows / 75 combos
- tool: `eval_codebook_ab.py` (selection rule in code path before any B read)
- source: `RESULT_codebook_key2_AB_step700_splitB.json` + `data_codebook_key2/ab_split.json`

## cb.key2.fit.gate
- value: 0.9567 pp / 0.8733 exact @700 · CE-selected @800: 0.9650 / 0.9000
- tier: PyTorch
- n: 150

## cb.n1.deployed_single_glyph
- value: deployed single-glyph 0.625 first-char / 0.214 wrong-present (BASE 0.000 / 0.343); 5.0× chance; 9.6 se null-convention / 6.5 se Wald -- BOTH conventions named wherever quoted
- tier: deployed (ANE)
- n: 40
- tool: demo path
- source: `RESULT_n1_single_glyph.json`

## cb.supp.deployed_compose
- instrument: DEPLOYED (ANE, mutable bind, e5rt/CoreML demo harness)
- article: supplement-v1 adapter, ON blob d3e8ed5795caedee, on q35_08b_conv/gdnmut_gc0
- corpus/recipe: data_codebook fit_train, 6 epochs, seed 0 (the v1 config)
- selector: held-out-CE minimum, step 1750/2100 · OVERTRAINING_SUSPECTED fired ·
- split: compose (untrained combinations), REPORT HALF B, n=60 (selection used half A)
- values: per-char 0.887 · exact 0.683 · base 0.000/0.000 · deployed prompt arm 0.000/0.000 ·
- gates: bogus-path took-check PASS (refused, "ARM DID NOT TAKE") · base md5 constant ·
- source: deploy_v1.log + ane-mutable-adapters/vision/RESULT_syriac_ane_raw_all.json
- correction (2026-11-16): direct raw counts are 213/240 characters and 41/60 exact strings over 60 images of 30 compose combinations. The deploy command used EVAL_HINT empty, so the logged 'deployed prompt arm' repeated BASE(off) and is not an independent prompting control. The verified split is fit_train / fit_heldout / compose; the generic 'REPORT HALF B' print is not a separately verified subdivision.
- cited_in: §2.3 · F6 caption · §1.5 (pointer) · §1.3 carries the availability note number-free

## cb.visualkey.arms
- value: BASE and VISUALKEY: UNDETERMINED (well-formed 0/24, prose) · FEWSHOT4 0.1667 pp, ONE constant string · VISUALKEY+FMT4 0.1667 pp, the SAME constant (`decc`, 24/24 byte-identical to no-key)
- tier: PyTorch
- n: 24
- tool: `_cb_visual_key.py`
- source: `RESULT_cb_visual_key.json`

## cb.visualkey.baselines
- value: best-constant 0.3333 (observed constant scores BELOW it) · combination unit: prompt +0.62 se above the 0.125 floor · the paired "+10.1 se" adapter figure is STRUCK 2026-08-24 (computed at n=24 for an n=75 arm; SPEC §1 names it; §1.5 already removed it -- different n, no shared se)
- tier: derived
- n: 24
- tool: packet `20-30-00`

## census.null_dynamic_range_linear08b
- instrument: deployed harness, same metric as the conv nulls
- article: 0.8B linear-form (broken op form) build
- value: +10.6659 (null-arm dynamic range; the third sink, disambiguated 2026-08-24)
- window: rows predate the conv nulls
- source: TOOLS.md gather/repack rows, 2026-08-13 era
- cited_in: §2.9

## coldwarm.order_pin
- value: arms run arm-by-arm, contiguous , arm confounded with time-of-run; spreads NOT exchangeable across arms. No bind-time ranking is licensed (pal8 vs fp16 ranges do not overlap and are STILL not read)
- tier: design
- n: ,
- tool: ,
- source: ,

## coldwarm.spread
- value: warm med [min, max] per arm: fp16 71.7 [69.9, 79.4] · int8-pc 77.1 [76.0, 79.2] · pal8-pt 89.6 [84.0, 91.7] · pal4-pgc 75.3 [75.0, 79.3] · pal4-pt 79.8 [78.9, 84.5] · int4-pc 23,758.8 [23,382.6, 24,157.5]
- tier: deployed, 0.6B
- n: 4 warm/arm (+4 cold, = 48 runs)
- tool: `coldwarm.sh`
- source: `tools/quant_mutable_probe/coldwarm_raw/table.json`

## coldwarm.tok_s
- value: fp16 39.45 · int8-pc 44.2 · pal8-pt 42.6 · pal4-pgc 44.25 · pal4-pt 43.5 · int4-pc 9.05 (medians; §3.3 rounds 39.45→39.5)
- tier: deployed, 0.6B
- n: 4

## contention.coefficient
- value: ~0.24 pp of reasoner throughput per GB/s moved by a co-resident peripheral , ANE `+0.279 + 0.2407·GB/s`, GPU `+0.235 + 0.2409·GB/s`; slopes agree to 0.08%. INTERCEPT NOT ESTABLISHED (n=3, k=2 ⇒ df=1; F=130.6 vs F_crit(1,1)=161.4). Report the SUBSTRATE-INDEPENDENCE, never a magnitude. Fitted 1.26–20.1 GB/s; do not extrapolate past 20 GB/s (the 20→40 pair gives a steeper 0.2928). Paired design, each arm against its own adjacent baseline.
- tier: deployed, wall-clock
- n: 3 rates × 2 substrates, 2 reps × 3 prompts per arm
- tool: `necessity/run_necessity_v7.sh`, raw `necessity/v7_pairs.tsv`
- source: `18-25-00`

## corner.ce
- value: guaranty 2.8734 · S-1 2.9912
- tier: deployed
- tool: `qce_deployed.py`
- source: `22-00-00` §4

## corner.flip_warm
- value: ~88 ms (87–89)
- tier: deployed
- source: `22-00-00`

## corner.gates
- value: 5/5 , none-throws · bench-steady · A≠B · md5 constant · blobs differ
- tier: deployed
- tool: ,
- source: `22-00-00`

## corner.model
- value: Qwen3-1.7B, pal8-per-tensor, mutable-wired, 1.73 GB
- tier: deployed
- n: `_pal8wire17/`
- tool: `build_quant.py QMODE=pal8`
- source: `22-00-00`

## corner.placement
- value: 2731 ANE / 1 CPU
- tier: deployed
- source: `22-00-00` §4

## corner.tok_s
- value: 31.4 (30.8–32.0 across arms)
- tier: deployed
- tool: `gen_qwen`
- source: `22-00-00`

## corner.vs_06b
- value: −0.44 / −0.36 nats (the size step)
- tier: deployed
- n: vs 0.6B 3.3134/3.3506
- source: `22-00-00` §4

## corner.vs_fp16
- value: +0.0081 / +0.0047 (pal8 ~free)
- tier: cross-tier
- source: `22-00-00` §4

## dep.guaranty.base
- value: 3.3137
- tier: deployed ANE
- n: Qwen3-0.6B fp16 mutable
- tool: 50 / stride 110
- source: ,

## dep.guaranty.finance
- value: 3.3080 (−0.0057, flat)
- tier: deployed ANE

## dep.guaranty.legal
- value: 3.0005 (−0.3132)
- tier: deployed ANE
- source: 11.12% own / 1.16% other

## dep.noise_floor
- value: 0.0004
- tier: deployed ANE
- n: two independent builds, same graph
- tool: ,
- source: ,

## dep.pal8.maxmove
- value: ≤0.0016 vs fp16
- tier: deployed ANE
- n: pal8-per-tensor
- source: ,

## dep.s1.base
- value: 3.3509
- tier: deployed ANE
- tool: 50 / stride 231
- source: ,

## dep.s1.finance
- value: 3.1763 (−0.1746)
- tier: deployed ANE
- source: 2.61% own / 0.84% other

## dep.s1.legal
- value: 3.5298 (+0.1789)
- tier: deployed ANE

## dep.vs.pytorch.maxdelta
- value: ≤0.0017 on every cell
- tier: cross-tier
- n: ,
- tool: matched window
- source: ,

## dtforward.substrate_stream
- value: CANDIDATE , E5RT ROUTE ONLY (runtime-pinned 2026-08-24; the CoreML runtime measures 2 orders slower, `stage1.fedchar`). fed dispatch ANE busy, steady state ~65 µs @ 5.24 MB ⇒ ~81 GB/s (thirds-median, xctrace slope rig; a CONSISTENCY CHECK against the arc's ~80 GB/s AMCC-RD figure , different workload, NOT a confirmation) · transient first ~150 dispatches ~2× faster (obs ≤142 GB/s); MECHANISM UNDETERMINED between cache-residency and power-state ramp , the BAKED arm ramps the same way (1.1→9.6 µs) with NOTHING to cache · N=100 medians are PRE-STEADY-STATE, do not use
- tier: xctrace NE instrument
- n: 502 disp
- tool: `d_zerocopy_trace.py` + thirds re-parse
- source: `_d_zerocopy/trace.json` + XMLs

## dtforward.zerocopy_identity
- value: the fed zero-copy arm COMPUTES (not a stale-weights fast path): mode-invariance byte-identical, weight A→B→A through the input_view changes+restores, fed==baked max\
- tier: diff\
- n: 0.000000
- tool: ID-COMPUTES 4/4
- source: ,

## equiv.margins
- value: min 0.0000 (exact fp16 ties, SAME positions both builds , moot at bit tier); p5 0.391, median 10.83, 274 decisions
- tier: deployed
- n: 6 prompts
- tool: `bo_margin_typo.py`
- source: `13-30-00`

## equiv.pure_vs_mutOFF.logits
- value: bit-identical (max abs diff 0.000000, 9,723,904 fp16/prompt)
- tier: deployed, LOGIT-BIT
- n: 6 prompts
- tool: `logit_dump.mm`
- source: `13-30-00`

## equiv.tokens
- value: 6/6 token-identical, 137 tok (subsumed by logit tier)
- tier: deployed
- n: 6 prompts
- tool: `ladder.py`
- source: `12-55-00`

## fedswitch.restage_8mb
- value: fed-arm "switch" = restage of an ~8 MB adapter tensor ~0.60 ms (copy-in regime; the zero-copy restage measures 0.06-0.20 ms at 0.79-4.19 MB warm, `_stage1_fedchar` 2026-08-24) -- cited §4.3/§4.5 as the fed switch cost
- tier: measured
- n: --
- tool: d_zerocopy rig + stage1_fedchar
- source: `_d_zerocopy/` + `_stage1_fedchar/stage1_results.json`

## frame.degeneracy
- value: worst 4-gram-repeat 0.96 raw → 0.04 chat-framed, all rungs × queries
- tier: deployed
- n: 24 rows
- source: `12-55-00`

## frame.typo_exhibit
- value: chat-framed, 3 arms: BASE→"Infectious Diseases Society of America" (defensibly CORRECT literal reading) · FINANCE→"International Data Security Association" (a fabrication , the MISMATCHED arm producing a plausible invention; name it, don't let a reader find it, BP7) · LEGAL only →ISDA
- tier: deployed, observational
- n: 2 queries (1 discriminates)
- tool: `bo_margin_typo.py`
- source: `13-30-00`

## geni.hidden_restore_bind
- value: the vision demo's per-call cost was TWO `reloadChunks` (~50 ms each), ONE reported , GENI bound `startOn` at entry AND silently restored `curOn` after GENIDONE; every prior "~50 ms per decode" reading of that demo understated the actual per-call bind cost ~2×
- tier: deployed (source-read + gate-timed)
- n: ,
- tool: `gdn_kvswap.mm` pre-edit; `geni_rebind_gate.py`
- source: `_rebind_gate_2026-08-23.json`

## geni.norebind_same_arm
- value: 0.0 ms bind on a same-arm repeat decode; ~52 ms only on a base↔adapter change (nb pattern [0.0, 52.8, 0.0, 53.1] vs shipped [47.2, 51.2, 51.1, 47.5]); tokens BYTE-IDENTICAL to bind-every-time across 3 binaries × 4 arms; Outcome C excluded by measurement
- tier: deployed
- n: 12 calls
- tool: `geni_rebind_gate.py` (6/6 gates)
- source: `_rebind_gate_2026-08-23.json`

## inst.keynone.two_sided
- value: pure (no BFMI) runs · mutable throws E5RT BFMI , same binary/day/path
- tier: deployed
- n: 1+1
- tool: `ladder.py`
- source: `12-55-00`

## mutable.ane_coupled
- value: `MLComputeUnits` 0 and 1 both −14; only 3 and 2 build
- tier: deployed
- n: 0.6B mutable
- tool: ,
- source: `19-00-00`

## quant.break
- value: linear-int4 (any granularity) · int8-per-block(32)
- tier: deployed
- source: `16-00-00`

## quant.ce.int8
- value: +0.0287 / +0.0128
- tier: deployed
- n: 0.6B
- source: `19-00-00`

## quant.ce.pal4_pgc
- value: +0.844 / +0.626 (not a candidate)
- tier: deployed
- n: 0.6B
- source: `19-00-00`

## quant.ce.pal8
- value: +0.0004 / +0.0032
- tier: deployed
- n: 0.6B
- tool: `qce_deployed.py`
- source: `19-00-00`

## quant.int8.cliff
- value: collapses between activation peaks 2,000 (clean) and 14,358 (cosine 0.000)
- tier: deployed, isolated one-layer
- n: real block-2 `down_proj`, 1.7B
- tool: `int8_threshold_sweep.py`
- source: `20-00-00`

## quant.mechanism
- value: `Too many fvmlibs (more than 255)` → mutable compile fails → espresso silently rebuilds NON-mutable → adapter read statically
- tier: deployed, CC-reproduced
- tool: `mutable_bind_logcapture.sh`
- source: `16-00-00`

## quant.preserve
- value: int8-linear (per-ch/per-tensor) · pal4 · pal8 preserve the swap
- tier: deployed
- n: Qwen3-0.6B
- tool: `qbuild.py`, `runN.sh`
- source: `16-00-00`

## rebind.anchor_4b
- value: WITHDRAWN 2026-08-24 (operator ruling + no locatable raw). DO NOT CITE 84.4. On disk 84.4/n=186/[75.8,98.6] survives ONLY as an inherited header field inside two n=40 runs that measured 77.255 and 79.8. ⇒ SUPERSEDED BY `switch.gdn_4chunk_n40` (77.3, n=40, sha-verified) for this article, and by `switch.demo_change` (~52 ms, the bedrock demo) for mechanism/switch-cost claims. Golden source: `goldensource/switch.gdn_4chunk.json` (status SUPERSEDED, forwarding address recorded)
- tier: withdrawn
- n: ,
- tool: ,
- source: `goldensource/`

## rebind.coldfile
- value: a swap to a never-read adapter file costs ~22 ms (mincore-verified 0.0% resident before bind); the cold-file READ itself is 0.9–2.6 ms on a 1,344,320 B blob
- tier: deployed, 4B GDN
- n: ,
- tool: `bind_seq_timed.mm`
- source: packet `2026-08-10T09-00-00`

## stage1.fedchar
- value: POST-HOC (discovered, not predicted , the prereg's SHAPE-3 did not occur; the un-anticipated result is that the regime splits by RUNTIME, not staging). Three-arm per-forward walls, demo SHAPES and op form, NOT the demo article (16-section/2.10 MB fp16 stack vs the article's 96/7.18): baked 2.6584 · bound 2.6630 (Δ+0.0046, BELOW the 0.0242 floor , 16 sections only; the 96–112-section deployed aggregate 0.16–0.19 ms REMAINS A BOUND, not measured) · fed 9.9182 (Δ+7.26, batch-flat, 52% ANE busy by replicated trace pairs). Fed walls: +1.55 ms (0.39 MB fp16, s1024_512) and +7.26 ms (2.10 MB fp16, s2048_6144), batch-flat; INVARIANT to payload precision (fp32-mutated 4.19 MB and fp16 2.10 MB arms identical) ⇒ per-dispatch input handling, not bytes. Instrument: coremltools `predict()` MUTATES the caller's dict fp16→fp32 on first call , byte accounting after a predict is 2× wrong. Bind costs, 2.10 MB blob: first-in-process endpoints 53.6 and 360.8 ms, n=4 (53.6/53.9/324.5/360.8) , UNEXPLAINED VARIATION, endpoints not a range, per the §3.5 discipline; different-file 8.7–9.3; warm rebind 6.6–6.9. Took-check sumabs 1382.17/2048 finite nonzero; bound cell at BOTH unit settings identical
- tier: measured (5×300, arm-order reversed, floor pinned first)
- n: 6 cells + bound
- tool: `stage1_fedchar.py` + `s1_zc_driver.mm` + `s1_bound_driver.mm`
- source: `_stage1_fedchar/{stage1_results.json, zc_rows.jsonl, trace_split_v2.json}`

## stage1.fedchar.fullarticle
- value: THE THREE-ARM ROW ON THE DEMO ARTICLE ITSELF (0.8B GDN decode step, 96 mutable sections, REAL codebook adapters; supersedes the shapes-stack row above for article-level claims). Walls (5×300, floor pinned 0.0312 first): baked 10.0325 · bound 10.0525 (Δ+0.0200 BELOW floor; z_zero 96-section aggregate ≤0.158 ms , consistent, still a bound; bound-vs-baked arms differ in TWO axes , BFMI declaration AND build , a numerical bit-identity is banked for pure-vs-mutOFF but not a TIMING identity, so the +0.02 could be adapter resolution, BFMI overhead, or build variance; below-floor either way) · fed 38.3809 (Δ+28.35, 908× floor). Fed = 96 factor inputs / 7.18 MB fp16. NO RATE IS QUOTED, BY FINDING: the cost is BYTE-INVARIANT (fp16 2.10 MB and fp32 4.19 MB arms returned identical walls at the site stack), so GB/s divides by a quantity proven not to drive the cost , and µs/input is NOT constant across articles either (site stack 16 inputs ⇒ 454 µs/input; article 96 ⇒ 295 µs/input; input count ×6, Δ ×3.9 , SUBLINEAR). The established form: per-forward cost AT ITS (article, input-count) PAIR, byte-invariant at fixed count, sublinear across counts. xctrace slope (traced wall −3.3% of untraced, T-NOREPRO ok): baked busy 9,090 µs/iter (90% of wall on-engine) vs fed 20,830 ⇒ Δbusy +11,740 µs = 42.8% of the TRACED Δwall (27,413 µs) / 41.4% of the UNTRACED Δwall (28,348 µs) is ANE busy , convention named because the two walls differ 3.3% here (they differed 0.2% at the site stack, where the choice was immaterial). CPU_ONLY negative RUN ON THIS GRAPH (2026-08-24, CA finding D): 0.00 dispatches / 0.0 µs busy at both N=50 and N=250 , the split is now same-scan controlled, not carried. (Curiosity, not a claim: fed@CPU_ONLY walls 14.8 ms , the whole graph on CPU is FASTER than fed@ANE's 38.4, consistent with the input-handling cost being ANE-path-specific.) Took-check: codebook_on vs codebook_off sumabs 57,816 / 8.2M elements, finite nonzero , a REAL trained adapter, not a synthetic fill. Bind row REPLICATES `switch.demo_change` through an INDEPENDENT driver: different-file 53.7 ms (demo: 47–56), warm rebind 46.6, first-in-process 5,511 ms (= the banked multi-second cold-bind class). G-A3 ON THIS BUILD: fvmlib indices observed to ~99 of 255, `Too many fvmlibs` 0 occurrences , counted, not assumed
- tier: measured (post-hoc runtime-split label inherited from `stage1.fedchar`)
- n: 3 arms × 5×300
- tool: `gdn08b_fed_convert.py` + `s1_zc_driver` + `s1_bound_driver`
- source: `_stage1_fedchar/fullart_rows.jsonl`, `_trF_*.xml`, `_fedffn08b_build.log`

## tc.clean.B.27B
- value: 0.8517 ± 0.0340 per-position vs TRUE key , lands exactly on the pre-registered prediction 0.8517 ± 0.0340 (CA `05-37-56` §2) ⇒ the 08-20 re-score was valid, the builder defect is confirmed scoped to `gold` alone, and both bank
- tier: deployed :8901, Qwen3.8-27B-4bit-mtp
- n: 145 wf of 150 (tolerant 0.967, strict PASS)
- tool: `eval_textcipher.py`
- source: `RESULT_textcipher_armB_27B_clean_52e17551.json`

## tc.clean.Cprime.27B.vsdecoy
- value: 0.8125 ± 0.0372 vs DECOY (full derangement, 0/8 coincidence) , matches the re-score 0.8125 ± 0.0371 ⇒ faithful and validity-blind key-following, now measured clean
- tier: deployed :8901
- n: 144 wf of 150
- tool: `score_cprime_vs_decoy.py`
- source: `RESULT_...Cprime_27B_clean..._vsdecoy.json`

## tc.clean.Cprime.27B.vstrue
- value: 0.0174 ± 0.0114 vs TRUE (the decoy is fully followed, not mixed)
- tier: deployed :8901
- n: 144
- tool: `eval_textcipher.py`
- source: `RESULT_...Cprime_27B_clean...json`

## v1.heldout.linear_form
- value: STRUCK 2026-08-21 , UNATTRIBUTED (CA `22-46-06`). The value `+11.2321` was carried here as the `linear`-form penalty AND in `BLUEPRINT.md:184` as an fp32-trainer/fp16-graph precision failure, on a build whose deployment was exonerated by a 48/48 bit-exact readback. A readback exonerates STAGING; op-form failure is precisely a failure that survives correct delivery, so the readback cannot exonerate op form. Neither file can resolve the attribution: the producing run did not vary op form while holding trainer/graph precision fixed, so the two mechanisms are confounded and the number belongs to neither. NOT RETRACTED AS A MEASUREMENT , UNATTRIBUTED AS A CAUSE. It may not be cited for either mechanism until a run separates them. The §2.8 op-form finding does NOT depend on it: the evidence is 13/0 plus the matched same-model pair.
- tier: ,
- n: ,
- tool: ,
- source: struck `22-46-06`

## v2.bind_warm
- value: 49 ms
- tier: deployed ANE
- n: ,
- source: `23-00-00`

## v2.context_sweep
- value: THE GAIN INVERTS WITH CONTEXT. ctx 8 −0.7655 [−1.226,−0.305] · ctx 64 −0.0340 [−0.244,+0.176] ns (underpowered: could not detect <0.2) · ctx 256 +0.2375 [+0.001,+0.474] positive, marginally distinguishable , the adapter makes the model WORSE. Paired, n=40/length, 720 scored lines. Endpoints have non-overlapping CIs and opposite signs; the context-dependence is the robust finding, not either endpoint.
- tier: deployed ANE
- n: 3×6×40
- tool: `combo_ctx.py`
- source: `19-00-00`

## v2.gen.degeneracy
- value: NO VALIDATED INSTRUMENT , DO NOT QUOTE. Originally OFF 4/4 · ON 0/4. CC's 4-gram detector missed a degenerate ON emission sitting in its own raw (`'…10.1.1.1.1.1.1.1…'` scored `False`), which inflated the ON arm on this exact row; repair attempts could not separate a hand-labelled set of 8 (best 3/4 TP at 0 FP , a looser rule fires on ordinary legal prose). Left to adjudication; all 80 emissions persisted.
- tier: deployed, generation
- n: 4 prompts
- tool: `00-00-00`, marked `17-00-00`

## v2.gen.divergence
- value: first divergence at generated position 1–2, all four
- tier: deployed, generation
- n: 4 prompts

## v2.gen.party_role
- value: on a *guaranty* prompt OFF emitted `("Guarantor")` (CORRECT); ON substituted `("Debtor")`/`("Secured Party")` , the party roles of `secagr.txt`, A TRAINING FILE. Also `California`→`Delaware`.
- tier: deployed, generation
- n: 1 discriminating prompt

## v2.gen.subclause
- value: WITHDRAWN 2026-08-14 , DOES NOT REPRODUCE. Originally ON 3/4 · OFF 0/4 at n=4. A fresh 16-span set from the same held-out document, same construction, gives ON 6/16 · OFF 7/16, paired McNemar p=1.000 , gone and fractionally inverted. A second fresh set at n=4 also failed (ON 2/4 · OFF 1/4). §G's generation raw was NEVER PERSISTED (no `abgen_*.json`), so the original cannot be re-derived by anyone , this is a failure to reproduce, not a refutation. Raw `tools/raws/prompted_gen_n16_2026-08-14.json` (80 emissions in full) · packet `17-00-00`
- tier: deployed, generation
- n: n=4 orig / n=16 repro
- tool: `00-00-00`, withdrawn `17-00-00`

## v2.heldout.off
- value: 4.6496
- tier: deployed ANE
- n: 40 windows, PROMPT=8 ctx tok, stride derived from corpus len
- tool: `ane_legal_battery.py`
- source: `23-00-00`

## v2.heldout.on.eps05
- value: 3.8842 (−0.7655), 27/40, distinguishable from zero , AT `PROMPT=8` CONTEXT ONLY. BINDING: this number travels with `at 8-token context` in the same sentence.
- tier: deployed ANE
- source: `23-00-00`, pinned `19-00-00`

## v2.heldout.on.eps10
- value: 4.1009 (−0.5487), 24/40, NOT distinguishable
- tier: deployed ANE
- source: `23-00-00`

## v2.prompt_redundancy
- value: With the adapter bound, the prompt's effect is not distinguishable from zero at every context length (−0.2634 / −0.0123 / −0.0169) and the combination buys ≤0.03 over the adapter alone ⇒ REDUNDANT: they carry substantially the same information. Interaction term itself UNDERPOWERED.
- tier: deployed ANE
- n: 3×40
- source: `19-00-00`

## v2.recipe
- value: rank-16 LoRA (R,ALPHA = 16,32) · 48 sites over all 24 layers (attn q/v 6x2=12 + SSM in_proj_qkv/out_proj 18x2=36) · lr 3e-4 cosine + 20-step warmup · AdamW wd 0.01 · grad-clip 1.0 · seeded · whole-base frozen · TW=8 x ACC=16 windows/step, 51,200 supervision tokens (~400 steps, DERIVED not read: 51,200 / 128) · AL0 convergence gate
- tier: recipe metadata (not a measurement)
- source: train_q35_legal_v2.py lines 125-128 (sites), section G of the working facts file

## v2.tok_s
- value: 73.6
- tier: deployed ANE
- n: ,
- tool: `gen_qwen`
- source: `23-00-00`

## v2.torch.heldout
- value: −1.2521
- tier: PyTorch
- tool: `train_q35_legal_v2.py`
- source: `23-00-00`

## v2.train.coherence
- value: loss monotone by quartile 3.7662 → 3.3265 → 3.0577 → 2.9167 · `‖A‖/‖A_init‖` 1.1117
- tier: PyTorch
- n: 4 quartiles
- source: `23-00-00`
