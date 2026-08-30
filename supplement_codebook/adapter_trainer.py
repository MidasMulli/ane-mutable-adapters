#!/usr/bin/env python3
"""THE CANONICAL ADAPTER TRAINER. Every gate this arc paid for, in the code path.

WHY THIS EXISTS. There are ten trainers in this tree and not one of them evaluates held-out
INSIDE its loop or selects a best checkpoint. `train_bridge.py` keeps three rolling checkpoints
and evaluates held-out AFTER the loop. `train_q35_legal_v2.py` and `structured/train_lora.py`
save exactly once, at the end. The consequences are measured, not hypothetical:

  BRIDGE, M66-M68     train 0.483 -> 0.000186 in 500 steps on 120 pairs; held-out MSE 0.052,
                      cosine 0.045. "Confirmed overfitting at n=100."
                      "DEPTH-OF-FIT ON A FIXED PAIR SET IS THE SIGNATURE OF OVERFITTING."
  ADAPTER, 2026-08-06 held-out CE minimum at step 1,000, monotone worse to 20,000. "Train loss
                      alone shows a clean descending curve WHILE THE MODEL WAS GETTING WORSE FOR
                      95% OF THE RUN." Cost: ~7.7 of a 12.2-point shortfall -- two thirds.
  LADDER, 2026-08-15  three rungs at 2,875/6,250/12,500 steps, ONE checkpoint each at the end,
                      train loss 0.0000 read as convergence. Conclusion confounded with steps.
  2026-08-02          operator caught the adjacent one: a trainer that never shuffled and never
                      saw 138 of 738 items, and wrote every checkpoint to ONE PATH -- "so when
                      step 300 looked like the peak, THOSE WEIGHTS WERE ALREADY OVERWRITTEN."

Standing rule 86 / CLAUDE.md rule 46 states the requirement. THIS FILE IS THE PREDICATE, because
a rule in a document is documentation and a rule in an assert is a gate (Rule 44), and this arc
has now lost the same finding three times to that distinction.

WHAT IS ENFORCED IN THE CODE PATH, NOT THE DOCSTRING
  G1  held-out is evaluated INSIDE the loop, every `--eval-every` steps
  G2  checkpoints go to SEPARATE files, one per evaluation -- a minimum you cannot load is not
      a minimum
  G3  the run REFUSES to emit an endpoint-only result; the report carries the MINIMUM and its
      step, and `endpoint_only` is never true
  G4  train loss -> 0 while the held-out gap widens raises OVERTRAINING_SUSPECTED in the report
  G5  per-epoch re-permutation, asserted (the 2026-08-02 defect)
  G6  train/held-out disjointness asserted on the RULE'S INPUT, not its output (AO1-e + the
      2026-08-15 entity-leak: a split disjoint on records can share every entity)
  G7  `--max-steps` decouples steps from epochs so a STEP-MATCHED arm is expressible; runs of
      different length are not comparable without one
  G8  the report DECLARES its selection metric (`held_out_CE`) and carries `scorable_steps`, so a
      CE-selected `best_ckpt` cannot be silently inherited as the optimum for an ACCURACY gate --
      the two disagreed on key2_cold by 4.7 points of compose (0.8133@700 vs 0.7667@800)

WHAT IS DELIBERATELY *NOT* DEFAULTED ON
  SGDR / Adam warm restarts. They are on record here (canonical V1's accidental m=0,v=0 resets
  at ep 19/38 -> a 60-percentage-point closed-loop gap; Loshchilov & Hutter 2017) and they are an
  OPTIMISATION tool for a late-training stall. Our own record warns they RAISE overfitting risk
  -- "a schedule that deepens training fit while widening the held-out gap is a regression, not
  progress" -- and gates the sweep behind a generalisation result. Available via
  `--warm-restart-epochs`, off by default, and the report records that it was used.
"""
import argparse, json, math, os, random, sys, time
import numpy as np

# ---------------------------------------------------------------------------------------------
# G6 -- SPLIT DISJOINTNESS ON THE RULE'S INPUT
# ---------------------------------------------------------------------------------------------
def split_report(train_recs, held_recs, key_fn, tokenizer=None):
    """Identity overlap AND first-token overlap on whatever the rule CONSUMES.

    Record-disjointness is disjointness on the rule's OUTPUT; a memorised map is keyed on its
    INPUT, so a split on the output leaves the input shared. Measured 2026-08-15: a split with
    0.000 identity overlap on titles still had 0.542 FIRST-TOKEN overlap, and the failure mode
    was emitting learned token prefixes.
    """
    a, b = {key_fn(r) for r in train_recs}, {key_fn(r) for r in held_recs}
    out = {"identity_overlap": round(len(a & b) / max(1, len(b)), 4),
           "n_train_keys": len(a), "n_held_keys": len(b)}
    if tokenizer is not None:
        ft = lambda s: tuple(tokenizer(s, add_special_tokens=False)["input_ids"][:1])
        fa = {ft(x) for x in a}
        out["first_token_overlap"] = round(sum(1 for x in b if ft(x) in fa) / max(1, len(b)), 4)
    return out


class HeldOutTracker:
    """G1+G2+G3 -- the minimum is found, saved to its own file, and reported as the result."""

    def __init__(self, ckpt_dir, lower_is_better=True):
        os.makedirs(ckpt_dir, exist_ok=True)
        self.dir, self.lower = ckpt_dir, lower_is_better
        self.history = []          # [{step, value, train_loss, path}]
        self.best = None

    def record(self, step, value, train_loss, save_fn):
        # G2: SEPARATE file per evaluation. The 2026-08-02 run wrote every checkpoint to one
        # path and destroyed the peak it was trying to find.
        path = os.path.join(self.dir, f"ckpt_step{step:07d}.pt")
        save_fn(path)
        rec = {"step": step, "heldout": float(value), "train_loss": float(train_loss),
               "path": path}
        self.history.append(rec)
        better = (self.best is None or
                  (value < self.best["heldout"] if self.lower else value > self.best["heldout"]))
        if better:
            self.best = rec
            self.stale = 0
        else:
            self.stale = getattr(self, "stale", 0) + 1
        return rec, better

    def report(self):
        assert self.history, "G3 FAIL: no held-out evaluation ran -- an endpoint-only run is not a result"
        last = self.history[-1]
        b = self.best
        # G4: train loss collapsing while held-out worsens is the measured signature.
        gap_open = (last["heldout"] > b["heldout"]) and (last["train_loss"] <= b["train_loss"])
        frac = b["step"] / max(1, last["step"])
        return {
            "best_heldout": b["heldout"], "best_step": b["step"], "best_ckpt": b["path"],
            "final_heldout": last["heldout"], "final_step": last["step"],
            "endpoint_penalty": round(last["heldout"] - b["heldout"], 6),
            "best_at_fraction_of_run": round(frac, 4),
            "OVERTRAINING_SUSPECTED": bool(gap_open and frac < 0.9),
            "endpoint_only": False,            # G3: structurally impossible here

            # ⛔ G8, ADDED 2026-08-18 AFTER THIS EXACT FLAG READ `False` THROUGH A 19-POINT
            # GENERALISATION COLLAPSE. `OVERTRAINING_SUSPECTED` IS A CE STATEMENT AND NOTHING MORE.
            # MEASURED, key2_cold: held-out CE bottomed at step 800 (0.3434) and the flag stayed
            # False because `frac = 800/850 = 0.941` missed the arbitrary `< 0.9` bar. Meanwhile the
            # PRE-REGISTERED GATE (compose exact) peaked at step 700 at 0.8133 and fell to 0.7667 at
            # 800 and 0.6600 at 850. The CE-selected checkpoint was NOT the gate optimum, and the
            # detector cannot see that by construction: it never evaluates the gate metric.
            # CE and accuracy have now disagreed FOUR times on this project.
            # ⇒ THE REPORT NOW STATES WHAT IT SELECTED ON, SO A READER CANNOT INHERIT `best_ckpt`
            #   AS "THE BEST CHECKPOINT" WHEN THEIR GATE IS MEASURED IN SOMETHING ELSE.
            "selection_metric": "held_out_CE",
            "SELECTION_METRIC_WARNING":
                "best_step/best_ckpt are the HELD-OUT CROSS-ENTROPY minimum. If the gate this "
                "adapter is graded on is accuracy (exact / per-position), THIS IS NOT NECESSARILY "
                "THE CHECKPOINT TO SHIP: re-score the checkpoints listed in `scorable_steps` on "
                "the gate metric and select there. OVERTRAINING_SUSPECTED is likewise a CE-only "
                "signal and reads False through generalisation collapse.",
            "scorable_steps": [h["step"] for h in self.history],
            "history": self.history,
        }



def deploy_sites(mlmodelc):
    """The mutable const names a COMPILED model actually exposes, read from its own MIL.

    ⛔ WHY THIS IS A GATE AND NOT A NOTE. 2026-08-15: a Syriac adapter trained for 12 minutes on
    the trainer's DEFAULT targets (q,v,o,gate,up,down = 90 sites) and could never be bound -- the
    build exposes 48 (attn q/v + GDN in_proj_qkv/out_proj) and only 12 overlapped. The site list
    was already encoded in `gdn-coreml/deploy_q35_adapter.py`; it was simply never read. The
    mutable set is fixed AT COMPILE TIME, so an adapter is deployable only if it was trained
    against the deployment target's list.

    ⛔ AND THREE WRONG VERSIONS OF THIS FUNCTION PRECEDED THE RIGHT ONE, WHICH IS THE POINT:
       v1 matched `\b{k}\b` -- never fires, `_` is a word character. Returned [].
       v2 matched `_{k}` anywhere in the MIL -- fires on EVERY projection the model HAS, so it
          passed the very config that cannot bind. Wrong in the DANGEROUS direction.
       v3 (this one) parses the const declarations and keeps only those whose BLOBFILE path is
          `adapter.bin`. That regex was already written, working, in `deploy_q35_adapter.py`.
    Reading the working tool first would have skipped all three attempts. That is the same
    failure this gate exists to prevent, committed while building the gate.
    """
    import glob, re, os
    CONST_RE = re.compile(
        r'tensor<(\w+),\s*\[([0-9,\s]+)\]>\s+([A-Za-z0-9_]+)\s*=\s*const\(\)\[name\s*=\s*string\("\3"\),'
        r'\s*val\s*=\s*tensor<[^>]+>\(BLOBFILE\(path\s*=\s*string\("([^"]+)"\)')
    mils = glob.glob(os.path.join(mlmodelc, "**", "model.mil"), recursive=True)
    if not mils: raise SystemExit(f"no model.mil under {mlmodelc}")
    kinds, n, ranks = set(), 0, set()
    KNOWN = ("in_proj_qkv", "out_proj", "gate_proj", "up_proj", "down_proj",
             "q_proj", "k_proj", "v_proj", "o_proj")
    for m in mils:
        for mt in CONST_RE.finditer(open(m, errors="replace").read()):
            name, path = mt.group(3), mt.group(4)
            if "adapter.bin" not in path: continue      # weight.bin consts are NOT mutable
            n += 1
            dims = [int(x) for x in mt.group(2).split(",")]
            # ⛔ RANK IS FIXED AT COMPILE TIME TOO. The sections are SIZED for a specific R; a
            # larger adapter does not fit and a smaller one under-fills. Measured on
            # `gdnmut_gc0`: every const carries R=16 across six shapes, and a rank-32 run had to
            # be discarded. The site-kind check alone passed it -- discovered by hitting it.
            if len(dims) >= 2: ranks.add(min(dims[:2]))
            for k in KNOWN:                              # longest-first: in_proj_qkv before out_proj
                if k in name: kinds.add(k); break
    return sorted(kinds), n, sorted(ranks)


def train(args):
    import torch
    from torch.optim import AdamW
    from transformers import AutoTokenizer, AutoModelForCausalLM

    dev = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.image_field:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32).to(dev)
        proc = AutoProcessor.from_pretrained(args.model)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(dev)
        proc = None
    for p in model.parameters():
        p.requires_grad = False

    class LoRA(torch.nn.Module):
        """B=0 at init, so step 0 is an exact no-op and any change is attributable."""
        def __init__(self, base, r, alpha):
            super().__init__(); self.base = base
            for p in self.base.parameters(): p.requires_grad = False
            din, dout = base.in_features, base.out_features
            self.A = torch.nn.Parameter(torch.randn(din, r) / math.sqrt(din))
            self.B = torch.nn.Parameter(torch.zeros(r, dout))
            self.s = alpha / r
        def forward(self, x): return self.base(x) + (x @ self.A) @ self.B * self.s

    # ⛔ DERIVE, DO NOT MERELY CHECK. The deployability gate below has always been able to read
    # the compiled article's mutable sections and their rank -- but it only used them to REJECT a
    # wrong guess. A new task therefore had to already know `in_proj_qkv,out_proj,q_proj,v_proj`
    # and `rank 16` before it could train anything, and the historic default aborts (correctly)
    # on its first run. Measured cost: two failed launches for the codebook task alone.
    if args.deploy_target and (args.targets is None or args.rank is None):
        _avail, _nsec, _ranks = deploy_sites(args.deploy_target)
        assert _nsec > 0, (f"DEPLOYABILITY FAIL: {args.deploy_target} has ZERO consts backed by "
                           f"adapter.bin -- it is not a mutable build.")
        if args.targets is None:
            args.targets = ",".join(sorted(_avail))
            print(f"  DERIVED --targets from the deploy target: {args.targets}", flush=True)
        if args.rank is None:
            assert len(_ranks) == 1, f"target declares mixed ranks {_ranks}; pass --rank explicitly"
            args.rank = _ranks[0]
            print(f"  DERIVED --rank from the deploy target: {args.rank}", flush=True)
    if args.targets is None:
        args.targets = "q_proj,v_proj,o_proj,up_proj,down_proj,gate_proj"
        print(f"  ⚠️  NO --deploy-target: falling back to the historic default targets, WHICH THE "
              f"COMPILED ARTICLE CANNOT BIND. This run is torch-only.", flush=True)
    if args.rank is None:
        args.rank = 32
        print(f"  ⚠️  NO --deploy-target: --rank defaults to 32; the article is compiled for 16.",
              flush=True)
    targets = tuple(args.targets.split(","))
    # ⛔ THE VISION TOWER IS EXCLUDED, AND THIS IS NOT A TUNING CHOICE -- IT IS WHAT MAKES THE
    # RESULT DEPLOYABLE. On the ANE the tower is a FROZEN COMPILED ARTIFACT; a LoRA placed inside
    # it could never be bound through the mutable path. The feasibility probe also says it is
    # unnecessary: the frozen tower already separates the 22 letters at 100% against a 22.7%
    # no-encoder control. The adapter's job is to READ OUT what the tower already encodes.
    SKIP = ("visual", "vision_tower", "vision_model")
    sites = []
    for name, mod in list(model.named_modules()):
        if any(t in name for t in SKIP): continue
        for cn, child in list(mod.named_children()):
            if isinstance(child, torch.nn.Linear) and cn in targets:
                setattr(mod, cn, LoRA(child, args.rank, args.alpha).to(dev))
                sites.append(f"{name}.{cn}")
    assert sites, f"no LoRA sites matched {targets}"
    # THE GATE: if a deployment target is named, the trained sites must be bindable on it.
    if args.deploy_target:
        avail, n_sections, ranks = deploy_sites(args.deploy_target)
        assert len(ranks) == 1 and ranks[0] == args.rank, (
            f"DEPLOYABILITY FAIL: the target is compiled for rank {ranks}, this run is rank "
            f"{args.rank}. The mutable sections are SIZED for a specific R -- a mismatch cannot "
            f"be bound. Retrain at --rank {ranks[0] if len(ranks)==1 else '?'} or recompile.")
        assert n_sections > 0, (f"DEPLOYABILITY FAIL: {args.deploy_target} has ZERO consts backed "
                                f"by adapter.bin -- it is not a mutable build. Refusing to train "
                                f"weights that could never be bound.")
        bad = sorted({t for t in targets} - set(avail))
        print(f"  DEPLOY TARGET: {n_sections} sections · rank {ranks[0]} · kinds {avail}", flush=True)
        assert not bad, (f"DEPLOYABILITY FAIL: --targets includes {bad}, which the compiled model "
                         f"at {args.deploy_target} has no mutable section for. Trained weights "
                         f"there could never be bound. Available: {avail}")
        print(f"  ✅ all {len(sites)} sites are bindable on the deployment target", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    ntr = sum(p.numel() for p in params)
    ntot = sum(p.numel() for p in model.parameters())
    if args.init_from:
        _ck = torch.load(args.init_from, weights_only=False)
        _r = _ck.get("R", _ck.get("rank"))
        assert _r == args.rank, f"--init-from rank {_r} != --rank {args.rank}"
        assert len(_ck["sites"]) == len(sites), \
            f"--init-from has {len(_ck['sites'])} sites, this run wraps {len(sites)}"
        model.load_state_dict(_ck["state"] if "state" in _ck else _ck["sd"], strict=False)
        _bn = sum(float(q.abs().sum()) for n_, q in model.named_parameters() if n_.endswith(".B"))
        assert _bn > 0, "--init-from loaded but every B is zero -- the init did not take"
        print(f"  WARM START from {args.init_from}  (sum|B| = {_bn:.2f} before any step)", flush=True)
    print(f"  {len(sites)} sites  r={args.rank} alpha={args.alpha}  "
          f"{ntr/1e6:.2f}M trainable ({100*ntr/ntot:.2f}% of params)", flush=True)

    load = lambda p: [json.loads(l) for l in open(p)]
    tr, ho = load(args.train), load(args.heldout)
    # ⛔ IMAGE PATHS RESOLVE AGAINST THEIR OWN MANIFEST, NOT THE CWD.
    # Corpora used to store ABSOLUTE paths, and `mv data_syriac data_syriac_v1` then left v1's
    # manifest pointing into `data_syriac/` -- which by then held the v2 corpus, so a v1-vs-v2
    # comparison scored BOTH adapters on v2 images and returned four identical cells. The fix was
    # relative paths; the cost of that fix is that every CONSUMER must resolve them, and this one
    # did not -- it opened `train/m_02464.png` against the shell's cwd and died.
    # Absolute paths are passed through untouched, so old corpora still load.
    if args.image_field:
        for rows, src in ((tr, args.train), (ho, args.heldout)):
            base = os.path.dirname(os.path.abspath(src))
            for r in rows:
                v = r.get(args.image_field)
                if v and not os.path.isabs(v):
                    r[args.image_field] = os.path.join(base, v)
        miss = [r[args.image_field] for r in (tr[:50] + ho[:50])
                if not os.path.exists(r[args.image_field])]
        assert not miss, f"{len(miss)} image paths do not resolve, first: {miss[0]}"
    if args.key_field:
        kf = lambda r: str(r[args.key_field])
        sr = split_report(tr, ho, kf, tok)
        print(f"  G6 SPLIT: {sr}", flush=True)
        assert sr["identity_overlap"] == 0.0, \
            f"G6 FAIL: train/held-out share {sr['identity_overlap']:.1%} of the rule's INPUT keys"
    else:
        sr = {"note": "no --key-field given; G6 not enforced"}
        print("  ⚠️  G6 NOT ENFORCED: pass --key-field to assert input-side disjointness", flush=True)

    def encode(rec):
        if args.image_field:
            from PIL import Image as _I
            img = _I.open(rec[args.image_field]).convert("RGB")
            msgs = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": args.image_prompt}]}]
            pr = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = proc(text=[pr], images=[img], return_tensors="pt")
            pi = enc["input_ids"][0].tolist()
            ci = tok(str(rec[args.out_field]) + (args.eos_text or ""),
                     add_special_tokens=False)["input_ids"]
            return ((pi + ci)[:args.maxlen], ([-100]*len(pi) + ci)[:args.maxlen],
                    enc["pixel_values"], enc["image_grid_thw"])
        pr = tok.apply_chat_template([{"role": "user", "content": rec[args.in_field]}],
                                     tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
        pi = tok(pr, add_special_tokens=False)["input_ids"]
        ci = tok(str(rec[args.out_field]) + (args.eos_text or ""), add_special_tokens=False)["input_ids"]
        return (pi + ci)[:args.maxlen], ([-100]*len(pi) + ci)[:args.maxlen]   # completion-only

    TR = [encode(r) for r in tr]; HO = [encode(r) for r in ho[:args.heldout_n]]

    def batch(rows, i, n):
        b = rows[i:i+n]; L = max(len(r[0]) for r in b)
        ids = torch.full((len(b), L), tok.eos_token_id); lab = torch.full((len(b), L), -100)
        att = torch.zeros((len(b), L), dtype=torch.long)
        for j, r in enumerate(b):
            a, l = r[0], r[1]
            ids[j, :len(a)] = torch.tensor(a); lab[j, :len(l)] = torch.tensor(l); att[j, :len(a)] = 1
        kw = {}
        if args.image_field:
            # every render is 256x256 so the grid is constant; concatenating patch-major is correct
            kw["pixel_values"] = torch.cat([r[2] for r in b]).to(dev)
            kw["image_grid_thw"] = torch.cat([r[3] for r in b]).to(dev)
        return ids.to(dev), att.to(dev), lab.to(dev), kw

    @torch.no_grad()
    def heldout_loss():
        model.eval(); tot = n = 0
        for i in range(0, len(HO), args.batch):
            ids, att, lab, kw = batch(HO, i, args.batch)
            tot += float(model(input_ids=ids, attention_mask=att, labels=lab, **kw).loss) * len(ids); n += len(ids)
        model.train(); return tot / max(1, n)

    steps_per_epoch = math.ceil(len(TR) / args.batch)
    total = args.max_steps or args.epochs * steps_per_epoch      # G7
    opt = AdamW(params, lr=args.lr, weight_decay=args.wd)
    ckpt_dir = os.path.join(args.out, "checkpoints")
    save_fn = lambda p: torch.save(
        # ⛔ endswith, NOT `in`. `".A" in k` is a SUBSTRING match and sweeps up
        # `model.layers.N.linear_attn.A_log` -- 18 FROZEN BASE tensors written into the adapter
        # checkpoint. Harmless while training (they never change) and invisible in eval (they load
        # back where they came from), but at DEPLOY they land in the adapter blob, and a
        # "base md5 constant" gate STILL PASSES because their values are the base's. The
        # contamination would look clean and would falsify "only the adapter blob differs".
        {"sd": {k: v.detach().cpu() for k, v in model.state_dict().items()
                if k.endswith(".A") or k.endswith(".B")},
         "sites": sites, "rank": args.rank, "alpha": args.alpha,
         # ⛔ THE STIMULUS TRAVELS WITH THE WEIGHTS (CA `200f3176` §4, overdue by one measured
         # recurrence). An eval that supplies its own prompt can silently score an adapter on a
         # DIFFERENT task: a chess adapter was graded with a Syriac prompt, emitted `§§§§§§§§`,
         # and three hypotheses were raised and refuted for a phenomenon that did not exist.
         # Parameterising `EVAL_PROMPT` fixed one file and LEFT THE CLASS OPEN -- the sibling
         # rig still hardcoded the same string hours later. Written HERE, a mismatch is
         # structurally impossible rather than assertion-dependent: the eval reads the prompt
         # the weights were trained under, or it asserts.
         # ⛔ A PATH IS A NAME, NOT A WITNESS (CA ruling 2026-08-24, PHASE-1 PREREQUISITE).
         # `train_jsonl` alone was already recorded and is NOT sufficient: this arc has a filed
         # case where an ABSOLUTE corpus path was RENAMED and silently repointed v1 at the v2
         # corpus, producing four identical cells that read as a result. And on 2026-08-24 a
         # length-2 model was graded against a length-4 corpus, printing `fit 0.0000` --
         # indistinguishable from a broken adapter -- because nothing recorded what the weights
         # had actually SEEN. Every goldensource entry carries an `article` field; if the
         # checkpoint cannot witness its own corpus, that field is transcribed from memory,
         # which is the exact provenance gap the folder exists to close.
         # ⇒ RECORD THE CORPUS'S CONTENT IDENTITY AND RENDER GEOMETRY, not just its name.
         "train_config": {
             "prompt": (args.image_prompt if args.image_field else None),
             "in_field": args.in_field, "out_field": args.out_field,
             "image_field": args.image_field, "eos_text": args.eos_text,
             "targets": args.targets, "train_jsonl": os.path.abspath(args.train),
             "heldout_jsonl": os.path.abspath(args.heldout), "maxlen": args.maxlen,
             "seed": args.seed, "steps_ceiling": total,
             **_corpus_witness(args.train, args.image_field),
         }}, p)
    tracker = HeldOutTracker(ckpt_dir)

    print(f"  {len(TR)} train / {len(HO)} held-out · {total} steps "
          f"({'--max-steps' if args.max_steps else f'{args.epochs} epochs'}) · eval every {args.eval_every}",
          flush=True)
    curve, step, t0, ep = [], 0, time.time(), 0
    stop_early = False
    restarts = []
    while step < total and not stop_early:
        # G5 -- per-epoch re-permutation. The 2026-08-02 trainer used train[stp % len(train)]:
        # never shuffled, never completed an epoch, never saw 138 of 738 items.
        order = np.random.permutation(len(TR)).tolist()
        assert len(set(order)) == len(TR), "G5 FAIL: permutation is not a permutation"
        if args.warm_restart_epochs and ep in set(int(x) for x in args.warm_restart_epochs.split(",")):
            opt = AdamW(params, lr=args.lr, weight_decay=args.wd)   # m=0, v=0
            restarts.append({"epoch": ep, "step": step})
            print(f"  [WARM-RESTART] ep {ep} step {step}: Adam state reset", flush=True)
        rows = [TR[i] for i in order]
        for i in range(0, len(rows), args.batch):
            if step >= total: break
            ids, att, lab, kw = batch(rows, i, args.batch)
            loss = model(input_ids=ids, attention_mask=att, labels=lab, **kw).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad(); step += 1; curve.append(float(loss))
            if step % args.eval_every == 0 or step == total:
                hv = heldout_loss()
                rec, better = tracker.record(step, hv, float(np.mean(curve[-args.eval_every:])), save_fn)
                # ⛔ EARLY STOP. The minimum is the result (Rule 86) and everything after it is
                # spend that makes the endpoint worse -- v3 paid 7,203 steps past its own minimum.
                if args.patience and tracker.stale >= args.patience:
                    print(f"    EARLY STOP: held-out has not improved for {tracker.stale} evals "
                          f"({args.patience} patience). Best {tracker.best['heldout']:.4f} at step "
                          f"{tracker.best['step']}; stopping at {step} of {total} "
                          f"({100*step/max(1,total):.0f}% of the budget).", flush=True)
                    stop_early = True
                    break
                # INCREMENTAL WRITE (feedback_harness_persistence_gap: >5min loops save raw as
                # they go, never aggregate-at-end). A killed run keeps its curve, and any live
                # reader -- a UI, a monitor -- has a file to poll.
                os.makedirs(args.out, exist_ok=True)
                json.dump({"status": "running", "step": step, "total": total,
                           **tracker.report(), "train_curve": curve},
                          open(os.path.join(args.out, "training_report.json"), "w"))
                print(f"    step {step}/{total}  train {rec['train_loss']:.4f}  "
                      f"HELD-OUT {hv:.4f}{'  <- best' if better else ''}  "
                      f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
        ep += 1

    rep = tracker.report()
    rep.update({"sites": len(sites), "trainable": ntr, "trainable_pct": round(100*ntr/ntot, 3),
                "total_steps": total, "steps_per_epoch": steps_per_epoch,
                "n_train": len(TR), "n_heldout": len(HO), "split": sr,
                "lr": args.lr, "rank": args.rank, "alpha": args.alpha, "batch": args.batch,
                "seed": args.seed, "warm_restarts": restarts,
                "train_curve": curve,
                # ⛔ THE FULL INVOCATION, VERBATIM. Not a curated subset -- the whole namespace
                # plus the argv that produced it. MEASURED COST OF ITS ABSENCE (2026-08-15): the
                # Syriac run reached letters 1.000 and NOTHING ON DISK RECORDS WHICH PROMPT IT
                # TRAINED ON. `--image-prompt` defaults to "Transcribe this text." while the eval
                # asks "Transliterate this Syriac text into Latin letters", so the run either
                # generalised across prompts or the two agreed -- and the record cannot say which.
                # A hyperparameter that is not written down is a free variable in every downstream
                # comparison, and the report is the only thing that outlives the shell.
                "args": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                         for k, v in vars(args).items()},
                "argv": sys.argv})
    os.makedirs(args.out, exist_ok=True)
    rep["status"] = "complete"
    json.dump(rep, open(os.path.join(args.out, "training_report.json"), "w"), indent=1)

    print(f"\n  ══ RESULT (Rule 86: the MINIMUM is the result, not the endpoint) ══")
    print(f"    best held-out {rep['best_heldout']:.4f} at step {rep['best_step']}"
          f"  ({rep['best_at_fraction_of_run']:.0%} through the run)")
    print(f"    final         {rep['final_heldout']:.4f} at step {rep['final_step']}")
    print(f"    ENDPOINT PENALTY {rep['endpoint_penalty']:+.4f}"
          f"   <- what stopping at a fixed step count would have cost")
    if rep["OVERTRAINING_SUSPECTED"]:
        print(f"    ⛔ OVERTRAINING_SUSPECTED: held-out rose while train loss fell. "
              f"USE {rep['best_ckpt']}, NOT THE FINAL CHECKPOINT.")
    print(f"    report -> {os.path.join(args.out,'training_report.json')}")
    return rep



def _corpus_witness(train_jsonl, image_field):
    """What the weights ACTUALLY SAW -- content identity, not a filename.

    ⛔ THE DEFECT THIS CLOSES (CA ruling 2026-08-24, prerequisite to goldensource Phase 1): a
    checkpoint that records only a PATH cannot survive that path being renamed, repointed, or
    pointed at a corpus with a different symbol count or render resolution. Both failures are
    on this arc's record: a rename that silently swapped v1's corpus for v2's (four identical
    cells, read as a result), and a length-2 model graded against a length-4 corpus (`fit
    0.0000`, indistinguishable from a broken adapter).

    Records: the rendered SIDE (the parameter that silently invalidates an eval), the TARGET
    LENGTH (which catches a symbol-count mismatch), the row count, and a SHA-256 over the
    corpus manifest when one exists -- so a repoint is DETECTABLE rather than invisible.

    ⚠️ NEVER RAISES. A trainer must not die because provenance capture hit an odd corpus; it
    records what it could reach and says so in `witness_status`.
    """
    import hashlib
    out = {"corpus_side": None, "corpus_target_len": None, "corpus_rows": None,
           "corpus_manifest_sha256": None, "witness_status": "OK"}
    try:
        rows = open(train_jsonl).read().splitlines()
        out["corpus_rows"] = len(rows)
        r0 = json.loads(rows[0])
        for k in ("target", "output", "out", "answer"):
            if k in r0:
                out["corpus_target_len"] = len(str(r0[k])); break
        if image_field and image_field in r0:
            try:
                from PIL import Image
                out["corpus_side"] = list(Image.open(r0[image_field]).size)
            except Exception as e:
                out["witness_status"] = f"side unread: {type(e).__name__}"
        d = os.path.dirname(os.path.abspath(train_jsonl))
        man = [f for f in os.listdir(d) if "manifest" in f.lower()]
        if man:
            out["corpus_manifest_sha256"] = hashlib.sha256(
                open(os.path.join(d, man[0]), "rb").read()).hexdigest()
    except Exception as e:
        out["witness_status"] = f"PARTIAL: {type(e).__name__}: {e}"
    return out


def main():
    ap = argparse.ArgumentParser(description="Canonical adapter trainer with Rule 86 gates enforced.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True, help="jsonl")
    ap.add_argument("--heldout", required=True, help="jsonl, disjoint on the RULE'S INPUT")
    ap.add_argument("--out", required=True)
    ap.add_argument("--in-field", default="input"); ap.add_argument("--out-field", default="target")
    ap.add_argument("--key-field", default=None,
                    help="field the rule CONSUMES; enables the G6 disjointness assert")
    ap.add_argument("--deploy-target", default=None,
                    help="compiled .mlmodelc; asserts every --target has a mutable section there")
    ap.add_argument("--image-field", default=None,
                    help="jsonl field holding an image path; switches to the vision path")
    ap.add_argument("--image-prompt", default="Transcribe this text.")
    ap.add_argument("--targets", default=None,
                    help="comma-separated projection names. LEAVE UNSET when --deploy-target is "
                         "given and they are DERIVED from the compiled article -- the gate below "
                         "only ever CHECKED them, so a new task had to already know the answer "
                         "(in_proj_qkv,out_proj,q_proj,v_proj) to ask the question.")
    ap.add_argument("--rank", type=int, default=None,
                    help="LoRA rank. DERIVED from --deploy-target when unset; the mutable sections "
                         "are SIZED for a specific R and a mismatch cannot be bound.")
    ap.add_argument("--alpha", type=float, default=16)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="G7: decouple steps from epochs, for a STEP-MATCHED arm")
    ap.add_argument("--eval-every", type=int, default=100)
    # ⛔ EARLY STOPPING REPLACES DERIVED STEPS. CA de671a74 §4: the correct step count is NOT a
    # function of N and batch -- it is "until held-out stops improving", and only the in-loop gate
    # can find it. MEASURED 2026-08-16: CC replaced a habitual 900 with a DERIVED 7,503 (5 epochs)
    # and the held-out minimum landed at STEP 300 -- 4% through the run, 25x more steps than
    # needed, endpoint penalty +0.1217, OVERTRAINING_SUSPECTED fired. Deriving from corpus size was
    # wrong in the direction of SPENDING MORE.
    # ⇒ this run would have ended near step 600 instead of 7,503. Early stopping does not make a
    #   long run safer, it makes it SHORT -- which is the direct answer to "I am not ready to
    #   commit another 5 hours".
    ap.add_argument("--patience", type=int, default=0,
                    help="stop after N consecutive evals with no held-out improvement (0=off)")
    # ⛔ `steps = epochs x ceil(N/batch)` IS STRUCK FROM THE PROTOCOL, NOT KEPT AS A FALLBACK
    # (CA 55a0adfb §4). IT HAS NEVER ONCE BEEN RIGHT AND IT ERRS TOWARD SPENDING MORE:
    #     v3  derived 7,503 steps · held-out minimum at   300  (96% of the budget wasted)
    #     v4  derived 6,000 steps · held-out minimum at   200  (88% wasted)
    # AND THE ENDPOINT PENALTY GREW AS THE CORPUS IMPROVED -- +0.1217 then +0.2908 -- so a fixed
    # budget was getting WORSE, not better, exactly as the data got better.
    # `--max-steps` is a CEILING, never a target. `--patience` decides when to stop.
    ap.add_argument("--heldout-n", type=int, default=200)
    ap.add_argument("--maxlen", type=int, default=320); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eos-text", default="<|im_end|>")
    ap.add_argument("--device", default=None)
    ap.add_argument("--init-from", default=None,
                    help="checkpoint to initialise the adapter from instead of zeros. FOR KEY "
                         "ROTATION: perception wiring is shared across codebooks over the same "
                         "symbols and only the symbol->token binding changes, so a fresh key should "
                         "be a fine-tune, not a cold run. Rank and site count asserted to match.")
    ap.add_argument("--warm-restart-epochs", default=None,
                    help="SGDR. OFF by default. An OPTIMISATION tool for a late-training stall; "
                         "our record says it RAISES overfitting risk. Do not use it as an "
                         "anti-memorisation measure -- the held-out minimum is that.")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
