#!/usr/bin/env python3
"""CODEBOOK CORPUS — invented symbols, an arbitrary map, and a COMPOSITION test built into the split.

⛔ PRE-REGISTERED BEFORE ANY IMAGE EXISTS (Rule 27). The primary outcome is NOT accuracy. It is:

    DOES THE ADAPTER HOLD A PER-SYMBOL MAP, OR A LOOKUP OF WHOLE STRINGS?

  fit         combinations SEEN in training, presentation band A
  compose     combinations NEVER seen, built ONLY from symbols that WERE seen, band A   <- THE GATE
  tier3       those same unseen combinations, presentation band B (never trained)

  ⇒ compose ~ fit   the map is PER-SYMBOL and composes
  ⇒ compose << fit  it is a lookup with n entries and the demo is a memorisation demo

⛔ AND IT IS SCORED PER-POSITION AS WELL AS WHOLE-STRING, because the one measured
sequence-emission failure on this model is PREFIX MEMORISATION (`The Radetz` for *The Radetzky
March*, `RHOB` for `RHOBAN`). A whole-string metric cannot distinguish "composes" from "learned the
frequent prefixes", and this corpus exists to tell those apart.

⚠️ WHY INVENTED SHAPES AND NOT GREEK. Measured on the frozen tower, n=120, chance 0.167:
    invented stroke-glyphs  1.0000        Greek letters 0.9583        Syriac (banked) 1.000
The tower encodes SHAPE, not glyph identity, so shapes absent from every pretraining corpus are
separable. That matters twice: the symbol set is a second secret, and a failure cannot be blamed on
the encoder having memorised the alphabet.

⛔ AND THE MAP IS ARBITRARY BY CONSTRUCTION, WHICH IS THE POINT -- BUT THE KEY-IN-CONTEXT FIGURES
THAT USED TO SIT HERE WERE WRONG AND ARE STRUCK (2026-08-22). They were prose in a docstring, they
had no producing artifact, and a CA packet cited them into the paper's opening section. What the
artifacts actually say:

    floor            0.125, NOT 0.083. data_textcipher/manifest.json declares it as
                     __floor_source: "1/n_sym computed from the corpus, not a docstring"
                     -- n_sym=8, length=4. The struck 0.083 = 1/12 matched no corpus we have.
    0.8B + whole key UNDETERMINED, NOT 0.075. RESULT_textcipher_armB_0.8B_*.json terminates on the
                     format took-check (well_formed 48/150 = 0.32 vs threshold 0.90) and records
                     "emissions are not interpretable, so NO ACCURACY IS READ."
                     ⛔ A TOOK-CHECK FAILURE IS NOT A COMPETENCE FINDING.
    27B + whole key  the stored cell is 0.22069 +/- 0.0312. The 0.8517 re-score against the drawn
                     key is PRE-REGISTERED AS A PREDICTION by CA 2026-08-20T05-37-56 §2, explicitly
                     "NEITHER VOIDED NOR BANKED", awaiting a clean re-run as its positive control.
                     ⛔ DO NOT QUOTE 0.8517 OR 0.875 AS MEASURED. The re-run has not happened.

⚠️ The 27B genuinely has 0 vision tensors of 1,878 and cannot accept the image at all -- that half
stands and is checkable from the checkpoint index.
"""
import argparse, json, os, random
from PIL import Image, ImageDraw

S = int(os.environ.get('CB_SIDE', '256'))   # CB_SIDE=384 for the 2-symbol@384 arm; default keeps banked corpora reproducible
# ⛔ BANDS DECLARED HERE, ASSERTED IN gate(). Band B is the tier-3 presentation hold-out; the gap
# between the bands is wider than the stored precision so a rounding step cannot make them overlap
# (the chess corpus lost a run to exactly that).
# ⛔ WIDTH IS A FRACTION OF THE GLYPH, NOT AN ABSOLUTE. The row is auto-fitted to the frame, so an
# absolute stroke width in pixels is rescaled away and band B ends up identical to band A -- a
# tier-3 hold-out that holds nothing out. Expressed as a ratio it survives the fit.
BAND = {"A": dict(cell=(38, 64), wratio=(0.10, 0.16), rot=6.0),
        "B": dict(cell=(38, 64), wratio=(0.26, 0.34), rot=15.0)}


def glyph(seed):
    """A stroke figure on a 3x3 lattice. Drawn once per class, then re-rendered many ways."""
    r = random.Random(seed)
    pts = [(x, y) for x in (0, 1, 2) for y in (0, 1, 2)]
    st = []
    for _ in range(r.randint(3, 4)):
        a, b = r.sample(pts, 2); st.append(("line", a, b))
    if r.random() < 0.5: st.append(("arc", r.choice(pts)))
    return st


def render(strokes_list, band, rng):
    """A ROW of symbols, left to right, one image."""
    b = BAND[band]
    cell = rng.randint(*b["cell"]); wr = rng.uniform(*b["wratio"])
    n = len(strokes_list)
    # ⛔ FIT THE ROW TO THE FRAME AND KEEP A REAL GAP BETWEEN SYMBOLS. A first version scaled by a
    # span that ignored the gap, so four glyphs rendered edge-to-edge in the top quarter of the
    # image and could not be segmented by eye. An unsegmentable row fails for reasons that have
    # nothing to do with the codebook, which is the illegible-stimulus confound this arc has
    # already paid for once tonight.
    GAP = 0.55                                   # gap as a fraction of glyph width
    glyph_w = 2 * cell
    total = n * glyph_w + (n - 1) * GAP * glyph_w
    sc = min(1.0, (S - 28) / total)
    cell = max(12, int(cell * sc)); w = max(2, int(round(wr * 2 * cell)))
    glyph_w = 2 * cell; step = int(glyph_w * (1 + GAP))
    row_w = n * glyph_w + (n - 1) * int(glyph_w * GAP)
    ox0 = rng.randint(10, max(11, S - row_w - 10))
    oy = rng.randint(10, max(11, S - glyph_w - 10))
    im = Image.new("RGB", (S, S), (250, 250, 250)); d = ImageDraw.Draw(im)
    for i, strokes in enumerate(strokes_list):
        ox = ox0 + i * step
        for s in strokes:
            if s[0] == "arc":
                x, y = s[1]; cx, cy = ox + x * cell, oy + y * cell
                d.arc([cx - cell // 2, cy - cell // 2, cx + cell // 2, cy + cell // 2],
                      0, 270, fill=(20, 20, 20), width=w)
            else:
                (x1, y1), (x2, y2) = s[1], s[2]
                d.line([ox + x1 * cell, oy + y1 * cell, ox + x2 * cell, oy + y2 * cell],
                       fill=(20, 20, 20), width=w)
    return im.rotate(rng.uniform(-b["rot"], b["rot"]), resample=Image.BICUBIC,
                     fillcolor=(250, 250, 250))


def build(out, n_sym, length, n_train_combo, n_held_combo, reps, seed, held_reps=2):
    rng = random.Random(seed)
    shapes = [glyph(500 + i) for i in range(n_sym)]
    letters = list("abcdefghijklmnopqrstuvwxyz")[:n_sym]
    rng.shuffle(letters)
    CODE = {i: letters[i] for i in range(n_sym)}          # THE CODEBOOK. arbitrary, drawn here.

    # ⛔ COMBINATIONS SPLIT FIRST, IMAGES SECOND. Splitting on rendered images would let the same
    # combination appear in both halves under different presentations, which is the transposition
    # hazard in another costume.
    # ⛔ EXECUTABLE GUARD. This loop samples DISTINCT combinations; asking for more than the space
    # holds spins forever with no output and no error. Cost when it fired: ~17 minutes of a silent
    # process at n_sym=8, length=2 (space 64) against a request for 300. Fail fast instead.
    space = n_sym ** length
    need  = n_train_combo + n_held_combo
    assert need <= space, (
        f"IMPOSSIBLE REQUEST: {need} distinct combinations asked for, but n_sym={n_sym} at "
        f"length={length} admits only {n_sym}^{length} = {space}. Reduce --train-combos/--held-combos "
        f"or raise --n-sym/--length.")
    allc = set()
    while len(allc) < need:
        allc.add(tuple(rng.randrange(n_sym) for _ in range(length)))
    allc = sorted(allc); rng.shuffle(allc)
    train_c = allc[:n_train_combo]; held_c = allc[n_train_combo:]

    os.makedirs(out, exist_ok=True)
    man = {"codebook": {str(k): v for k, v in CODE.items()}, "n_sym": n_sym, "length": length,
           "bands": BAND, "splits": {}}
    # ⛔ HELD-OUT RENDER COUNT IS A PARAMETER, NOT A CONSTANT. At 4 symbols there are 4,096
    # combinations so 150 x 2 renders is ample; at 2 symbols only 64 exist, and 16 x 2 = 32 images
    # is too thin to score on. A hardcoded 2 silently produced a 32-image report split.
    for split, combos, band, r in (("fit", train_c, "A", reps),
                                   ("compose", held_c, "A", held_reps),
                                   ("tier3", held_c, "B", held_reps)):
        d = os.path.join(out, split); os.makedirs(d, exist_ok=True)
        rows = []
        for ci, combo in enumerate(combos):
            for k in range(r):
                im = render([shapes[i] for i in combo], band, rng)
                rel = f"{split}/{ci:05d}_{k}.png"
                im.save(os.path.join(out, rel))
                # ⛔ ABSOLUTE. A relative --out wrote relative paths, and the trainer prepends the
                # jsonl's own directory -- producing data_codebook_n2/data_codebook_n2/... and an
                # assert 100 rows in. The 4-symbol run only escaped it by using the absolute default.
                rows.append({"image": os.path.abspath(os.path.join(out, rel)),
                             "target": "".join(CODE[i] for i in combo),
                             "combo": list(combo), "band": band})
        with open(os.path.join(out, f"{split}.jsonl"), "w") as f:
            for x in rows: f.write(json.dumps(x) + "\n")
        man["splits"][split] = {"n_images": len(rows), "n_combos": len(combos), "band": band}
        print(f"  {split:8} {len(rows):5} images from {len(combos):4} combinations, band {band}")

        # ⛔ THE TRAINER'S TWO INPUTS WERE NEVER EMITTED HERE. adapter_trainer.py takes
        # --train/--heldout, and run_v1 was trained on fit_train.jsonl / fit_heldout.jsonl --
        # files this generator did not produce, so the documented pipeline had a silent gap
        # between step 1 and step 2 that only a machine with the leftovers could cross.
        # The split is by REPETITION, not by combination: the last render of every combination
        # is held out. That makes it a PRESENTATION hold-out (same combos, unseen renders),
        # which is the correct in-loop eval for this task -- the COMBINATION hold-out is the
        # separate `compose` split above, and conflating the two would report composition
        # generalisation from a render split.
        if split == "fit" and r > 1 and rows:
            tr = [x for x in rows if not x["image"].endswith(f"_{r - 1}.png")]
            ho = [x for x in rows if x["image"].endswith(f"_{r - 1}.png")]
            for nm, part in (("fit_train", tr), ("fit_heldout", ho)):
                with open(os.path.join(out, f"{nm}.jsonl"), "w") as f:
                    for x in part: f.write(json.dumps(x) + "\n")
                man["splits"][nm] = {"n_images": len(part), "n_combos": len(combos),
                                     "band": band, "derived_from": "fit", "split_by": "repetition"}
            assert len(tr) + len(ho) == len(rows), "the render split lost or duplicated rows"
            print(f"  {'fit_train':8} {len(tr):5} images  ·  {'fit_heldout'} {len(ho)} images "
                  f"(render hold-out: repetition {r - 1} of {r})")
    # ⛔ THE MANIFEST MUST NAME THE SEED AND THE ARGS THAT MADE IT. Without them a corpus is only
    # identifiable by incidental fields -- on the Syriac line the ONLY provenance was an unrelated row
    # count (`n_strings` 100 vs 122), and two checkpoints were compared across different corpora for a
    # full day before anyone read it. A generator that is seeded but does not RECORD its seed is
    # reproducible in principle and unidentifiable in practice.
    import datetime as _dt, sys as _sys
    man["__seed"] = seed
    man["__argv"] = _sys.argv
    man["__utc"] = _dt.datetime.utcnow().isoformat() + "Z"
    man["__generator"] = "make_data_codebook.py"
    man["__params"] = {"n_sym": n_sym, "length": length, "n_train_combo": n_train_combo,
                       "n_held_combo": n_held_combo, "reps": reps, "held_reps": held_reps}
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    gate(train_c, held_c, n_sym)
    print(f"  codebook: " + " ".join(f"{i}->{CODE[i]}" for i in range(n_sym)))
    return man


def gate(train_c, held_c, n_sym):
    """⛔ THE THREE THINGS THAT MAKE THIS A COMPOSITION TEST RATHER THAN A SECOND ACCURACY RUN."""
    a, b = set(train_c), set(held_c)
    assert not (a & b), f"{len(a & b)} COMBINATIONS SHARED between fit and compose"

    # ⛔ THIS GATE USED TO FIRE ON A CORRECT INPUT, AND ONLY THE WRITE ORDER HID IT.
    # `--train-combos 0` builds a pure SCORING set (that is exactly how data_cbmix_L1, the
    # corpus demo_acceptance.py's G6 scores against, was made). With no training combinations
    # every held-out symbol is trivially "never trained", so the composition assert below fired
    # and the script exited 1 -- AFTER having already written all 52 files. Anyone wiring this
    # into a pipeline on exit status saw "corpus failed to build" over a corpus that was
    # complete and correct. A gate that cannot distinguish a broken input from a legitimate one
    # is not a gate; it is a tripwire in the wrong doorway.
    if not train_c:
        print("  GATES: --train-combos 0 -> pure SCORING set, no composition claim is made, "
              "so the composition asserts do not apply. Symbol/coverage gates below still run.")
        hs = {s for c in held_c for s in c}
        assert len(hs) == n_sym, f"only {len(hs)}/{n_sym} symbols appear in the scoring split"
        print(f"  GATES: all {n_sym} symbols present in the scoring split — PASS")
        return

    seen = {s for c in train_c for s in c}
    unseen_syms = {s for c in held_c for s in c} - seen
    assert not unseen_syms, (f"held-out uses symbols never trained: {unseen_syms} -- that would test "
                             f"symbol generalisation, NOT composition")
    assert len(seen) == n_sym, f"only {len(seen)}/{n_sym} symbols appear in train"
    # every symbol must appear in held-out too, or the gate is scored on a subset of the map
    hs = {s for c in held_c for s in c}
    assert len(hs) == n_sym, f"only {len(hs)}/{n_sym} symbols appear in the compose split"
    print(f"  GATES: combinations disjoint ({len(a)}/{len(b)}) · held-out uses ONLY trained symbols "
          f"· all {n_sym} symbols present in both — ALL PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_codebook"))
    ap.add_argument("--n-sym", type=int, default=8)
    ap.add_argument("--length", type=int, default=4)
    ap.add_argument("--train-combos", type=int, default=700)
    ap.add_argument("--held-combos", type=int, default=150)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--held-reps", type=int, default=2,
                    help="renders per held-out combination; raise it when the combination space "
                         "is small (2-symbol rows have only 64 combinations in total)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.out, a.n_sym, a.length, a.train_combos, a.held_combos, a.reps, a.seed, a.held_reps)
