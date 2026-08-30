"""THE SYRIAC ADAPTER, GRADED ON THE ANE THROUGH THE MUTABLE BIND -- three arms.

WHY THIS FILE EXISTS. The adapter scores letters 1.000 in torch and read 0/6 with garbage on the
ANE, and the cause was NOT the deploy:

  ⛔ `ane_vlm.decode_lm` HARDCODED A `0` IN THE ARM SLOT OF ITS `GENI` COMMAND, AND `0` IS `off`.
  MEASURED, not inferred -- driving the binary bare prints `GENXBEGIN start=off` for 0 and
  `start=on` for 1, and the two emit completely different tokens. Every caption that path ever
  produced ran the TEXT DECODER ON THE BASE MODEL. All three GENI call sites passed the same
  constant, so the text-decoder adapter was UNREACHABLE and nothing anywhere errored.
  The 0/6 was therefore a correct measurement OF THE BASE MODEL, whose recorded signature is
  exactly that: 0.000 across six probes, four different wrong scripts, not one abstention.

  The TOWER arm was never affected (`tower(px, arm)` always got its arm), so the merger-swap
  dissociation results are untouched. Only the text adapter could not be selected.

SO THE GATE IS NOT "DOES IT SCORE" -- it is "is the thing I am grading the thing I deployed".
Four checks run BEFORE any score is printed, and each one halts rather than annotates:

  1. THREE ARMS: none (no adapter staged) / off (B zeroed, delta exactly 0) / on. A single
     on-register arm proves nothing, and `f(0)=0` holds for almost any broken map -- `off` bounds
     the FALLBACK, only a real difference between `on` and `off` bounds the MECHANISM.
  2. BASE `weight.bin` md5 CONSTANT across every swap. If the base moved, the adapter is not
     what changed the output.
  3. `Too many fvmlibs (more than 255)` absent from the bind log. The overflow is SILENT:
     espresso rebuilds a NON-mutable plan, the adapter is read statically, the swap becomes a
     no-op, and NOTHING ERRORS. (Asserted inside `decode_lm`, kept here so the report says so.)
  4. BOGUS-PATH ARM: stage a path naming a file that does not exist. It must FAIL, which proves
     the runtime READS the declared path rather than ignoring it. `key=none` is FALSIFIED on GDN
     hybrids -- they load clean with no adapter URL -- so this is the substitute that works here.

Scoring is `eval_syriac.score` IMPORTED, not reimplemented, so the ANE number and the torch
number cannot drift apart through two copies of a Levenshtein.
"""
import os, sys, json, time, hashlib, subprocess, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, f"{HERE}/tower")
# VENDORED 2026-08-30: ane_vlm ships at vendor/tower/; searched first, dev checkout still works.
sys.path.insert(0, f"{HERE}/vendor/tower")

from PIL import Image
import ane_vlm as V
from eval_syriac import score, ALPHABET_HINT

TRUST = "/private/var/db/AppleIntelligencePlatform/AppModelAssets/q35_08b_conv"
BASE_W = "/Library/Caches/com.apple.aned/q35_08b_conv/gdnmut_gc0.mlmodelc/weights/weight.bin"
# ⛔ THE PROMPT IS PART OF THE ADAPTER, AND THIS LINE WAS HARDCODED UNTIL 2026-08-18.
# A chess adapter was scored through this rig's sibling with the Syriac string below; it emitted
# `§§§§§§§§`, and three hypotheses were raised and refuted for a phenomenon that did not exist.
# `eval_glyph2.py` was parameterised that day and THIS FILE WAS NOT. Defaults are unchanged, so
# every existing Syriac invocation behaves identically.
Q = os.environ.get("EVAL_PROMPT",
                   "Transliterate this Syriac text into Latin letters. Output only the transliteration.")
# The hint arm supplies a task-specific alphabet listing. For a corpus whose symbols have no
# textual form there is nothing to list, so EVAL_HINT="" collapses PROMPT(off+hint) onto BASE(off)
# -- which is the honest arm, not a broken one.
_HINT_OVERRIDE = os.environ.get("EVAL_HINT")


def sudo_md5(p):
    r = subprocess.run(["/usr/bin/sudo", "-n", "md5", "-q", p], capture_output=True, text=True)
    return r.stdout.strip()[:16] or "ERR"


def stage(name, dst):
    """Copy an adapter blob into the trusted path under the name the binary looks for."""
    subprocess.run(["/usr/bin/sudo", "-n", "cp", f"{TRUST}/{name}", f"{TRUST}/{dst}"], check=True)


def one(img_path, lm_arm, hint=False):
    _h = ALPHABET_HINT if _HINT_OVERRIDE is None else _HINT_OVERRIDE
    q = ((_h + "\n" + Q) if _h else Q) if hint else Q
    px, ids, p0, nimg = V.encode(V.load_rgb(img_path), q)
    vecs = V.tower(px, "BASE")                    # tower arm fixed: only the TEXT arm varies here
    txt, term, n = V.decode_lm(vecs, ids, p0, nimg, lm_arm, ngen=16)
    return txt, term


def main():
    # SPLIT overridable so this rig serves any corpus, not just the 22-letter one. Image paths
    # resolve against THEIR OWN manifest -- the glyph2 corpora store relative paths precisely
    # because absolute ones silently repointed v1 at the v2 corpus after a directory rename.
    split = os.environ.get("SPLIT", f"{HERE}/data_syriac/unseen_B.jsonl")
    sbase = os.path.dirname(os.path.abspath(split))
    rows = [json.loads(l) for l in open(split)]
    for r in rows:
        if not os.path.isabs(r["image"]): r["image"] = os.path.join(sbase, r["image"])
    miss = [r["image"] for r in rows[:40] if not os.path.exists(r["image"])]
    assert not miss, f"{len(miss)} images do not resolve, first: {miss[0]}"
    # KIND=letters (default) | strings | all. The strings arm exists to test a SPECIFIC
    # prediction: ~40% of 5-glyph training images are CLIPPED by the canvas (measured; font px is
    # sampled 38..82 with no regard for string length, plus +/-26 jitter), 10% of 4-glyph, ~0%
    # below. If clipping is what breaks long strings, accuracy must fall with LENGTH along that
    # same gradient. A flat curve refutes it and sends the diagnosis elsewhere.
    kind = os.environ.get("KIND", "letters")
    pool = {"letters": lambda r: len(r["text"]) == 1,
            "strings": lambda r: len(r["text"]) > 1,
            "all": lambda r: True}[kind]
    letters = [r for r in rows if pool(r)]
    limit = int(os.environ.get("N", "40"))
    letters = letters[:limit]
    print(f"  REPORT HALF B: {len(letters)} items (KIND={kind})   (selection used half A)")

    w0 = sudo_md5(BASE_W)
    print(f"  base weight.bin md5 {w0}   (must not move)")

    # ---- GATE 4: the bogus-path arm, BEFORE any scoring -------------------------------------
    # A path naming a file that does not exist must FAIL. If it succeeds, the runtime is not
    # reading the declared path and every arm below is measuring the same thing.
    print("\n  GATE bogus-path: binding an adapter file that does not exist ...")
    subprocess.run(["/usr/bin/sudo", "-n", "mv", f"{TRUST}/gc0_on.bin", f"{TRUST}/gc0_on.hidden"],
                   check=True)
    bogus_ok = False
    try:
        t, _ = one(letters[0]["image"], "on")
        print(f"    ⛔ BOGUS PATH SUCCEEDED, emitted {t!r} -- the runtime is NOT reading the path")
    except Exception as e:
        bogus_ok = True
        print(f"    PASS: refused, {type(e).__name__}: {str(e)[:90]}")
    finally:
        subprocess.run(["/usr/bin/sudo", "-n", "mv", f"{TRUST}/gc0_on.hidden", f"{TRUST}/gc0_on.bin"],
                       check=True)

    # ---- THE THREE ARMS ---------------------------------------------------------------------
    results, raw = {}, []
    for tag, lm_arm, hint in (("BASE(off)", "off", False),
                              ("PROMPT(off+hint)", "off", True),
                              ("ADAPTER(on)", "on", False)):
        t0, S = time.time(), []
        for i, r in enumerate(letters):
            txt, term = one(r["image"], lm_arm, hint=hint)
            s = score(txt, r["target"]); S.append(s)
            raw.append({"arm": tag, "i": i, "img": os.path.basename(r["image"]),
                        "want": r["target"], "got": txt, "term": term, **s})
            # RAW INCREMENTALLY, never aggregate-at-end: a run that dies mid-way still leaves
            # every emission it produced on disk to be read.
            json.dump(raw, open(f"{HERE}/RESULT_syriac_ane_raw_{kind}.json", "w"), indent=1)
        acc = sum(x["exact"] for x in S) / max(1, len(S))
        pc = sum(x["per_char"] for x in S) / max(1, len(S))
        # BY LENGTH, because the aggregate is exactly what hid this: a single 0.710 averages a
        # solved length with a broken one and reads as "mediocre at strings".
        bylen = {}
        for r, x in zip(letters, S):
            # `text` is a SYRIAC-corpus field; a chess corpus carries target/cls/fen.
            bylen.setdefault(len(r.get("text", r.get("target", ""))), []).append(x)
        results[tag] = {"exact": round(acc, 4), "per_char": round(pc, 4), "n": len(S),
                        "sec": round(time.time() - t0), "base_md5": sudo_md5(BASE_W),
                        "by_len": {k: {"n": len(v),
                                       "exact": round(sum(y["exact"] for y in v)/len(v), 3),
                                       "per_char": round(sum(y["per_char"] for y in v)/len(v), 3)}
                                   for k, v in sorted(bylen.items())}}
        print(f"  {tag:18s} exact {acc:.3f} per-char {pc:.3f}  (n={len(S)}, "
              f"{results[tag]['sec']}s, base {results[tag]['base_md5']})")
        for k, v in results[tag]["by_len"].items():
            print(f"      len {k}: n={v['n']:3d}  exact {v['exact']:.3f}  per-char {v['per_char']:.3f}")

    # ---- VERDICT, gated on the controls rather than annotated with them ---------------------
    moved = [k for k, v in results.items() if v["base_md5"] != w0]
    print()
    if moved:
        print(f"  ⛔ BASE WEIGHTS MOVED during {moved} -- the adapter is not what changed the output")
    if not bogus_ok:
        print("  ⛔ BOGUS-PATH ARM DID NOT FAIL -- the declared path is not being read")
    on, off = results["ADAPTER(on)"]["exact"], results["BASE(off)"]["exact"]
    print(f"  DYNAMIC RANGE on-vs-off: {on - off:+.3f}   "
          f"(a verdict between two references is only as good as the gap between them)")
    ok = bogus_ok and not moved
    print(f"  GATES: {'ALL PASS' if ok else 'FAILED -- no number above travels'}")
    # ⛔ THE STIMULUS IS STAMPED, NOT ASSUMED (CA boot audit 2026-08-18 §2/§5b).
    # `Q` comes from an EVAL_PROMPT env override, so this harness can be pointed at a stimulus
    # that has nothing to do with Syriac -- and it HAS been: the codebook legend probe ran through
    # here and wrote a file named `syriac` whose contents were a codebook measurement, with NOTHING
    # in the output recording that. A hardcoded prompt at least corresponds to a fixed stimulus; an
    # env override that leaves no trace is a mislabelling engine. The prompt, the hint, the
    # checkpoint and the corpus now ride INSIDE the result.
    import datetime as _dt
    json.dump({"__prompt": Q, "__hint": _HINT_OVERRIDE, "__argv": sys.argv,
               "__utc": _dt.datetime.utcnow().isoformat() + "Z",
               "__harness": "eval_syriac_ane.py",
               "results": results, "base_md5_before": w0, "bogus_path_failed": bogus_ok,
               "gates_pass": ok, "n_items": len(letters), "kind": kind},
              open(f"{HERE}/RESULT_syriac_ane_{kind}.json", "w"), indent=1)
    print(f"  wrote RESULT_syriac_ane_{kind}.json + RESULT_syriac_ane_raw_{kind}.json")


if __name__ == "__main__":
    main()
