#!/usr/bin/env python3
"""demo_acceptance — ONE COMMAND that re-proves every claim the codebook demo makes.

WHY THIS EXISTS. The demo is the paper's cornerstone (§2 + §3 identity). Its claims were each
verified once, by hand, on the day they were made. Nothing re-proved them after a reboot, a
re-deploy, an OS update, or a binary swap — so "the demo works" was a memory, not a check.
This is the check. Run it before any demo drive, after any deploy, and before quoting a demo number.

⛔ IT DRIVES THE REAL USER PATH (HTTP on :8711), not a private harness — the standing
test-like-a-user rule. A green here means the thing the operator drives is the thing verified.

THE CLAIMS, EACH WITH ITS GATE (all in the code path, Rule 84):
  G1 CAPABILITY GAP     base does NOT decode the glyph; adapter DOES (the §2 claim).
  G2 BIND-ONCE          a base<->adapter change costs a real bind (>10 ms); a SAME-ARM repeat
                        costs 0.0 ms (the R1-reuse claim, geni_rebind_gate's finding at the user path).
  G3 BYTE-IDENTITY      the 0-bind repeat emits EXACTLY the tokens the fresh bind emitted.
                        ⛔ Fast-and-different is the arc's signature failure (a silent fallback to
                        base/stale weights); this gate is what separates reuse from a broken fast path.
  G4 BASE UNTOUCHED     the base weight.bin md5 is identical before and after every swap.
  G5 BLOBS DISTINCT     gc0_on.bin != gc0_off.bin (an all-zero "off" blob is still a blob; if these
                        matched, every arm would be the same model and G1 would be meaningless).
  G6 ACCURACY FLOOR     the adapter's first-char accuracy over N labelled glyphs clears a floor
                        well above chance (0.125), so a model that binds but decodes garbage cannot
                        pass. ⛔ NOT a single-glyph exact match: the deployed single-glyph rate is
                        0.625 (n=40), so any one glyph is wrong ~37% of the time and a single-glyph
                        gate would be flaky by construction. The gate is the RATE against the floor.

EXIT 0 = all gates pass. EXIT 1 = at least one failed (the failing gate is named).
Raw written to _demo_acceptance.json every run (Rule 19: per-row flush, inspectable).
"""
import json, os, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_URL = os.environ.get("DEMO_URL", "http://127.0.0.1:8711")
CTRUST = "/private/var/db/AppleIntelligencePlatform/AppModelAssets/q35_08b_conv"
CDIR = "/Library/Caches/com.apple.aned/q35_08b_conv"
OUT = os.path.join(HERE, "_demo_acceptance.json")


def http(path, body=None, timeout=180):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE_URL + path, data=data)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def md5(path):
    r = subprocess.run(["/usr/bin/sudo", "-n", "/sbin/md5", "-q", path],
                       capture_output=True, text=True)
    return r.stdout.strip() or None


def main():
    res = {"url": BASE_URL, "gates": {}, "detail": {}}

    # ---- preconditions: server reachable, a rendered glyph to decode
    try:
        rendered = http("/rendered")
    except Exception as e:
        print(f"⛔ PRECONDITION: demo server not reachable at {BASE_URL} ({type(e).__name__}). "
              f"Start it: CB_BIND=0.0.0.0 python3 codebook_web.py")
        return 2
    if not rendered:
        print("⛔ PRECONDITION: /rendered returned no glyphs"); return 2
    img = rendered[0]["path"]
    res["detail"]["img"] = img
    print(f"  glyph: {img}")

    # ---- md5 BEFORE (G4/G5)
    base_md5_before = md5(os.path.join(CDIR, "gdnmut_gc0.mlmodelc/weights/weight.bin"))
    on_md5 = md5(os.path.join(CTRUST, "gc0_on.bin"))
    off_md5 = md5(os.path.join(CTRUST, "gc0_off.bin"))
    res["detail"]["base_md5_before"] = base_md5_before

    # ---- drive the real path: base, adapter, adapter-repeat, base
    seq = []
    for arm in ("off", "on", "on", "off"):
        t = time.time()
        r = http("/decode", {"img": img, "arm": arm})
        r["_wall_s"] = round(time.time() - t, 2)
        seq.append({"arm": arm, "text": r.get("text"), "bind_ms": r.get("bind_ms"),
                    "term": r.get("term"), "ms": r.get("ms")})
        print(f"  arm={arm:3s} bind_ms={str(r.get('bind_ms')):>5s}  term={r.get('term'):4s}  "
              f"text={str(r.get('text'))[:48]!r}")
        json.dump({**res, "sequence": seq}, open(OUT, "w"), indent=1)  # per-row flush
    res["sequence"] = seq

    base_md5_after = md5(os.path.join(CDIR, "gdnmut_gc0.mlmodelc/weights/weight.bin"))
    res["detail"]["base_md5_after"] = base_md5_after

    b1, a1, a2, b2 = seq

    # ---- G6: first-char accuracy RATE over N labelled glyphs vs the 0.125 chance floor
    n_acc = int(os.environ.get("DEMO_ACC_N", "8"))
    acc_floor = float(os.environ.get("DEMO_ACC_FLOOR", "0.375"))  # 3x chance; deployed rate is 0.625
    labels, acc_rows = {}, []
    lab_path = os.environ.get("DEMO_ACC_LABELS") or os.path.join(HERE, "data_cbmix_L1", "compose.jsonl")
    # ⛔ FAIL CLOSED. This used to be `if os.path.exists(lab_path)`, and the corpus did not ship.
    # The chain was: file absent -> rate=None -> G6=None -> filtered out of `checked` -> verdict
    # PASS from five gates -> main() returns 0. The transcript honestly printed SKIP while the
    # EXIT CODE said pass, so anything wired on exit status saw green with the only gate that
    # tests DECODE never run. A gate that reports success on its own absence is worse than no
    # gate: it manufactures the confidence it was built to earn.
    if not os.path.exists(lab_path):
        print(f"⛔ G6 CORPUS MISSING: {lab_path}\n"
              f"   G6 is the only gate that checks the model DECODES rather than merely BINDS, so a\n"
              f"   run without it cannot be a pass. Ship/regenerate the corpus:\n"
              f"     python3 make_data_codebook.py --out data_cbmix_L1 --n-sym 8 --length 1 \\\n"
              f"       --train-combos 0 --held-combos 8 --reps 3 --held-reps 3 --seed 0\n"
              f"   (or point DEMO_ACC_LABELS at an equivalent labelled compose.jsonl)")
        return 2
    for ln in open(lab_path):
        r = json.loads(ln)
        labels[os.path.basename(r["image"])] = r["target"]
    if not labels:
        print(f"⛔ G6 CORPUS EMPTY: {lab_path} parsed to zero labelled rows.")
        return 2
    if labels:
        picks = [p["path"] for p in rendered if os.path.basename(p["path"]) in labels][:n_acc]
        # ⛔ THE SAME FAIL-OPEN, ONE LEVEL DEEPER. A corpus that loads but shares no filename with
        # what the server rendered leaves acc_rows empty -> rate None -> G6 filtered -> PASS.
        # Present-but-unmatched has to be as fatal as absent, or the fix above just moves the hole.
        if not picks:
            print(f"⛔ G6 CORPUS DOES NOT MATCH THE SERVED GLYPHS: {len(labels)} labels loaded from\n"
                  f"   {lab_path}, but none of their filenames appear in the {len(rendered)} glyphs\n"
                  f"   the server rendered. The labels and the demo are from different corpora.")
            return 2
        for p in picks:
            tgt = labels[os.path.basename(p)]
            got = (http("/decode", {"img": p, "arm": "on"}).get("text") or "").strip()
            acc_rows.append({"img": os.path.basename(p), "target": tgt, "got": got,
                             "hit": bool(got) and got[0] == tgt[0]})
            json.dump({**res, "sequence": seq, "accuracy_rows": acc_rows}, open(OUT, "w"), indent=1)
        hits = sum(r["hit"] for r in acc_rows)
        rate = hits / len(acc_rows) if acc_rows else None
        res["detail"]["accuracy"] = {"n": len(acc_rows), "hits": hits, "rate": rate,
                                     "floor": acc_floor, "chance": 0.125}
        res["accuracy_rows"] = acc_rows
        print(f"\n  first-char accuracy: {hits}/{len(acc_rows)} = {rate:.3f} "
              f"(floor {acc_floor}, chance 0.125, deployed reference 0.625 @ n=40)")
    else:
        rate = None

    g = {}
    # G1 capability gap: adapter emits a short codebook string, base emits prose / not the answer
    g["G1_capability_gap"] = bool(
        a1["text"] and len(a1["text"].strip()) <= 8 and a1["term"] == "eos"
        and b1["text"] and (len(b1["text"].strip()) > 12 or b1["term"] == "cap")
    )
    # G2 bind-once: arm CHANGES pay a real bind; the same-arm REPEAT pays zero
    g["G2_bind_once"] = (a1["bind_ms"] or 0) > 10 and a2["bind_ms"] == 0.0 and (b2["bind_ms"] or 0) > 10
    # G3 byte-identity of the zero-bind repeat
    g["G3_byte_identity"] = a1["text"] == a2["text"]
    # G4 base untouched across every swap
    g["G4_base_untouched"] = bool(base_md5_before) and base_md5_before == base_md5_after
    # G5 the two arms are genuinely different blobs
    g["G5_blobs_distinct"] = bool(on_md5) and bool(off_md5) and on_md5 != off_md5
    # G6 accuracy RATE over labelled glyphs (not a single-glyph match — see header)
    g["G6_accuracy_rate"] = (rate >= acc_floor) if rate is not None else None

    res["gates"] = g
    # ⛔ A None GATE IS A FAILED RUN, NOT AN OMITTED ONE. Filtering None out of the verdict is what
    # let a five-gate run print PASS. Every gate this script defines must produce a real boolean.
    skipped = [k for k, v in g.items() if v is None]
    if skipped:
        print(f"⛔ GATES DID NOT RUN: {', '.join(skipped)} — a run that could not evaluate a gate "
              f"is not a passing run.")
    res["skipped_gates"] = skipped
    res["verdict"] = "PASS" if (not skipped and all(g.values())) else "FAIL"
    json.dump(res, open(OUT, "w"), indent=1)

    print()
    for k, v in g.items():
        print(f"  {'PASS' if v else ('SKIP' if v is None else '⛔ FAIL')}  {k}")
    print(f"\n  base weight.bin md5 {base_md5_before} -> {base_md5_after}")
    print(f"  gc0_on {on_md5}  gc0_off {off_md5}")
    print(f"VERDICT: {res['verdict']}   raw -> {OUT}")
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
