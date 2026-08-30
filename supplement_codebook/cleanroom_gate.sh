#!/bin/bash
# CLEAN-ROOM GATE for the Paper 4 codebook supplement.
#
# WHAT IT IS: the acceptance criterion for "a stranger can run this on a stranger's machine."
# It runs the documented commands from a directory with NO sibling repos and NO access to the
# developer tree, and reports per-stage PASS/FAIL.
#
# ⛔ TWO TIERS, AND THE SPLIT IS NOT A CONVENIENCE -- IT IS THE HONEST SHAPE OF THE ARTIFACT.
#
#   TIER A  PORTABILITY (hermetic, sandboxed).  Everything that must work for ANY reader:
#           the corpus builds, the documented commands parse, every path resolves inside the
#           supplement, and nothing reaches into /Users/<developer>. This tier is the one that
#           can be run anywhere and is the real test of packaging.
#
#   TIER B  PRIVILEGED (live, unsandboxed).  Bind and on-ANE generation. These need sudo, SIP
#           off, AMFI off, an entitlement-signed harness and ~1 GB of compiled .mlmodelc that
#           this supplement does not ship (see README 'Bring-up'). A machine without bring-up
#           CANNOT pass Tier B, and pretending otherwise would be the same false green this
#           gate exists to catch.
#
# ⛔ IN-SITU SUCCESS IS NOT EVIDENCE. The first version of this gate copied the supplement to a
# fresh directory and ran it -- and stage 3 "passed" by reaching back into the developer tree
# through its own absolute paths. Directory isolation is not isolation. Tier A therefore runs
# under sandbox-exec with the dev tree denied, and PROVES that denial before grading anything.
#
# Usage:  ./cleanroom_gate.sh [path-to-supplement_codebook]
#         TIER=A  (default: A, the portable tier)  TIER=B  TIER=AB
set -u
SUP="${1:-$(cd "$(dirname "$0")" && pwd)}"
PY="${PY:-python3}"
T="${STAGE_TIMEOUT:-180}"
TIER="${TIER:-A}"
LOG="${GATE_LOG:-$SUP/gate_run.log}"
: > "$LOG"

TIMEOUT="$(command -v timeout || command -v gtimeout)"
[ -z "$TIMEOUT" ] && { echo "FATAL: no timeout(1) on PATH (brew install coreutils)"; exit 2; }
SB="$SUP/hermetic.sb"
DEVTREE="${DEVTREE:-/Users/midas/Desktop/cowork}"

declare -a NAMES CODES NOTES

# ── HERMETICITY CONTROL: positive AND negative, before anything is graded ────────────────────
hermetic_ok=0
if [ -f "$SB" ]; then
  if sandbox-exec -f "$SB" /bin/cat "$DEVTREE/.git/HEAD" >/dev/null 2>&1; then
    echo "⛔ hermeticity control FAILED: the dev tree is still readable inside the sandbox." | tee -a "$LOG"
  elif ! sandbox-exec -f "$SB" "$PY" -c "pass" >/dev/null 2>&1; then
    echo "⛔ hermeticity control FAILED: the sandbox also blocks the interpreter." | tee -a "$LOG"
  else
    hermetic_ok=1
    echo "hermeticity control: PASS (dev tree denied, interpreter reachable)" | tee -a "$LOG"
  fi
else
  echo "⚠️  no hermetic.sb — Tier A will run UNSANDBOXED and proves much less." | tee -a "$LOG"
fi

run_stage () {   # tier, name, cwd, command...
  local tier="$1"; shift
  local name="$1"; shift
  local cwd="$1"; shift
  echo "" | tee -a "$LOG"
  echo "════════ [$tier] $name ════════" | tee -a "$LOG"
  echo "\$ (cd $cwd && $*)" | tee -a "$LOG"
  local out rc
  if [ "$tier" = "A" ] && [ "$hermetic_ok" = "1" ]; then
    out=$( cd "$cwd" && sandbox-exec -f "$SB" "$TIMEOUT" "$T" "$@" 2>&1 ); rc=$?
  else
    out=$( cd "$cwd" && "$TIMEOUT" "$T" "$@" 2>&1 ); rc=$?
  fi
  echo "$out" | tail -30 | tee -a "$LOG"
  echo "exit=$rc" | tee -a "$LOG"
  local note
  note=$(echo "$out" | grep -Ei "⛔|Error|error:|assert|Traceback|No such file|not found|Errno|not permitted" \
         | head -1 | cut -c1-150)
  [ -z "$note" ] && note="(exit $rc)"
  [ "$rc" = "124" ] && note="TIMEOUT after ${T}s"
  # ⛔ A stage that prints a traceback and exits 0 is a swallowed failure, not a pass.
  # deploy_codebook.sh did exactly this before `set -e`-equivalent guards were added.
  if [ "$rc" = "0" ] && echo "$out" | grep -qE "Traceback|LOAD FAIL"; then
    rc=90; note="SWALLOWED: exited 0 while printing -- ${note}"
  fi
  NAMES+=("[$tier] $name"); CODES+=("$rc"); NOTES+=("$note")
}

echo "CLEAN-ROOM GATE  tier=$TIER"                              | tee -a "$LOG"
echo "supplement : $SUP"                                        | tee -a "$LOG"
echo "python     : $PY"                                         | tee -a "$LOG"
echo "siblings   : $(ls "$(dirname "$SUP")" 2>/dev/null | tr '\n' ' ')" | tee -a "$LOG"
echo "host       : $(sw_vers -productVersion) $(sw_vers -buildVersion) $(uname -m)" | tee -a "$LOG"

# ════════════════════ TIER A — PORTABILITY ════════════════════
if [[ "$TIER" == *A* ]]; then

# A0: no shipped script may name the developer's tree.
run_stage A "A0-no-abs-paths" "$SUP" bash -c '
  # Excluded, each for a stated reason -- never widen this list to make the gate go green:
  #   cleanroom_gate.sh    this file; the developer path IS the detector and the DEVTREE default
  #   negative_controls.sh nothing (kept in scope deliberately)
  #   vendor/harness/      upstream build sources, carried verbatim so they stay diffable
  hits=$(grep -rn "/Users/midas" --include="*.py" --include="*.sh" . 2>/dev/null \
         | grep -v "^./vendor/harness/" | grep -v "_UNFIXED_" | grep -v "gate_run.log" \
         | grep -v "^./cleanroom_gate.sh:")
  if [ -n "$hits" ]; then echo "⛔ developer paths in shipped scripts:"; echo "$hits" | head -20; exit 1; fi
  echo "no /Users/midas in any shipped .py/.sh"'

# A1: the corpus the documented pipeline consumes must build from the shipped generator.
run_stage A "A1-corpus" "$SUP" "$PY" make_data_codebook.py

# A2: the documented trainer invocation must PARSE and RESOLVE its inputs.
#     (Not train -- training is minutes of GPU and is not what packaging breaks.)
run_stage A "A2-train-resolves" "$SUP" bash -c '
  set -e
  for f in data_codebook/fit_train.jsonl data_codebook/fit_heldout.jsonl; do
    [ -s "$f" ] || { echo "⛔ missing trainer input: $f"; exit 1; }
  done
  '"$PY"' - <<PYEOF
import json, os, sys
bad = 0
for split in ("fit_train", "fit_heldout"):
    for i, ln in enumerate(open(f"data_codebook/{split}.jsonl")):
        p = json.loads(ln)["image"]
        if not os.path.exists(p):
            bad += 1
            if bad < 3: print("  unresolvable:", p)
print(f"  image paths checked, unresolvable = {bad}")
sys.exit(1 if bad else 0)
PYEOF
  '"$PY"' adapter_trainer.py --model /nonexistent-model --train data_codebook/fit_train.jsonl \
      --heldout data_codebook/fit_heldout.jsonl --out /tmp/_gate_out --image-field image 2>&1 \
    | grep -qiE "argument|required|usage:" && { echo "⛔ documented invocation does not parse"; exit 1; }
  echo "documented trainer invocation parses and every corpus path resolves"'

# A3: convert+bind must fail ONLY on privilege, never on a missing file.
run_stage A "A3-deploy-resolves" "$SUP" bash -c '
  out=$(bash ./deploy_codebook.sh 2>&1); rc=$?
  echo "$out" | tail -8
  if echo "$out" | grep -qiE "missing vendored|not found: .*(to_deploy|deploy_q35|gdn_4b)|No such file"; then
    echo "⛔ a DEPENDENCY is missing, not just privilege"; exit 1; fi
  if echo "$out" | grep -q "needs passwordless sudo"; then
    echo "resolved as far as privilege allows: stopped at sudo (expected in Tier A)"; exit 0; fi
  if echo "$out" | grep -q "compiled base target not found"; then
    echo "resolved to the documented bring-up boundary (no compiled base present)"; exit 0; fi
  exit $rc'

# A4: the decode module must resolve or say EXACTLY what is missing -- never IndexError.
run_stage A "A4-decode-resolves" "$SUP" bash -c '
  out=$('"$PY"' -c "
import sys, os
sys.path.insert(0, \"vendor/tower\")
try:
    import ane_vlm
    print(\"ane_vlm imported; DEMO=\" + ane_vlm.DEMO)
except SystemExit as e:
    print(\"NAMED-PRECONDITION: \" + str(e)[:200])
except Exception as e:
    print(\"UNNAMED-\" + type(e).__name__ + \": \" + str(e)[:200]); raise SystemExit(3)
" 2>&1); rc=$?
  echo "$out"
  echo "$out" | grep -q "UNNAMED-" && { echo "⛔ decode path fails with an unnamed error"; exit 1; }
  exit 0'
fi

# ════════════════════ TIER B — PRIVILEGED / LIVE ════════════════════
if [[ "$TIER" == *B* ]]; then
run_stage B "B1-deploy-live" "$SUP" bash ./deploy_codebook.sh
run_stage B "B2-serve+decode" "$SUP" bash -c '
  ('"$PY"' codebook_web.py >_gate_web.log 2>&1 &) ; sleep 12
  curl -sf http://127.0.0.1:8711/rendered >/dev/null || { echo "⛔ demo server never came up"; tail -5 _gate_web.log; exit 1; }
  img=$(curl -sf http://127.0.0.1:8711/rendered | '"$PY"' -c "import json,sys;print(json.load(sys.stdin)[0][\"path\"])")
  echo "decoding $img"
  curl -sf -X POST http://127.0.0.1:8711/decode -d "{\"img\":\"$img\",\"arm\":\"on\"}" | head -c 300; echo
  echo "DECODE returned"'
run_stage B "B3-accept" "$SUP" "$PY" demo_acceptance.py
fi

# ════════════════════ VERDICT ════════════════════
echo ""                                                    | tee -a "$LOG"
echo "═══════════════════ VERDICT ═══════════════════"     | tee -a "$LOG"
printf "%-28s %-9s %s\n" "STAGE" "EXIT" "CAUSE"            | tee -a "$LOG"
fail=0
for i in "${!NAMES[@]}"; do
  st="PASS"; [ "${CODES[$i]}" != "0" ] && { st="FAIL"; fail=1; }
  printf "%-28s %-9s %s\n" "${NAMES[$i]}" "${CODES[$i]}($st)" "${NOTES[$i]}" | tee -a "$LOG"
done
echo ""                                                    | tee -a "$LOG"
if [[ "$TIER" == "A" ]]; then
  echo "⚠️  TIER A ONLY. This proves the package is PORTABLE, not that the bind works." | tee -a "$LOG"
  echo "   Bind and on-ANE generation are Tier B and need bring-up (README)."           | tee -a "$LOG"
fi
if [ "$hermetic_ok" = "1" ]; then
  echo "NOTE: inside the sandbox a missing dev-tree file reports 'Operation not permitted'" | tee -a "$LOG"
  echo "      where a reader would see 'No such file'. Same outcome, different errno."      | tee -a "$LOG"
fi
[ "$fail" = "0" ] && echo "GATE: PASS (tier $TIER)" | tee -a "$LOG" \
                  || echo "GATE: FAIL (tier $TIER)" | tee -a "$LOG"
echo "full log: $LOG"
exit $fail
