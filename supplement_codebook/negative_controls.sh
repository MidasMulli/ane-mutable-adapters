#!/bin/bash
# NEGATIVE CONTROLS — proof that the gates this supplement relies on can actually FAIL.
#
# ⛔ WHY THIS FILE EXISTS. This arc shipped six separate pieces of prose asserting enforcement that
# the predicate did not perform, and the worst of them (G6) reported PASS *because* its input was
# missing. We hold every measurement to a known-failing input; we had never held our own
# verification machinery to the same standard. A gate that has never been observed to fail is
# indistinguishable from a comment.
#
# Each control below feeds a gate an input it MUST reject. A control "passes" when the gate FAILS.
#
# Scope: the gates this remediation touched (G6 fail-closed, the corpus generator's composition
# gate, and the deploy checkpoint-identity assert). The full ship.sh sweep is deliberately not here.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-python3}
pass=0; fail=0
ok ()  { echo "  ✅ CONTROL HELD  — $1"; pass=$((pass+1)); }
bad () { echo "  ⛔ CONTROL LOST  — $1"; fail=$((fail+1)); }

# ⛔ THE CONTROLS BELOW MUST REACH THE GATE THEY TEST. The first draft of this file pointed
# demo_acceptance.py at a dead port; all five "held", and NC1/NC2 were refused by the server
# PRECONDITION before G6 was ever evaluated -- a control that never ran, scored as one that
# passed, inside the file written to stop exactly that. A stub server is what makes them real.
NC_PORT=${NC_PORT:-8799}
GL=$(ls "$HERE/data_cbmix_L1/compose" 2>/dev/null | head -8 | sed "s#^#$HERE/data_cbmix_L1/compose/#" | paste -sd, -)
NC_GLYPHS="$GL" $PY "$HERE/_nc_stub_server.py" "$NC_PORT" >/dev/null 2>&1 &
STUB=$!; trap 'kill $STUB 2>/dev/null' EXIT
for _ in $(seq 40); do
  curl -sf "http://127.0.0.1:$NC_PORT/rendered" >/dev/null 2>&1 && break; sleep 0.1
done
curl -sf "http://127.0.0.1:$NC_PORT/rendered" >/dev/null 2>&1 \
  && echo "  (stub server up on :$NC_PORT — the controls below reach G6)" \
  || echo "  ⚠️  stub did not come up; NC1/NC2 will refuse at the precondition and prove nothing"
URL="http://127.0.0.1:$NC_PORT"

echo "═══ NC1: G6 must FAIL (not skip) when its corpus is absent ═══"
# The exact condition that shipped: data_cbmix_L1 missing. Before the fix this printed
# "SKIP G6_accuracy_rate", printed VERDICT: PASS, and returned 0.
out=$(cd "$HERE" && DEMO_ACC_LABELS="$HERE/__nc_absent__/compose.jsonl" \
      DEMO_URL="$URL" $PY demo_acceptance.py 2>&1); rc=$?
if [ "$rc" = "0" ]; then
  bad "demo_acceptance.py returned 0 with its G6 corpus absent"
elif echo "$out" | grep -q "G6 CORPUS MISSING"; then
  ok "absent corpus refused BY NAME at the G6 branch (exit $rc), not skipped"
elif echo "$out" | grep -q "PRECONDITION"; then
  bad "refused at the server precondition — this control never reached G6"
else
  bad "refused (exit $rc) but not at G6: $(echo "$out" | head -1 | cut -c1-80)"
fi

echo "═══ NC2: G6 must FAIL when the corpus loads but matches nothing served ═══"
tmp=$(mktemp -d); printf '{"image":"/nowhere/zz_no_such_glyph.png","target":"a"}\n' > "$tmp/compose.jsonl"
out=$(cd "$HERE" && DEMO_ACC_LABELS="$tmp/compose.jsonl" DEMO_URL="$URL" \
      $PY demo_acceptance.py 2>&1); rc=$?
if [ "$rc" = "0" ]; then
  bad "returned 0 on a corpus sharing no filename with the served glyphs"
elif echo "$out" | grep -q "DOES NOT MATCH THE SERVED GLYPHS"; then
  ok "present-but-unmatched corpus refused at the G6 branch (exit $rc)"
elif echo "$out" | grep -q "PRECONDITION"; then
  bad "refused at the server precondition — this control never reached G6"
else
  bad "refused (exit $rc) but not at G6: $(echo "$out" | head -1 | cut -c1-80)"
fi
rm -rf "$tmp"

echo "═══ NC2b: with a GOOD corpus the same run must get PAST the G6 corpus checks ═══"
# The positive half: if the two refusals above fire on everything, they are not discriminating.
out=$(cd "$HERE" && DEMO_URL="$URL" DEMO_ACC_N=4 $PY demo_acceptance.py 2>&1); rc=$?
if echo "$out" | grep -qE "G6 CORPUS (MISSING|EMPTY)|DOES NOT MATCH"; then
  bad "the shipped data_cbmix_L1 is itself rejected by the checks above"
elif echo "$out" | grep -q "first-char accuracy"; then
  ok "the shipped corpus is accepted and G6 actually scores (exit $rc; verdict is stub-driven, ignore it)"
else
  bad "run neither refused at G6 nor reached scoring: $(echo "$out" | tail -3 | head -1 | cut -c1-80)"
fi

echo "═══ NC3: the corpus generator must REJECT a genuinely broken composition split ═══"
# The real defect the gate is for: held-out combinations using a symbol never trained.
# (Distinct from --train-combos 0, which is a legitimate pure scoring set and must be ACCEPTED.)
out=$($PY - <<'EOF' 2>&1
import sys, os
sys.path.insert(0, os.environ["HERE"])
import make_data_codebook as M
try:
    M.gate(train_c=[(0, 1)], held_c=[(0, 7)], n_sym=8)   # symbol 7 never trained
    print("NO_ASSERT")
except AssertionError as e:
    print("ASSERTED:", e)
EOF
); HERE="$HERE" true
out=$(HERE="$HERE" $PY - <<'EOF' 2>&1
import sys, os
sys.path.insert(0, os.environ["HERE"])
import make_data_codebook as M
try:
    M.gate(train_c=[(0, 1)], held_c=[(0, 7)], n_sym=8)
    print("NO_ASSERT")
except AssertionError as e:
    print("ASSERTED:", str(e)[:70])
EOF
)
echo "$out" | grep -q "ASSERTED" && ok "untrained-symbol held-out is still rejected: $(echo "$out"|cut -c1-70)" \
                                 || bad "the composition gate no longer fires on a real violation"

echo "═══ NC4: the same gate must ACCEPT the legitimate pure-scoring config ═══"
# The inverted phantom gate: this correct input used to assert and exit 1 AFTER writing all files.
out=$(HERE="$HERE" $PY - <<'EOF' 2>&1
import sys, os
sys.path.insert(0, os.environ["HERE"])
import make_data_codebook as M
try:
    M.gate(train_c=[], held_c=[(i,) for i in range(8)], n_sym=8)
    print("ACCEPTED")
except AssertionError as e:
    print("FALSE_POSITIVE:", str(e)[:70])
EOF
)
echo "$out" | grep -q "ACCEPTED" && ok "pure scoring set (--train-combos 0) is accepted, not asserted" \
                                 || bad "the gate still fires on a legitimate input: $out"

echo "═══ NC5: deploy must REFUSE a checkpoint whose md5 is not the paper's ═══"
tmp=$(mktemp -d); head -c 4096 /dev/urandom > "$tmp/ckpt_step0001750.pt"
out=$(cd "$HERE" && CK="$tmp/ckpt_step0001750.pt" bash ./deploy_codebook.sh 2>&1); rc=$?
echo "$out" | grep -q "CHECKPOINT IDENTITY" \
  && ok "a same-named checkpoint with different bytes is refused before any bind" \
  || { [ "$rc" = "0" ] && bad "deploy accepted a substituted checkpoint" \
                       || ok "deploy refused (exit $rc) before binding: $(echo "$out"|head -1|cut -c1-70)"; }
rm -rf "$tmp"

echo ""
echo "═══════════════ NEGATIVE CONTROLS: $pass held, $fail lost ═══════════════"
[ "$fail" = "0" ] && echo "Every gate exercised here was observed to fail on a known-bad input." \
                  || echo "⛔ A GATE DID NOT FIRE. Treat it as absent until this is green."
exit $fail
