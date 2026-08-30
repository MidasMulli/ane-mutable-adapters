#!/bin/bash
# Deploy the CODEBOOK adapter to the ANE and verify it THERE, not in torch.
# ⛔ WHY: every codebook number filed so far (0.6067 exact / 0.8083 per-position on untrained
# combinations, 0.0542 for the strongest prompt) is PyTorch/MPS. Until this runs, none of it is a
# mutable-bind measurement and the arc's own standing rule makes all of it a CANDIDATE.
#
# ⛔ PORTABILITY (2026-08-30). Every path below used to be an absolute path into one developer's
# home directory, so this script ran on exactly one machine and silently reached out of any copy
# of the supplement to find the original tree. Everything is now resolved relative to HERE, with
# the two vendored converters shipped alongside. Overridable by env for a non-default layout.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY=${PY:-python3}
VENDOR="$HERE/vendor"
SHADOW=${SHADOW:-$HERE/_mlc_shadow/gdnmut_gc0.mlmodelc}
TRUST=${TRUST:-/private/var/db/AppleIntelligencePlatform/AppModelAssets/q35_08b_conv}
OUTDIR=${OUTDIR:-$HERE/_adapters}
MLCREAL=${MLCREAL:-/Library/Caches/com.apple.aned/q35_08b_conv/gdnmut_gc0.mlmodelc}

# ⛔ THE CHECKPOINT IDENTITY TRAP, AND IT WAS A THREE-WAY COLLISION.
# This default used to name `runs_codebook/v1/checkpoints/ckpt_step0000850.pt` (md5 96dc1b54...)
# in a directory that does not ship. The supplement DOES ship a file with that exact basename --
# run/checkpoints/ckpt_step0000850.pt, md5 d8bf95e0... -- so the obvious "fix the path" repair
# silently binds a DIFFERENT adapter from a DIFFERENT run: `run` best-step 350 at held-out CE
# 1.714, against `run_v1` step 1750 at 0.299. Nearly 6x worse, structurally valid, and every
# downstream gate still passes. The paper's figure is run_v1 step 1750 and nothing else.
CK=${CK:-$HERE/run_v1/checkpoints/ckpt_step0001750.pt}
CK_MD5_EXPECT=${CK_MD5_EXPECT:-972e196c88946a5b26ed3d9172d6ea43}

# ── PRECONDITIONS: name what is missing, do not fail three steps later on a symptom ──────────
fatal () { echo "⛔ $*" >&2; exit 2; }
[ -f "$CK" ] || fatal "checkpoint not found: $CK"
if [ "${CK_MD5_CHECK:-1}" = "1" ]; then
  GOT=$(md5 -q "$CK")
  [ "$GOT" = "$CK_MD5_EXPECT" ] || fatal "CHECKPOINT IDENTITY: $CK
     expected md5 $CK_MD5_EXPECT
     got      md5 $GOT
   This assert exists because three different files have carried a ckpt_step*.pt basename on this
   arc. If you are deliberately deploying another checkpoint, pass CK_MD5_EXPECT=<md5> (or
   CK_MD5_CHECK=0) so the substitution is recorded in the command rather than assumed."
fi
[ -f "$VENDOR/to_deploy_format.py"   ] || fatal "missing vendored converter: $VENDOR/to_deploy_format.py"
[ -f "$VENDOR/deploy_q35_adapter.py" ] || fatal "missing vendored converter: $VENDOR/deploy_q35_adapter.py"
[ -d "$SHADOW" ] || fatal "compiled base target not found: $SHADOW
   Build it first (see README 'Bring-up'), or point SHADOW= at an existing gdnmut_gc0.mlmodelc."
command -v "$PY" >/dev/null || fatal "python not found: $PY  (set PY=/path/to/python3)"
mkdir -p "$OUTDIR"

need_sudo () { sudo -n true 2>/dev/null || fatal "this stage needs passwordless sudo (run 'sudo -v' first).
   Staging to the AppleIntelligence trusted path and reading the live base require it."; }
need_sudo

W0=$(sudo -n md5 -q "$MLCREAL/weights/weight.bin" | cut -c1-16)
echo "  base weight.bin $W0  (must not move)"
# ⛔ PRESERVE WHAT IS LIVE. gc0_on.bin may hold another arc's adapter; overwriting it without a
# backup would destroy the only staged copy of a filed result.
if sudo -n test -f "$TRUST/gc0_on.bin"; then
  PRIOR=$(sudo -n md5 -q "$TRUST/gc0_on.bin" | cut -c1-16)
  echo "  currently live in gc0_on.bin: $PRIOR -> backing up"
  sudo -n cp "$TRUST/gc0_on.bin" "$TRUST/pre_codebook_on.bin.bak"
fi

echo "=== convert checkpoint to deploy format ==="
$PY "$VENDOR/to_deploy_format.py" --ckpt "$CK" --target "$SHADOW" --out /tmp/codebook_deploy.pt 2>&1 | tail -4
[ -f /tmp/codebook_deploy.pt ] || fatal "conversion produced no /tmp/codebook_deploy.pt"

echo "=== build the mutable blobs (FORM=conv, the only form that composes) ==="
MLC="$SHADOW" PT=/tmp/codebook_deploy.pt FORM=conv OUTDIR="$OUTDIR" \
  $PY "$VENDOR/deploy_q35_adapter.py" 2>&1 | tail -5
for arm in on off; do
  [ -f "$OUTDIR/gc0_${arm}.bin" ] || fatal "blob build produced no $OUTDIR/gc0_${arm}.bin"
done

echo "=== stage under a distinct name AND as gc0_*, md5-verified both ways ==="
for arm in on off; do
  sudo -n cp "$OUTDIR/gc0_${arm}.bin" "$TRUST/codebook_${arm}.bin"
  sudo -n cp "$OUTDIR/gc0_${arm}.bin" "$TRUST/gc0_${arm}.bin"
  A=$(md5 -q "$OUTDIR/gc0_${arm}.bin" | cut -c1-16)
  B=$(sudo -n md5 -q "$TRUST/gc0_${arm}.bin" | cut -c1-16)
  C=$(sudo -n md5 -q "$TRUST/codebook_${arm}.bin" | cut -c1-16)
  [ "$A" = "$B" ] && [ "$A" = "$C" ] && echo "  ${arm}: $B verified in both names" \
    || { echo "  ⛔ STAGE MISMATCH  built=$A gc0=$B codebook=$C"; exit 1; }
done
# ⛔ DISTINCTNESS. A shared build dir has staged a PREVIOUS adapter under a new name on this arc;
# the assert is what caught it. If the codebook blob equals another arm, the deploy did nothing.
NEW=$(sudo -n md5 -q "$TRUST/codebook_on.bin" | cut -c1-16)
for other in chess glyph2f glyph6x syriac legal; do
  O=$(sudo -n md5 -q "$TRUST/${other}_on.bin" 2>/dev/null | cut -c1-16)
  [ -n "$O" ] && [ "$NEW" = "$O" ] && { echo "  ⛔ codebook_on.bin IS IDENTICAL TO ${other}_on.bin -- deploy did not take"; exit 1; }
done
echo "  distinct from any other staged arm: PASS ($NEW)"

if [ "${VERIFY:-1}" = "0" ]; then
  echo "=== VERIFY=0 (fast stage for the live demo) — skipping the on-ANE 3-arm verify ==="
else
echo "=== ON-ANE three-arm verify · COMPOSE split (untrained combinations) ==="
EVAL_PROMPT="Decode the symbols." EVAL_HINT="" \
  SPLIT="$HERE/data_codebook/compose.jsonl" KIND=all N=${VERIFY_N:-60} \
  $PY "$HERE/eval_syriac_ane.py" 2>&1 | tail -16
fi

W1=$(sudo -n md5 -q "$MLCREAL/weights/weight.bin" | cut -c1-16)
[ "$W0" = "$W1" ] && echo "  base weight.bin CONSTANT ($W1)" || { echo "  ⛔ BASE MOVED $W0 -> $W1"; exit 1; }
