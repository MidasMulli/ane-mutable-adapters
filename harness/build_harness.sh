#!/bin/bash
# Build + entitle the ANE mutable-weight swap harnesses.
#
# CRITICAL: without the entitlements in harness/ane.entitlements, the e5rt
# mutable-weight ANE plan build fails at load with error -14
# ("Failed to build the model execution plan using ... model.mil").
# The entitlements are honored only under SIP off + amfi_get_out_of_my_way=1.
#
# -14 is NOT specific to entitlements. It is a generic plan-build failure code;
# several distinct causes produce this same code and message, including a
# malformed per-weight MILBlob metadata header in the adapter blob. A -14 on an
# already-entitled binary is undiagnosed, not explained. See README.md.
set -e
cd "$(dirname "$0")/.."
ENT=harness/ane.entitlements
for name in swap swap_multi; do
  clang++ -fobjc-arc -framework Foundation -framework CoreML "harness/$name.mm" -o "$name"
  codesign -f -s - --entitlements "$ENT" "$name"
  n=$(codesign -d --entitlements - "$name" 2>&1 | grep -c "aned.private")
  echo "built + entitled: $name  ($n aned.private entitlements + iokit-user-access)"
done
