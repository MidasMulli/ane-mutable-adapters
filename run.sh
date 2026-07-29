#!/bin/bash
# End-to-end: build a small transformer with a LoRA head, split base(frozen)/adapter(mutable),
# then swap two adapters over the resident base and read the predicted token change.
#
# REQUIREMENTS (research environment only):
#   - Apple Silicon, macOS 27, SIP OFF + amfi_get_out_of_my_way=1. This is NOT a stock-Mac path.
#   - A directory the ANE trusts by PATH-PREFIX (an AppleIntelligence AppModelAssets path).
#     Export it as ANE_TRUSTED_ADAPTER_DIR. It is writable only under SIP-off + root.
#   - The ANE mutable-weight entitlements (see harness/ane.entitlements), self-signed.
#
# The literal trusted path is intentionally NOT hard-coded here. Locate it in your
# own research environment and export ANE_TRUSTED_ADAPTER_DIR before running.
set -euo pipefail
PY="${PY:-python3}"
ANED_CACHE="${ANED_MODEL_CACHE:-/Library/Caches/com.apple.aned}"
: "${ANE_TRUSTED_ADAPTER_DIR:?export ANE_TRUSTED_ADAPTER_DIR to an ANE path-prefix-trusted dir (SIP-off writable)}"

echo "== 1. build the transformer base (attention + MLP + LayerNorm + LM head) =="
$PY build/build_transformer.py                       # -> txf.mlpackage
xcrun coremlcompiler compile txf.mlpackage _out >/dev/null
cp -R _out/txf.mlmodelc txf.mlmodelc

echo "== 2. build two distinct LoRA-head adapters =="
$PY build/build_adapter.py 3 A                       # -> adT_A.mlpackage
$PY build/build_adapter.py 9 B                       # -> adT_B.mlpackage
for T in A B; do xcrun coremlcompiler compile adT_$T.mlpackage _a$T >/dev/null; cp -R _a$T/adT_$T.mlmodelc adT_$T.mlmodelc; done

echo "== 3. split: base head FROZEN in weight.bin, adapter MUTABLE in adapter.bin =="
cp adT_A.mlmodelc/weights/weight.bin txf.mlmodelc/weights/adapter.bin
$PY build/split_adapter.py txf.mlmodelc adapt_weight

echo "== 4. stage (SIP-off): compiled model in aned cache, adapters at the trusted path =="
sudo rm -rf "$ANED_CACHE/txf.mlmodelc"; sudo cp -R txf.mlmodelc "$ANED_CACHE/txf.mlmodelc"
sudo mkdir -p "$ANE_TRUSTED_ADAPTER_DIR/txf"
sudo cp adT_A.mlmodelc/weights/weight.bin "$ANE_TRUSTED_ADAPTER_DIR/txf/adapterA.bin"
sudo cp adT_B.mlmodelc/weights/weight.bin "$ANE_TRUSTED_ADAPTER_DIR/txf/adapterB.bin"

echo "== 5. build + sign the swap harness =="
bash harness/build_harness.sh   # builds + entitles BOTH swap and swap_multi (see script header re: -14)

echo "== 6. swap adapter A vs B over the frozen base -> predicted token =="
for T in A B A B; do
  echo -n "  adapter $T: "
  sudo ./swap "$ANED_CACHE/txf.mlmodelc" 3 "@model_path/weights/adapter.bin" "$ANE_TRUSTED_ADAPTER_DIR/txf/adapter$T.bin"
done
echo "base weight.bin md5 (constant across swaps): $(md5 -q txf.mlmodelc/weights/weight.bin)"
