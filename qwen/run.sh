#!/bin/bash
# Swap two TRAINED LoRA adapters over a resident Qwen3-0.6B on the ANE, and watch the generated
# text change register while the base weight.bin stays byte-identical.
#
# Prereqs (research environment only, see the top-level README):
#   - SIP off + amfi_get_out_of_my_way=1
#   - export ANE_TRUSTED_ADAPTER_DIR to an AppleIntelligence AppModelAssets path-prefix (SIP-off writable)
#   - python3 train_lora.py && python3 deploy_lora.py   (produce _out/qwen_lora.mlmodelc + adapter_{A,B}.bin)
set -e
cd "$(dirname "$0")"
OUT="${QWEN_OUT:-$PWD/_out}"
PROMPT="${QWEN_PROMPT:-Once upon a time}"
NGEN="${QWEN_NGEN:-40}"
ANED_CACHE="${ANED_MODEL_CACHE:-/Library/Caches/com.apple.aned}"
: "${ANE_TRUSTED_ADAPTER_DIR:?export ANE_TRUSTED_ADAPTER_DIR to an ANE path-prefix-trusted dir (SIP-off writable)}"
PY="${PYTHON:-python3}"

echo "== 1. host assets (fp16 embedding + prompt ids) =="
$PY gen_io.py prep "$PROMPT"

echo "== 2. build + entitle gen_qwen (mandatory, or the mutable plan build -14s) =="
clang++ -O2 -fobjc-arc -framework Foundation -framework CoreML gen_qwen.mm -o gen_qwen
codesign -f -s - --entitlements ../harness/ane.entitlements gen_qwen

echo "== 3. stage the resident base + the two adapters =="
sudo rm -rf "$ANED_CACHE/qwen_lora.mlmodelc"; sudo cp -R "$OUT/qwen_lora.mlmodelc" "$ANED_CACHE/qwen_lora.mlmodelc"
sudo mkdir -p "$ANE_TRUSTED_ADAPTER_DIR/qwen"
sudo cp "$OUT/adapter_A.bin" "$ANE_TRUSTED_ADAPTER_DIR/qwen/A.bin"
sudo cp "$OUT/adapter_B.bin" "$ANE_TRUSTED_ADAPTER_DIR/qwen/B.bin"

KEY="@model_path/weights/adapter.bin"
echo "== 4. generate on the ANE (units=3), swapping only the mutable adapter =="
sudo ./gen_qwen "$ANED_CACHE/qwen_lora.mlmodelc" 3 "$KEY" "$ANE_TRUSTED_ADAPTER_DIR/qwen/A.bin" "$OUT/emb.f16" "$OUT/prompt.txt" "$NGEN" | grep TOKENS > "$OUT/out_A.txt"
sudo ./gen_qwen "$ANED_CACHE/qwen_lora.mlmodelc" 3 "$KEY" "$ANE_TRUSTED_ADAPTER_DIR/qwen/B.bin" "$OUT/emb.f16" "$OUT/prompt.txt" "$NGEN" | grep TOKENS > "$OUT/out_B.txt"

echo ""
echo "== adapter A =="; $PY gen_io.py decode "$OUT/out_A.txt"
echo ""
echo "== adapter B =="; $PY gen_io.py decode "$OUT/out_B.txt"

echo ""
echo "base weight.bin md5 (constant across swaps): $(sudo md5 -q "$ANED_CACHE/qwen_lora.mlmodelc/weights/weight.bin")"
sudo rm -rf "$ANE_TRUSTED_ADAPTER_DIR/qwen" "$ANED_CACHE/qwen_lora.mlmodelc"
