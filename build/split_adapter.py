#!/usr/bin/env python3
"""
Split a compiled CoreML model into a FROZEN base blob + a MUTABLE adapter blob.

Given a compiled .mlmodelc whose weights live in weights/weight.bin, this:
  1. repoints ONE named weight const (the adapter) to a separate weights/adapter.bin,
  2. injects a BlobFileMutabilityInfo declaration marking ONLY adapter.bin mutable.

The base (everything still in weight.bin) stays frozen; only the adapter is swappable.

Usage:
  python3 split_adapter.py <model.mlmodelc> <const_name>
    <const_name> e.g. "adapt_weight" (the parallel LoRA head) or "fin_weight".
Then drop a valid single-tensor adapter blob at <model.mlmodelc>/weights/adapter.bin
(build_adapter.py produces one; its tensor sits at offset 64).

For several mutable regions at once (a multi-layer LoRA), see split_adapter_multi.py.
"""
import os, re, sys

model_dir, const = sys.argv[1], sys.argv[2]
mil = f"{model_dir}/model.mil"
s = open(mil).read()
cname = f"{const}_to_fp16"

# Match the target const's full single-line statement, up to its BLOBFILE offset close.
# ANCHORED: the character before the name must not be an identifier character. Without
# this, a short name matches inside a longer one (e.g. "adapt_weight_to_fp16" matches
# inside "blocks_0_oadapt_weight_to_fp16") and the WRONG const is silently split,
# producing a model that fails to load with a confusing error code.
stmt_re = re.compile(r'(?:^|[^A-Za-z0-9_])(' + re.escape(cname) +
                     r' = const\(\)\[.*?offset = uint64\(\d+\)\)\)\])', re.M)
m = stmt_re.search(s)
if not m:
    sys.exit(f"error: const '{cname}' with a weight.bin BLOBFILE not found in {mil}")
stmt = m.group(1)

# Guard the shape contract: the adapter blob must hold exactly this const's tensor.
# A silent size mismatch is the other way this fails confusingly at load time.
shape_m = re.search(r'tensor<fp16, \[([0-9, ]+)\]>', stmt)
blob = os.path.join(model_dir, "weights", "adapter.bin")
if shape_m and os.path.exists(blob):
    dims = [int(d) for d in shape_m.group(1).split(",")]
    want = 1
    for d in dims: want *= d
    want *= 2                                   # fp16
    have = os.path.getsize(blob) - 64           # tensor sits at offset 64
    if have < want:
        sys.exit(f"error: adapter.bin holds {have} bytes after the 64-byte offset but "
                 f"'{cname}' needs {want} (shape {dims}, fp16). Rebuild the adapter at "
                 f"matching dimensions (see build_adapter.py).")

new_stmt = re.sub(
    r'BLOBFILE\(path = string\("@model_path/weights/weight\.bin"\), offset = uint64\(\d+\)\)',
    'BLOBFILE(path = string("@model_path/weights/adapter.bin"), offset = uint64(64))',
    stmt)
if new_stmt == stmt:
    sys.exit(f"error: '{cname}' does not reference weights/weight.bin (already split?)")
s = s.replace(stmt, new_stmt)

# inject the mutability declaration after the coremltools-version program attribute
marker = '{"coremltools-version", "9.0"}})]'
bfmi = marker[:-1] + (', BlobFileMutabilityInfo = tuple<string, dict<string, string>>'
                      '(("Paths", {{"@model_path/weights/adapter.bin", '
                      '"@model_path/weights/adapter.bin"}}))]')
if s.count(marker) != 1:
    sys.exit("error: coremltools-version program marker not found (or not unique); "
             "adjust the marker string to match your compiled model.")
s = s.replace(marker, bfmi)

open(mil, "w").write(s)
print(f"split: {cname} -> adapter.bin@64 ; BFMI declares adapter.bin mutable ; base frozen in weight.bin")
