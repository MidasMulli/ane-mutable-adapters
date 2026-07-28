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
"""
import re, sys

model_dir, const = sys.argv[1], sys.argv[2]
mil = f"{model_dir}/model.mil"
s = open(mil).read()
cname = f"{const}_to_fp16"

# match the target const's full single-line statement, up to its BLOBFILE offset close
stmt_re = re.compile(re.escape(cname) + r' = const\(\)\[.*?offset = uint64\(\d+\)\)\)\]')
m = stmt_re.search(s)
if not m:
    sys.exit(f"error: const '{cname}' with a weight.bin BLOBFILE not found in {mil}")
stmt = m.group(0)
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
