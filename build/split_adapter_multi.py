#!/usr/bin/env python3
"""
Split a compiled CoreML model into a FROZEN base blob + N MUTABLE adapter blobs.

The multi-region form of split_adapter.py. Each named const is repointed to its own
weights/adapterN.bin, and ALL of them are declared mutable in one
BlobFileMutabilityInfo Paths dict. This is the shape a multi-layer LoRA needs (Apple's
own on-device model carries many mutable kernel sections, not one).

Usage:
  python3 split_adapter_multi.py <model.mlmodelc> <const1> [<const2> ...]
    -> const_i is repointed to weights/adapter{i}.bin, tensor at offset 64.

Then drop a matching single-tensor blob at each weights/adapter{i}.bin and pass every
key/URL pair to the swap harness (see harness/swap_multi.mm).

MIL dict syntax note: the Paths value is a dict literal whose OUTER braces wrap
comma-separated {"key", "value"} pairs, i.e. one entry is {{"a", "a"}} and two entries
are {{"a", "a"}, {"b", "b"}}. Double-brace each pair AND wrapping again produces
{{{...}}}, which is malformed and segfaults the loader rather than erroring.
"""
import os, re, sys

model_dir = sys.argv[1]
consts = sys.argv[2:]
if not consts:
    sys.exit("usage: split_adapter_multi.py <model.mlmodelc> <const1> [<const2> ...]")
mil = f"{model_dir}/model.mil"
s = open(mil).read()

paths = []
for i, const in enumerate(consts):
    cname = f"{const}_to_fp16"
    blobname = f"adapter{i}.bin"
    # ANCHORED, for the same reason as split_adapter.py: an unanchored short name
    # matches inside a longer const and silently splits the wrong tensor.
    pat = re.compile(r'(?:^|[^A-Za-z0-9_])(' + re.escape(cname) +
                     r' = const\(\)\[.*?offset = uint64\(\d+\)\)\)\])', re.M)
    m = pat.search(s)
    if not m:
        sys.exit(f"error: const '{cname}' with a weight.bin BLOBFILE not found in {mil}")
    stmt = m.group(1)

    shape_m = re.search(r'tensor<fp16, \[([0-9, ]+)\]>', stmt)
    blob = os.path.join(model_dir, "weights", blobname)
    if shape_m and os.path.exists(blob):
        dims = [int(d) for d in shape_m.group(1).split(",")]
        want = 1
        for d in dims: want *= d
        want *= 2
        have = os.path.getsize(blob) - 64
        if have < want:
            sys.exit(f"error: {blobname} holds {have} bytes after the 64-byte offset but "
                     f"'{cname}' needs {want} (shape {dims}, fp16).")

    new_stmt = re.sub(
        r'BLOBFILE\(path = string\("@model_path/weights/weight\.bin"\), offset = uint64\(\d+\)\)',
        f'BLOBFILE(path = string("@model_path/weights/{blobname}"), offset = uint64(64))',
        stmt)
    if new_stmt == stmt:
        sys.exit(f"error: '{cname}' does not reference weights/weight.bin (already split?)")
    s = s.replace(stmt, new_stmt)
    paths.append(f'@model_path/weights/{blobname}')

marker = '{"coremltools-version", "9.0"}})]'
if s.count(marker) != 1:
    sys.exit("error: coremltools-version program marker not found (or not unique); "
             "adjust the marker string to match your compiled model.")
inner = ", ".join('{"%s", "%s"}' % (p, p) for p in paths)
bfmi = marker[:-1] + (', BlobFileMutabilityInfo = tuple<string, dict<string, string>>'
                      '(("Paths", {%s}))]' % inner)
s = s.replace(marker, bfmi)

open(mil, "w").write(s)
print(f"split {len(consts)} clusters -> " +
      ", ".join(f"adapter{i}.bin" for i in range(len(consts))))
print(f"BFMI declares {len(paths)} mutable paths ; base frozen in weight.bin")
