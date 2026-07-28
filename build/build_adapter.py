#!/usr/bin/env python3
"""
Build a single-tensor adapter whose compiled weight.bin becomes adapter.bin.

Usage:
  python3 build_adapter.py <seed> <tag> [D] [V] [S]

D/V/S default to build_transformer.py's values (256 / 1000 / 32). If you resize the
base model you MUST pass matching dimensions here, or the blob is the wrong size for
the const it replaces. split_adapter.py checks this and fails loudly, but passing the
dimensions is how you avoid the problem in the first place.
"""
import torch, torch.nn as nn, coremltools as ct, shutil, sys

seed=int(sys.argv[1]); tag=sys.argv[2]
D = int(sys.argv[3]) if len(sys.argv) > 3 else 256
V = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
S = int(sys.argv[5]) if len(sys.argv) > 5 else 32
torch.manual_seed(seed)

class A(nn.Module):
    def __init__(s): super().__init__(); s.adapt=nn.Conv2d(D,V,1,bias=False)
    def forward(s,x): return s.adapt(x)

m=A().eval()
with torch.no_grad(): m.adapt.weight.copy_(torch.randn(V,D,1,1)*0.4)
ts=torch.jit.trace(m, torch.rand(1,D,1,S))
mlm=ct.convert(ts, inputs=[ct.TensorType(name="x",shape=(1,D,1,S))],convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15)
shutil.rmtree(f"adT_{tag}.mlpackage",ignore_errors=True); mlm.save(f"adT_{tag}.mlpackage")
print(f"adT_{tag} saved  (D={D} V={V} S={S} -> {V*D*2/1e6:.1f} MB fp16)")
