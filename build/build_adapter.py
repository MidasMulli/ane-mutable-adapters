import torch, torch.nn as nn, coremltools as ct, shutil, sys
seed=int(sys.argv[1]); tag=sys.argv[2]; torch.manual_seed(seed)
class A(nn.Module):
    def __init__(s): super().__init__(); s.adapt=nn.Conv2d(256,1000,1,bias=False)
    def forward(s,x): return s.adapt(x)
m=A().eval()
with torch.no_grad(): m.adapt.weight.copy_(torch.randn(1000,256,1,1)*0.4)
ts=torch.jit.trace(m, torch.rand(1,256,1,32))
mlm=ct.convert(ts, inputs=[ct.TensorType(name="x",shape=(1,256,1,32))],convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15)
shutil.rmtree(f"adT_{tag}.mlpackage",ignore_errors=True); mlm.save(f"adT_{tag}.mlpackage"); print(f"adT_{tag} saved")
