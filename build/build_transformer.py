import torch, torch.nn as nn, torch.nn.functional as F, coremltools as ct, shutil, math
D=256; H=4; L=4; FF=1024; S=32; V=1000; dh=D//H
torch.manual_seed(0)
class Attn(nn.Module):
    def __init__(s):
        super().__init__(); s.q=nn.Conv2d(D,D,1,bias=False); s.k=nn.Conv2d(D,D,1,bias=False)
        s.v=nn.Conv2d(D,D,1,bias=False); s.o=nn.Conv2d(D,D,1,bias=False)
    def forward(s,x):
        q=s.q(x).view(1,H,dh,S); k=s.k(x).view(1,H,dh,S); v=s.v(x).view(1,H,dh,S)
        att=torch.einsum('bhcs,bhct->bhst',q,k)/math.sqrt(dh)
        att=torch.softmax(att,dim=-1)
        out=torch.einsum('bhst,bhct->bhcs',att,v).reshape(1,D,1,S)
        return s.o(out)
class MLP(nn.Module):
    def __init__(s): super().__init__(); s.up=nn.Conv2d(D,FF,1,bias=False); s.down=nn.Conv2d(FF,D,1,bias=False)
    def forward(s,x): return s.down(F.gelu(s.up(x)))
class LN(nn.Module):
    def __init__(s): super().__init__(); s.g=nn.Parameter(torch.ones(1,D,1,1)); s.b=nn.Parameter(torch.zeros(1,D,1,1))
    def forward(s,x):
        u=x.mean(1,keepdim=True); var=x.var(1,keepdim=True,unbiased=False)
        return (x-u)/torch.sqrt(var+1e-5)*s.g+s.b
class Block(nn.Module):
    def __init__(s): super().__init__(); s.ln1=LN(); s.att=Attn(); s.ln2=LN(); s.mlp=MLP()
    def forward(s,x): x=x+s.att(s.ln1(x)); x=x+s.mlp(s.ln2(x)); return x
class T(nn.Module):
    def __init__(s):
        super().__init__(); s.blocks=nn.Sequential(*[Block() for _ in range(L)])
        s.lnf=LN(); s.head=nn.Conv2d(D,V,1,bias=False); s.adapt=nn.Conv2d(D,V,1,bias=False)
    def forward(s,x):
        h=s.lnf(s.blocks(x)); return s.head(h)+s.adapt(h)   # logits = base_head(h) + adapter(h)
m=T().eval()
ts=torch.jit.trace(m, torch.rand(1,D,1,S))
mlm=ct.convert(ts, inputs=[ct.TensorType(name="x",shape=(1,D,1,S))],convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15)
shutil.rmtree("txf.mlpackage",ignore_errors=True); mlm.save("txf.mlpackage"); print(f"transformer L={L} D={D} H={H} V={V} saved")
