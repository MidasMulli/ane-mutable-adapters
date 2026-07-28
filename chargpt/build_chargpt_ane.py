import os, math, shutil
import torch, torch.nn as nn, torch.nn.functional as F, coremltools as ct
CK=os.path.expanduser("_out/chargpt.pt")
ck=torch.load(CK, map_location="cpu"); cfg=ck["cfg"]
D=cfg["D"]; H=cfg["H"]; L=cfg["L"]; FF=cfg["FF"]; BLK=cfg["BLK"]; V=cfg["V"]; dh=D//H
OUT=os.environ.get("CHARGPT_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out")); S=BLK

class Attn(nn.Module):
    def __init__(s):
        super().__init__(); s.q=nn.Conv2d(D,D,1,bias=False); s.k=nn.Conv2d(D,D,1,bias=False)
        s.v=nn.Conv2d(D,D,1,bias=False); s.o=nn.Conv2d(D,D,1,bias=False)
        s.register_buffer("mbias", (torch.triu(torch.ones(BLK,BLK),1)*-1e4))  # additive causal (fp16-safe)
    def forward(s,x):
        q=s.q(x).view(1,H,dh,BLK); k=s.k(x).view(1,H,dh,BLK); v=s.v(x).view(1,H,dh,BLK)
        att=(q.transpose(-1,-2)@k)/math.sqrt(dh) + s.mbias
        att=torch.softmax(att,dim=-1)
        out=v@att.transpose(-1,-2)
        return s.o(out.reshape(1,D,1,BLK))
class MLP(nn.Module):
    def __init__(s): super().__init__(); s.up=nn.Conv2d(D,FF,1,bias=False); s.down=nn.Conv2d(FF,D,1,bias=False)
    def forward(s,x): return s.down(F.gelu(s.up(x)))
class LN(nn.Module):
    def __init__(s): super().__init__(); s.g=nn.Parameter(torch.ones(1,D,1,1)); s.b=nn.Parameter(torch.zeros(1,D,1,1))
    def forward(s,x):
        u=x.mean(1,keepdim=True); d=x-u; var=(d*d).mean(1,keepdim=True); return d/torch.sqrt(var+1e-5)*s.g+s.b
class Block(nn.Module):
    def __init__(s): super().__init__(); s.ln1=LN(); s.att=Attn(); s.ln2=LN(); s.mlp=MLP()
    def forward(s,x): x=x+s.att(s.ln1(x)); x=x+s.mlp(s.ln2(x)); return x
class Body(nn.Module):
    def __init__(s):
        super().__init__(); s.tok=nn.Embedding(V,D); s.pos=nn.Parameter(torch.zeros(1,D,1,BLK))
        s.blocks=nn.Sequential(*[Block() for _ in range(L)]); s.lnf=LN(); s.head=nn.Conv2d(D,V,1,bias=False)
    def forward(s,x0):                       # x0: (1,D,1,BLK) embedded (host does tok lookup)
        x=x0+s.pos
        return s.head(s.lnf(s.blocks(x)))    # (1,V,1,BLK) logits

m=Body().eval()
miss,unexp=m.load_state_dict(ck["model"], strict=False)
print(f"loaded trained weights; skipped buffers (mask->mbias): missing={len(miss)} unexpected={len(unexp)}")
ex=torch.rand(1,D,1,S)
ts=torch.jit.trace(m, ex)
mlm=ct.convert(ts, inputs=[ct.TensorType(name="x",shape=(1,D,1,S))], convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15)
p=os.path.join(OUT,"chargpt_body.mlpackage"); shutil.rmtree(p,ignore_errors=True); mlm.save(p)
print("saved", p)
