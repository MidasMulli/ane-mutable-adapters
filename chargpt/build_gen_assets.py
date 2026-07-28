import os, math, shutil, struct, re
import torch, torch.nn as nn, torch.nn.functional as F, coremltools as ct
DIR=os.environ.get("CHARGPT_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out"))
GEN=os.path.join(os.environ.get("CHARGPT_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out")), "gen"); os.makedirs(GEN, exist_ok=True)
ck=torch.load(os.path.join(DIR,"chargpt.pt"),map_location="cpu"); cfg=ck["cfg"]
ad=torch.load(os.path.join(DIR,"adapters.pt"),map_location="cpu")
D=cfg["D"]; H=cfg["H"]; L=cfg["L"]; FF=cfg["FF"]; BLK=cfg["BLK"]; V=cfg["V"]; dh=D//H
stoi=ck["stoi"]; itos=ck["itos"]

# ---- emb matrix (V x D fp32) + vocab (chars in id order) ----
tokw=ck["model"]["tok.weight"].float().contiguous()   # (V,D)
with open(os.path.join(GEN,"emb.bin"),"wb") as f: f.write(tokw.numpy().tobytes())
with open(os.path.join(GEN,"vocab.txt"),"wb") as f: f.write(bytes(ord(itos[i]) for i in range(V)))
print(f"emb {tuple(tokw.shape)} + vocab({V}) written")

# ---- base+adapter body (embedded input -> head(h)+adapt(h)), fixed shapes ----
class Attn(nn.Module):
    def __init__(s):
        super().__init__(); s.q=nn.Conv2d(D,D,1,bias=False); s.k=nn.Conv2d(D,D,1,bias=False)
        s.v=nn.Conv2d(D,D,1,bias=False); s.o=nn.Conv2d(D,D,1,bias=False)
        s.register_buffer("mbias",(torch.triu(torch.ones(BLK,BLK),1)*-1e4))
    def forward(s,x):
        q=s.q(x).view(1,H,dh,BLK); k=s.k(x).view(1,H,dh,BLK); v=s.v(x).view(1,H,dh,BLK)
        att=torch.softmax((q.transpose(-1,-2)@k)/math.sqrt(dh)+s.mbias,-1)
        return s.o((v@att.transpose(-1,-2)).reshape(1,D,1,BLK))
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
        super().__init__(); s.pos=nn.Parameter(torch.zeros(1,D,1,BLK))
        s.blocks=nn.Sequential(*[Block() for _ in range(L)]); s.lnf=LN()
        s.head=nn.Conv2d(D,V,1,bias=False); s.adapt=nn.Conv2d(D,V,1,bias=False)
    def forward(s,x0):
        h=s.lnf(s.blocks(x0+s.pos)); return s.head(h)+s.adapt(h)
m=Body().eval()
sd={k:v for k,v in ck["model"].items() if not k.startswith("tok")}
nn.init.zeros_(m.adapt.weight)
miss,unexp=m.load_state_dict(sd, strict=False)  # loads pos/blocks/lnf/head; adapt stays zero
ts=torch.jit.trace(m, torch.rand(1,D,1,BLK))
mlm=ct.convert(ts, inputs=[ct.TensorType(name="x",shape=(1,D,1,BLK))],convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15)
pk=os.path.join(GEN,"cgpt.mlpackage"); shutil.rmtree(pk,ignore_errors=True); mlm.save(pk)
print("base+adapter body saved")

# ---- adapter-only blobs (256->65 conv) carrying A / B weights ----
def adapter_blob(w, tag):
    a=nn.Conv2d(D,V,1,bias=False).eval()
    with torch.no_grad(): a.weight.copy_(w)
    tsm=torch.jit.trace(a, torch.rand(1,D,1,BLK))
    mm=ct.convert(tsm, inputs=[ct.TensorType(name="x",shape=(1,D,1,BLK))],convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15)
    p=os.path.join(GEN,f"ad_{tag}.mlpackage"); shutil.rmtree(p,ignore_errors=True); mm.save(p)
    o=os.path.join(GEN,f"ad_{tag}_c"); shutil.rmtree(o,ignore_errors=True)
    os.system(f"xcrun coremlcompiler compile '{p}' '{o}' >/dev/null 2>&1")
    return os.path.join(o,f"ad_{tag}.mlmodelc","weights","weight.bin")
blobA=adapter_blob(ad["A"],"A"); blobB=adapter_blob(ad["B"],"B")
print("adapter A/B blobs built")
print("ASSET_DIR", GEN)
