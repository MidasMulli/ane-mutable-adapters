import os, math, time, urllib.request
import torch, torch.nn as nn, torch.nn.functional as F
DIR=os.environ.get("CHARGPT_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out"))
ck=torch.load(os.path.join(DIR,"chargpt.pt"), map_location="cpu"); cfg=ck["cfg"]
D=cfg["D"]; H=cfg["H"]; L=cfg["L"]; FF=cfg["FF"]; BLK=cfg["BLK"]; V=cfg["V"]; dh=D//H
stoi=ck["stoi"]; itos=ck["itos"]
dev="mps" if torch.backends.mps.is_available() else "cpu"

# ---- base arch (training form, masked_fill) ----
class Attn(nn.Module):
    def __init__(s):
        super().__init__(); s.q=nn.Conv2d(D,D,1,bias=False); s.k=nn.Conv2d(D,D,1,bias=False)
        s.v=nn.Conv2d(D,D,1,bias=False); s.o=nn.Conv2d(D,D,1,bias=False)
        s.register_buffer("mask", torch.triu(torch.ones(BLK,BLK),1).bool())
    def forward(s,x):
        B=x.shape[0]; S=x.shape[3]
        q=s.q(x).view(B,H,dh,S); k=s.k(x).view(B,H,dh,S); v=s.v(x).view(B,H,dh,S)
        att=(q.transpose(-1,-2)@k)/math.sqrt(dh); att=att.masked_fill(s.mask[:S,:S],float("-inf"))
        att=torch.softmax(att,-1); return s.o((v@att.transpose(-1,-2)).reshape(B,D,1,S))
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
class GPT(nn.Module):
    def __init__(s):
        super().__init__(); s.tok=nn.Embedding(V,D); s.pos=nn.Parameter(torch.zeros(1,D,1,BLK))
        s.blocks=nn.Sequential(*[Block() for _ in range(L)]); s.lnf=LN(); s.head=nn.Conv2d(D,V,1,bias=False)
    def h(s,idx):
        x=s.tok(idx).transpose(1,2).unsqueeze(2)+s.pos[:,:,:,:idx.shape[1]]
        return s.lnf(s.blocks(x))
base=GPT().to(dev); base.load_state_dict(ck["model"]); base.eval()
for p in base.parameters(): p.requires_grad=False

# ---- corpora (A=Shakespeare base's own; B=Alice, filtered to base vocab) ----
shake=open(os.path.join(DIR,"input.txt")).read()
alice_path=os.path.join(DIR,"alice.txt")
if not os.path.exists(alice_path):
    urllib.request.urlretrieve("https://www.gutenberg.org/files/11/11-0.txt", alice_path)
alice="".join(c for c in open(alice_path,encoding="utf-8").read() if c in stoi)
def enc(t): return torch.tensor([stoi[c] for c in t],dtype=torch.long)
def batch(d,bs=32):
    ix=torch.randint(len(d)-BLK-1,(bs,)); x=torch.stack([d[i:i+BLK] for i in ix]); y=torch.stack([d[i+1:i+1+BLK] for i in ix])
    return x.to(dev),y.to(dev)

def train_adapter(corpus, iters=500, tag=""):
    d=enc(corpus)
    adapt=nn.Conv2d(D,V,1,bias=False).to(dev); nn.init.zeros_(adapt.weight)
    opt=torch.optim.AdamW(adapt.parameters(), lr=1e-3)
    for it in range(iters+1):
        x,y=batch(d)
        with torch.no_grad(): h=base.h(x)
        logits=(base.head(h)+adapt(h)).squeeze(2).transpose(1,2)
        loss=F.cross_entropy(logits.reshape(-1,V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if it%250==0: print(f"  [{tag}] iter {it} loss {loss.item():.3f}", flush=True)
    return adapt.weight.detach().cpu().clone()

@torch.no_grad()
def sample(adaptw, seed="\n", n=280):
    adapt=nn.Conv2d(D,V,1,bias=False).to(dev); adapt.weight.copy_(adaptw.to(dev))
    ids=torch.tensor([[stoi[c] for c in seed]],dtype=torch.long,device=dev)
    for _ in range(n):
        h=base.h(ids[:,-BLK:]); lg=(base.head(h)+adapt(h)).squeeze(2).transpose(1,2)[:,-1,:]
        p=torch.softmax(lg/0.8,-1); ids=torch.cat([ids,torch.multinomial(p,1)],1)
    return "".join(itos[int(i)] for i in ids[0])

print(f"device={dev} shakespeare={len(shake)} alice(filtered)={len(alice)}", flush=True)
wA=train_adapter(shake, 500, "A/shakespeare")
wB=train_adapter(alice, 500, "B/alice")
torch.save({"A":wA,"B":wB}, os.path.join(DIR,"adapters.pt"))
print("\n===== ADAPTER A (Shakespeare) =====\n"+sample(wA), flush=True)
print("\n===== ADAPTER B (Alice) =====\n"+sample(wB), flush=True)
print(f"\nsaved adapters -> {DIR}/adapters.pt", flush=True)
