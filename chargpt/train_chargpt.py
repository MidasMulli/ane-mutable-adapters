import os, math, time, sys, urllib.request
import torch, torch.nn as nn, torch.nn.functional as F

# ---- config (ANE-native txf arch + char-GPT pieces) ----
D=256; H=4; L=6; FF=4*D; BLK=128; dh=D//H
ITERS=int(sys.argv[1]) if len(sys.argv)>1 else 3000
BATCH=48; LR=3e-4
OUT=os.environ.get("CHARGPT_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out"))
os.makedirs(OUT, exist_ok=True)
dev = "mps" if torch.backends.mps.is_available() else "cpu"

# ---- corpus (tinyshakespeare) ----
CORP=os.path.join(OUT,"input.txt")
if not os.path.exists(CORP):
    url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, CORP)
text=open(CORP).read()
chars=sorted(set(text)); V=len(chars)
stoi={c:i for i,c in enumerate(chars)}; itos={i:c for i,c in enumerate(chars)}
data=torch.tensor([stoi[c] for c in text], dtype=torch.long)
n=int(0.9*len(data)); train_d, val_d = data[:n], data[n:]
def batch(split):
    d = train_d if split=="train" else val_d
    ix=torch.randint(len(d)-BLK-1,(BATCH,))
    x=torch.stack([d[i:i+BLK] for i in ix]); y=torch.stack([d[i+1:i+1+BLK] for i in ix])
    return x.to(dev), y.to(dev)

# ---- ANE-native GPT (matches txf: conv2d-as-linear, (B,C,1,S), attention-as-matmul, mean-of-squares LN) ----
class Attn(nn.Module):
    def __init__(s):
        super().__init__(); s.q=nn.Conv2d(D,D,1,bias=False); s.k=nn.Conv2d(D,D,1,bias=False)
        s.v=nn.Conv2d(D,D,1,bias=False); s.o=nn.Conv2d(D,D,1,bias=False)
        s.register_buffer("mask", torch.triu(torch.ones(BLK,BLK),1).bool())  # t>s -> masked (causal)
    def forward(s,x):  # x: (B,D,1,S)
        B=x.shape[0]; S=x.shape[3]
        q=s.q(x).view(B,H,dh,S); k=s.k(x).view(B,H,dh,S); v=s.v(x).view(B,H,dh,S)
        att=(q.transpose(-1,-2)@k)/math.sqrt(dh)                 # (B,H,S,S) = (query s, key t)
        att=att.masked_fill(s.mask[:S,:S], float("-inf"))
        att=torch.softmax(att,dim=-1)
        out=v@att.transpose(-1,-2)                               # (B,H,dh,S)
        return s.o(out.reshape(B,D,1,S))
class MLP(nn.Module):
    def __init__(s): super().__init__(); s.up=nn.Conv2d(D,FF,1,bias=False); s.down=nn.Conv2d(FF,D,1,bias=False)
    def forward(s,x): return s.down(F.gelu(s.up(x)))
class LN(nn.Module):
    def __init__(s): super().__init__(); s.g=nn.Parameter(torch.ones(1,D,1,1)); s.b=nn.Parameter(torch.zeros(1,D,1,1))
    def forward(s,x):
        u=x.mean(1,keepdim=True); d=x-u; var=(d*d).mean(1,keepdim=True)
        return d/torch.sqrt(var+1e-5)*s.g+s.b
class Block(nn.Module):
    def __init__(s): super().__init__(); s.ln1=LN(); s.att=Attn(); s.ln2=LN(); s.mlp=MLP()
    def forward(s,x): x=x+s.att(s.ln1(x)); x=x+s.mlp(s.ln2(x)); return x
class GPT(nn.Module):
    def __init__(s):
        super().__init__()
        s.tok=nn.Embedding(V,D); s.pos=nn.Parameter(torch.zeros(1,D,1,BLK))
        s.blocks=nn.Sequential(*[Block() for _ in range(L)])
        s.lnf=LN(); s.head=nn.Conv2d(D,V,1,bias=False)
    def forward(s,idx):                 # idx: (B,S) token ids
        B,S=idx.shape
        x=s.tok(idx).transpose(1,2).unsqueeze(2)      # (B,D,1,S)
        x=x+s.pos[:,:,:,:S]
        x=s.lnf(s.blocks(x))
        return s.head(x).squeeze(2).transpose(1,2)    # (B,S,V) logits

m=GPT().to(dev)
opt=torch.optim.AdamW(m.parameters(), lr=LR)
print(f"device={dev} vocab={V} params={sum(p.numel() for p in m.parameters())/1e6:.2f}M iters={ITERS}", flush=True)

@torch.no_grad()
def sample(seed="\n", n_new=300):
    m.eval(); ids=torch.tensor([[stoi[c] for c in seed]],dtype=torch.long,device=dev)
    for _ in range(n_new):
        logits=m(ids[:,-BLK:])[:,-1,:]
        p=torch.softmax(logits/0.8,dim=-1)
        nx=torch.multinomial(p,1)
        ids=torch.cat([ids,nx],1)
    m.train(); return "".join(itos[int(i)] for i in ids[0])

t0=time.time()
for it in range(ITERS+1):
    x,y=batch("train")
    logits=m(x)
    loss=F.cross_entropy(logits.reshape(-1,V), y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if it%250==0:
        with torch.no_grad():
            vx,vy=batch("val"); vl=F.cross_entropy(m(vx).reshape(-1,V), vy.reshape(-1))
        print(f"  iter {it:5d}  train {loss.item():.3f}  val {vl.item():.3f}  ({time.time()-t0:.0f}s)", flush=True)

torch.save({"model":m.state_dict(),"stoi":stoi,"itos":itos,"cfg":dict(D=D,H=H,L=L,FF=FF,BLK=BLK,V=V)}, os.path.join(OUT,"chargpt.pt"))
print("\n===== SAMPLE (seed newline) =====", flush=True)
print(sample("\n", 400), flush=True)
print(f"\nsaved -> {OUT}/chargpt.pt", flush=True)
