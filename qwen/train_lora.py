"""LoRA-finetune Qwen3-0.6B (base FROZEN) on two corpora, saving one adapter each.

Trains in the (T,D) linear form on MPS/CPU: mathematically identical to the deploy conv form
but avoids an MPS conv-kernel edge case on the 151936-wide head. rank-r LoRA on q+v of all 28
layers; the trained [out,in] weights transfer 1:1 to the parallel-delta deploy form.

  python3 train_lora.py            # trains adapter A (Shakespeare) and B (Alice) into _out/
"""
import os, math, glob, urllib.request, warnings, torch, torch.nn as nn, torch.nn.functional as F
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer
from qwen_ane import _weights, MODEL_ID, D, NL, NH, NKV, HD, IM, EPS, THETA, V

OUT = os.environ.get("QWEN_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out"))
os.makedirs(OUT, exist_ok=True)
R = int(os.environ.get("QWEN_LORA_RANK", 16)); ALPHA = 2 * R; SC = ALPHA / R; T = 64; STEPS = int(os.environ.get("QWEN_STEPS", 400))
dev = "mps" if torch.backends.mps.is_available() else "cpu"

def corpus(name, url):
    p = os.path.join(OUT, name)
    if not os.path.exists(p): urllib.request.urlretrieve(url, p)
    return open(p, encoding="utf-8").read()

tok = AutoTokenizer.from_pretrained(MODEL_ID)
W = _weights(); g = lambda k: W[k].float().to(dev)
def rms(x, w): return x / torch.sqrt((x * x).mean(-1, keepdim=True) + EPS) * w
inv = (1.0 / (THETA ** (torch.arange(0, HD, 2).float() / HD))).to(dev)
def roth(x): x1, x2 = x[..., :HD // 2], x[..., HD // 2:]; return torch.cat([-x2, x1], -1)

Bw = [{} for _ in range(NL)]
for L in range(NL):
    p = f"model.layers.{L}."
    for nm in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
        Bw[L][nm] = g(p + nm + ".weight")
    Bw[L]["iln"] = g(p + "input_layernorm.weight"); Bw[L]["pln"] = g(p + "post_attention_layernorm.weight")
    Bw[L]["qn"] = g(p + "self_attn.q_norm.weight"); Bw[L]["kn"] = g(p + "self_attn.k_norm.weight")
NORMF = g("model.norm.weight"); HEAD = g("lm_head.weight"); EMB = g("model.embed_tokens.weight")
cos, sin = (lambda e: (e.cos(), e.sin()))(torch.cat([torch.outer(torch.arange(T, device=dev).float(), inv)] * 2, -1))
mask = torch.triu(torch.full((T, T), float("-inf"), device=dev), 1)

def new_lora():
    d = {}
    for L in range(NL):
        d[f"{L}.qA"] = nn.Parameter(torch.randn(R, D, device=dev) / R); d[f"{L}.qB"] = nn.Parameter(torch.zeros(NH * HD, R, device=dev))
        d[f"{L}.vA"] = nn.Parameter(torch.randn(R, D, device=dev) / R); d[f"{L}.vB"] = nn.Parameter(torch.zeros(NKV * HD, R, device=dev))
    return d

def fwd(seqemb, lora):
    x = seqemb
    for L in range(NL):
        b = Bw[L]; r = x; h = rms(x, b["iln"])
        q = h @ b["self_attn.q_proj"].T + SC * (h @ lora[f"{L}.qA"].T) @ lora[f"{L}.qB"].T
        k = h @ b["self_attn.k_proj"].T
        v = h @ b["self_attn.v_proj"].T + SC * (h @ lora[f"{L}.vA"].T) @ lora[f"{L}.vB"].T
        q = q.view(T, NH, HD); k = k.view(T, NKV, HD); v = v.view(T, NKV, HD)
        q = rms(q, b["qn"]); k = rms(k, b["kn"]); q = q * cos[:, None] + roth(q) * sin[:, None]; k = k * cos[:, None] + roth(k) * sin[:, None]
        k = k.repeat_interleave(NH // NKV, 1); v = v.repeat_interleave(NH // NKV, 1)
        q = q.transpose(0, 1); k = k.transpose(0, 1); v = v.transpose(0, 1)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(HD) + mask; att = att.softmax(-1)
        o = (att @ v).transpose(0, 1).reshape(T, NH * HD) @ b["self_attn.o_proj"].T
        x = r + o; r = x; h = rms(x, b["pln"]); x = r + (F.silu(h @ b["mlp.gate_proj"].T) * (h @ b["mlp.up_proj"].T)) @ b["mlp.down_proj"].T
    return rms(x, NORMF) @ HEAD.T

def train_on(text, tag):
    ids = torch.tensor(tok(text).input_ids, dtype=torch.long)
    print(f"  [{tag}] {len(ids)} tokens; dev={dev}", flush=True)
    lora = new_lora(); opt = torch.optim.Adam(list(lora.values()), lr=1e-3)
    for step in range(STEPS):
        i = int(torch.randint(0, len(ids) - T - 1, (1,)).item()); seq = ids[i:i + T + 1].to(dev)
        loss = F.cross_entropy(fwd(EMB[seq[:T]], lora), seq[1:T + 1])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0 or step == STEPS - 1: print(f"    [{tag}] step {step:4d} loss {loss.item():.3f}", flush=True)
    return {k: v.detach().cpu().clone() for k, v in lora.items()}

if __name__ == "__main__":
    shake = corpus("input.txt", "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
    alice = corpus("alice.txt", "https://www.gutenberg.org/files/11/11-0.txt")
    adA = train_on(shake, "A/shakespeare"); adB = train_on(alice, "B/alice")
    torch.save({"A": adA, "B": adB, "R": R, "ALPHA": ALPHA}, os.path.join(OUT, "lora_adapters.pt"))
    print("  saved", os.path.join(OUT, "lora_adapters.pt"), flush=True)
