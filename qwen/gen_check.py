"""Sanity gate (no ANE, no SIP-off): confirm the trained adapters shift generation, in plain PyTorch,
before deploying to the ANE. Greedy autoregressive generation with base / adapter A / adapter B.

  python3 gen_check.py "Once upon a time"
"""
import os, sys, math, warnings, torch, torch.nn.functional as F
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer
from qwen_ane import _weights, MODEL_ID, D, NL, NH, NKV, HD, EPS, THETA

OUT = os.environ.get("QWEN_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out"))
dev = "mps" if torch.backends.mps.is_available() else "cpu"; T = 64
tok = AutoTokenizer.from_pretrained(MODEL_ID)
W = _weights(); g = lambda k: W[k].float().to(dev)
def rms(x, w): return x / torch.sqrt((x * x).mean(-1, keepdim=True) + EPS) * w
inv = (1.0 / (THETA ** (torch.arange(0, HD, 2).float() / HD))).to(dev)
def roth(x): x1, x2 = x[..., :HD // 2], x[..., HD // 2:]; return torch.cat([-x2, x1], -1)
Bw = [{} for _ in range(NL)]
for L in range(NL):
    p = f"model.layers.{L}."
    for nm in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]: Bw[L][nm] = g(p + nm + ".weight")
    Bw[L]["iln"] = g(p + "input_layernorm.weight"); Bw[L]["pln"] = g(p + "post_attention_layernorm.weight"); Bw[L]["qn"] = g(p + "self_attn.q_norm.weight"); Bw[L]["kn"] = g(p + "self_attn.k_norm.weight")
NORMF = g("model.norm.weight"); HEAD = g("lm_head.weight"); EMB = g("model.embed_tokens.weight")
ad = torch.load(os.path.join(OUT, "lora_adapters.pt"), map_location=dev); SC = ad["ALPHA"] / ad["R"]

@torch.no_grad()
def fwd(seqemb, lora):
    Tn = seqemb.shape[0]; e = torch.cat([torch.outer(torch.arange(Tn, device=dev).float(), inv)] * 2, -1); cos, sin = e.cos(), e.sin()
    mask = torch.triu(torch.full((Tn, Tn), float("-inf"), device=dev), 1); x = seqemb
    for L in range(NL):
        b = Bw[L]; r = x; h = rms(x, b["iln"])
        q = h @ b["self_attn.q_proj"].T; v = h @ b["self_attn.v_proj"].T
        if lora is not None:
            q = q + SC * (h @ lora[f"{L}.qA"].T) @ lora[f"{L}.qB"].T; v = v + SC * (h @ lora[f"{L}.vA"].T) @ lora[f"{L}.vB"].T
        k = h @ b["self_attn.k_proj"].T; q = q.view(Tn, NH, HD); k = k.view(Tn, NKV, HD); v = v.view(Tn, NKV, HD)
        q = rms(q, b["qn"]); k = rms(k, b["kn"]); q = q * cos[:, None] + roth(q) * sin[:, None]; k = k * cos[:, None] + roth(k) * sin[:, None]
        k = k.repeat_interleave(NH // NKV, 1); v = v.repeat_interleave(NH // NKV, 1)
        q = q.transpose(0, 1); k = k.transpose(0, 1); v = v.transpose(0, 1)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(HD) + mask; att = att.softmax(-1)
        o = (att @ v).transpose(0, 1).reshape(Tn, NH * HD) @ b["self_attn.o_proj"].T
        x = r + o; r = x; h = rms(x, b["pln"]); x = r + (F.silu(h @ b["mlp.gate_proj"].T) * (h @ b["mlp.up_proj"].T)) @ b["mlp.down_proj"].T
    return rms(x, NORMF) @ HEAD.T

@torch.no_grad()
def gen(prompt, lora, n=40):
    ids = tok(prompt).input_ids
    for _ in range(n): ids.append(int(fwd(EMB[torch.tensor(ids[-T:], device=dev)], lora)[-1].argmax()))
    return tok.decode(ids)

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"
    for tag, lora in [("BASE", None), ("A/shakespeare", ad["A"]), ("B/alice", ad["B"])]:
        print(f"\n=== {tag} ===\n{gen(prompt, lora, 40)}", flush=True)
