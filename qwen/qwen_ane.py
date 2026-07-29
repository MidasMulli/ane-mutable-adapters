"""Qwen3-0.6B in an ANE-friendly conv form (conv2d-as-linear, (1,C,1,S), matmul attention,
RoPE, GQA, per-head q/k RMSNorm, SwiGLU, fp16). Faithful to HF: logits cosine 1.000000 on a
reference prompt (see gen_check.py). Shared by deploy_lora.py.

Weights auto-download from the Hugging Face Hub on first use.
"""
import os, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from safetensors.torch import load_file

MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3-0.6B")
D, NL, NH, NKV, HD, IM, EPS, THETA, V = 1024, 28, 16, 8, 128, 3072, 1e-6, 1e6, 151936

def _weights():
    from huggingface_hub import snapshot_download
    snap = snapshot_download(MODEL_ID, allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.txt"])
    sf = glob.glob(os.path.join(snap, "*.safetensors"))[0]
    return load_file(sf)

class RMSc(nn.Module):  # RMSNorm over channels, weight (1,C,1,1)
    def __init__(s, C): super().__init__(); s.w = nn.Parameter(torch.ones(1, C, 1, 1))
    def forward(s, x): return x / torch.sqrt((x * x).mean(1, keepdim=True) + EPS) * s.w

class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.n1 = RMSc(D); s.n2 = RMSc(D)
        s.q = nn.Conv2d(D, NH * HD, 1, bias=False); s.k = nn.Conv2d(D, NKV * HD, 1, bias=False)
        s.v = nn.Conv2d(D, NKV * HD, 1, bias=False); s.o = nn.Conv2d(NH * HD, D, 1, bias=False)
        s.qn = nn.Parameter(torch.ones(1, 1, HD, 1)); s.kn = nn.Parameter(torch.ones(1, 1, HD, 1))
        s.gate = nn.Conv2d(D, IM, 1, bias=False); s.up = nn.Conv2d(D, IM, 1, bias=False); s.down = nn.Conv2d(IM, D, 1, bias=False)
    def rmsh(s, x, w): return x / torch.sqrt((x * x).mean(2, keepdim=True) + EPS) * w
    def forward(s, x, cos, sin, neg):
        T = x.shape[3]; r = x; h = s.n1(x)
        q = s.q(h).view(1, NH, HD, T); k = s.k(h).view(1, NKV, HD, T); v = s.v(h).view(1, NKV, HD, T)
        q = s.rmsh(q, s.qn); k = s.rmsh(k, s.kn)
        def roth(z): z1, z2 = z[:, :, :HD // 2], z[:, :, HD // 2:]; return torch.cat([-z2, z1], 2)
        q = q * cos + roth(q) * sin; k = k * cos + roth(k) * sin
        k = k.repeat_interleave(NH // NKV, 1); v = v.repeat_interleave(NH // NKV, 1)
        att = (q.transpose(-1, -2) @ k) / np.sqrt(HD) + neg; att = att.softmax(-1)
        o = (v @ att.transpose(-1, -2)).reshape(1, NH * HD, 1, T)
        x = r + s.o(o); r = x; h = s.n2(x)
        return r + s.down(F.silu(s.gate(h)) * s.up(h))

class Qwen(nn.Module):
    def __init__(s):
        super().__init__(); s.blocks = nn.ModuleList([Block() for _ in range(NL)]); s.nf = RMSc(D); s.head = nn.Conv2d(D, V, 1, bias=False)
    def forward(s, x, cos, sin, neg):
        for b in s.blocks: x = b(x, cos, sin, neg)
        return s.head(s.nf(x))

def rope_tab(T):
    inv = 1.0 / (THETA ** (torch.arange(0, HD, 2).float() / HD)); pos = torch.arange(T).float()
    emb = torch.cat([torch.outer(pos, inv)] * 2, -1)
    return emb.cos().T[None, None].contiguous(), emb.sin().T[None, None].contiguous()

def causal(T): return torch.triu(torch.full((1, 1, T, T), float("-inf")), 1)

def load(model):
    """Load real HF weights into a Qwen module. Returns an accessor g(key) for extra tensors (embeddings)."""
    W = _weights(); g = lambda k: W[k].float()
    def cw(conv, wt): conv.weight.data = wt[:, :, None, None].clone()
    for L, b in enumerate(model.blocks):
        p = f"model.layers.{L}."
        b.n1.w.data = g(p + "input_layernorm.weight").view(1, D, 1, 1)
        b.n2.w.data = g(p + "post_attention_layernorm.weight").view(1, D, 1, 1)
        cw(b.q, g(p + "self_attn.q_proj.weight")); cw(b.k, g(p + "self_attn.k_proj.weight"))
        cw(b.v, g(p + "self_attn.v_proj.weight")); cw(b.o, g(p + "self_attn.o_proj.weight"))
        b.qn.data = g(p + "self_attn.q_norm.weight").view(1, 1, HD, 1); b.kn.data = g(p + "self_attn.k_norm.weight").view(1, 1, HD, 1)
        cw(b.gate, g(p + "mlp.gate_proj.weight")); cw(b.up, g(p + "mlp.up_proj.weight")); cw(b.down, g(p + "mlp.down_proj.weight"))
    model.nf.w.data = g("model.norm.weight").view(1, D, 1, 1); cw(model.head, g("lm_head.weight"))
    return g
