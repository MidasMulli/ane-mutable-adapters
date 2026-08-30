"""
Full Qwen3.5-4B CoreML converter — all 32 layers + embedding + lm_head.

Scaled from proven 0.8B converter (gdn_full_model.py).
Architecture: 24 GatedDeltaNet (SSM) + 8 Full Attention layers.
Same tokenizer (248K vocab) as Qwen3.5-27B target — no cross-vocab issues.

For decode (seq_len=1), uses explicit state tensors — no Python cache objects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from pathlib import Path
from safetensors.torch import load_file

# ---------------------------------------------------------------------------
# CONFIG IS NOW SELECTABLE, DEFAULTING TO THE 4B SO EVERY EXISTING IMPORTER IS
# BYTE-IDENTICAL. Set GDN_VARIANT=0.8B to build Qwen3.5-0.8B, which is the SAME
# `qwen3_5` architecture at different dims (24L / 18 linear / 6 full, hidden 1024).
# Added 2026-08-12 to answer one question: does a hybrid of this shape place
# MONOLITHICALLY? The 0.8B lands near ~0.9 GB at pal8, below the 1.7351 GB highest
# artifact ever measured ANE_PREFERRED -- so it is the only hybrid candidate that
# might avoid chunking, the 512-token cap and the four-file ceiling in one step.
# Every value below is read from the repo's own config.json, never from card prose.
# ---------------------------------------------------------------------------
import os as _os
_V = _os.environ.get("GDN_VARIANT", "4B")
_CFG = {
    "4B":   dict(hidden=2560, layers=32, v_heads=32, k_heads=16, ffn=9216,
                 attn_heads=16, attn_kv=4, snap="models--Qwen--Qwen3.5-4B"),
    "0.8B": dict(hidden=1024, layers=24, v_heads=16, k_heads=16, ffn=3584,
                 attn_heads=8,  attn_kv=2, snap="models--Qwen--Qwen3.5-0.8B"),
}[_V]

HIDDEN_SIZE = _CFG["hidden"]
NUM_LAYERS = _CFG["layers"]
NUM_V_HEADS = _CFG["v_heads"]    # SSM value heads
NUM_K_HEADS = _CFG["k_heads"]    # SSM key heads
HEAD_K_DIM = 128
HEAD_V_DIM = 128
KEY_DIM = NUM_K_HEADS * HEAD_K_DIM
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM
CONV_DIM = KEY_DIM * 2 + VALUE_DIM
CONV_KERNEL = 4
RMS_EPS = 1e-6
# Attention params
ATTN_NUM_HEADS = _CFG["attn_heads"]
ATTN_NUM_KV_HEADS = _CFG["attn_kv"]
ATTN_HEAD_DIM = 256  # head_dim=256 in both configs
PARTIAL_ROTARY = 0.25
ROTARY_DIM = int(ATTN_HEAD_DIM * PARTIAL_ROTARY)  # 64
INTERMEDIATE_SIZE = _CFG["ffn"]
VOCAB_SIZE = 248320

# full_attention_interval=4 in both: every 4th layer starting at index 3
ATTN_LAYERS = set(range(3, NUM_LAYERS, 4))
NUM_SSM_LAYERS = NUM_LAYERS - len(ATTN_LAYERS)
NUM_ATTN_LAYERS = len(ATTN_LAYERS)

import glob as _glob
_snaps = sorted(_glob.glob(_os.path.expanduser(
    f"~/.cache/huggingface/hub/{_CFG['snap']}/snapshots/*")))
MODEL_PATH = _snaps[-1] if _snaps else ""


def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def rms_norm(x, weight, eps=1e-6):
    x_float = x.float()
    variance = x_float.pow(2).mean(-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + eps)
    # Qwen3.5 RMSNorm: weight is initialized to 0, actual scale is (1 + weight)
    return (x_normed * (1.0 + weight.float())).to(x.dtype)


class GDNLayer(nn.Module):
    """Single GatedDeltaNet SSM layer (decode step, seq_len=1)."""
    def __init__(self):
        super().__init__()
        self.in_proj_qkv = nn.Linear(HIDDEN_SIZE, CONV_DIM, bias=False)
        self.in_proj_z = nn.Linear(HIDDEN_SIZE, VALUE_DIM, bias=False)
        self.in_proj_b = nn.Linear(HIDDEN_SIZE, NUM_V_HEADS, bias=False)
        self.in_proj_a = nn.Linear(HIDDEN_SIZE, NUM_V_HEADS, bias=False)
        self.out_proj = nn.Linear(VALUE_DIM, HIDDEN_SIZE, bias=False)
        self.conv_weight = nn.Parameter(torch.randn(CONV_DIM, CONV_KERNEL))
        self.dt_bias = nn.Parameter(torch.ones(NUM_V_HEADS))
        self.A_log = nn.Parameter(torch.zeros(NUM_V_HEADS))
        self.norm_weight = nn.Parameter(torch.ones(HEAD_V_DIM))
        self.input_layernorm_weight = nn.Parameter(torch.ones(HIDDEN_SIZE))
        self.post_attn_layernorm_weight = nn.Parameter(torch.ones(HIDDEN_SIZE))
        # MLP
        self.gate_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.up_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False)

    def forward(self, hidden_states, conv_state, recurrent_state):
        """
        hidden_states: [1, 1, 2560]
        conv_state: [1, 8192, 3]
        recurrent_state: [1, 32, 128, 128]
        """
        residual = hidden_states

        # Pre-norm
        hidden_states = rms_norm(hidden_states, self.input_layernorm_weight, RMS_EPS)

        # --- GatedDeltaNet attention ---
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # Conv1d update
        mixed_qkv_t = mixed_qkv.transpose(1, 2)
        conv_input = torch.cat([conv_state, mixed_qkv_t], dim=2)
        new_conv_state = conv_input[:, :, 1:]
        # GDN_CONV4TAP (BLUEPRINT §6, 2026-07-29, decoded then and re-read 2026-08-13):
        # "the -14 is the grouped depthwise CONV1D specifically, NOT the recurrence (which is
        #  mutable-OK) ... rewritten as an explicit 4-tap weighted sum (elementwise, no
        #  grouped-conv op) it LOADS clean (no -14)."
        # The grouped F.conv1d below is that named cause, co-resident with mutable weights.
        # Mathematically identical: a depthwise conv over a CONV_KERNEL-tap causal window with
        # padding=0 and a single output position IS the elementwise product summed over taps.
        # Default stays the grouped form so every prior artifact remains byte-reproducible.
        if _os.environ.get("GDN_CONV4TAP"):
            conv_out = F.silu((conv_input * self.conv_weight.unsqueeze(0)).sum(dim=2, keepdim=True))
        else:
            conv_w = self.conv_weight.unsqueeze(1)
            conv_out = F.conv1d(conv_input, conv_w, groups=CONV_DIM, padding=0)
            conv_out = F.silu(conv_out)
        mixed_qkv = conv_out.transpose(1, 2)

        # Split Q, K, V
        query, key, value = mixed_qkv.split([KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
        query = query.reshape(1, 1, NUM_K_HEADS, HEAD_K_DIM)
        key = key.reshape(1, 1, NUM_K_HEADS, HEAD_K_DIM)
        value = value.reshape(1, 1, NUM_V_HEADS, HEAD_V_DIM)

        # Gates
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # L2 norm + scale
        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)
        scale = HEAD_K_DIM ** -0.5

        # Transpose to [1, H, 1, dim]
        query = (query * scale).transpose(1, 2).float()
        key = key.transpose(1, 2).float()
        value = value.transpose(1, 2).float()
        beta = beta.transpose(1, 2).float()
        g = g.transpose(1, 2).float()

        # Single step recurrence
        # Note: v_heads (32) != k_heads (16), need to repeat k/q for state ops
        kv_ratio = NUM_V_HEADS // NUM_K_HEADS  # 2
        q_t = query[:, :, 0].repeat_interleave(kv_ratio, dim=1)  # [1, 32, 128]
        k_t = key[:, :, 0].repeat_interleave(kv_ratio, dim=1)    # [1, 32, 128]
        v_t = value[:, :, 0]                                      # [1, 32, 128]
        g_t = g[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)        # [1, 32, 1, 1]
        beta_t = beta[:, :, 0].unsqueeze(-1)                      # [1, 32, 1]

        state = recurrent_state.float()
        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        attn_out = (state * q_t.unsqueeze(-1)).sum(dim=-2)
        new_recurrent_state = state.half()

        # RMSNormGated
        attn_out_flat = attn_out.reshape(-1, HEAD_V_DIM)
        z_flat = z.reshape(-1, HEAD_V_DIM).float()
        variance = attn_out_flat.pow(2).mean(-1, keepdim=True)
        attn_out_flat = attn_out_flat * torch.rsqrt(variance + RMS_EPS)
        attn_out_flat = self.norm_weight * attn_out_flat.half()
        attn_out_flat = attn_out_flat * F.silu(z_flat).half()

        hidden_states = self.out_proj(attn_out_flat.reshape(1, 1, VALUE_DIM))

        # Residual
        hidden_states = residual + hidden_states

        # --- MLP ---
        residual = hidden_states
        hidden_states = rms_norm(hidden_states, self.post_attn_layernorm_weight, RMS_EPS)
        hidden_states = self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        hidden_states = residual + hidden_states

        return hidden_states, new_conv_state, new_recurrent_state


class AttnLayer(nn.Module):
    """Single attention layer (decode step, seq_len=1)."""
    def __init__(self, context_length=256):
        super().__init__()
        self.context_length = context_length
        self.q_proj = nn.Linear(HIDDEN_SIZE, ATTN_NUM_HEADS * ATTN_HEAD_DIM * 2, bias=False)
        self.k_proj = nn.Linear(HIDDEN_SIZE, ATTN_NUM_KV_HEADS * ATTN_HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN_SIZE, ATTN_NUM_KV_HEADS * ATTN_HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(ATTN_NUM_HEADS * ATTN_HEAD_DIM, HIDDEN_SIZE, bias=False)
        self.q_norm_weight = nn.Parameter(torch.ones(ATTN_HEAD_DIM))
        self.k_norm_weight = nn.Parameter(torch.ones(ATTN_HEAD_DIM))
        self.input_layernorm_weight = nn.Parameter(torch.ones(HIDDEN_SIZE))
        self.post_attn_layernorm_weight = nn.Parameter(torch.ones(HIDDEN_SIZE))
        self.gate_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.up_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False)

    def forward(self, hidden_states, kv_cache, cos_cur, sin_cur, causal_mask):
        """
        hidden_states: [1, 1, 2560]
        kv_cache: [2, NUM_KV_HEADS, context_length, HEAD_DIM]
        cos_cur, sin_cur: [1, 1, 1, ROTARY_DIM]
        causal_mask: [1, 1, 1, context_length]
        """
        residual = hidden_states
        hidden_states = rms_norm(hidden_states, self.input_layernorm_weight, RMS_EPS)

        q_out = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Split q_proj output into query and gate
        q_out = q_out.reshape(1, 1, ATTN_NUM_HEADS, ATTN_HEAD_DIM * 2)
        q, gate = q_out.chunk(2, dim=-1)
        gate = gate.reshape(1, 1, -1)

        k = k.reshape(1, 1, ATTN_NUM_KV_HEADS, ATTN_HEAD_DIM)
        v = v.reshape(1, 1, ATTN_NUM_KV_HEADS, ATTN_HEAD_DIM)

        # QK Norm
        q = rms_norm(q, self.q_norm_weight, RMS_EPS)
        k = rms_norm(k, self.k_norm_weight, RMS_EPS)

        # RoPE (partial rotary)
        q_rot = q[..., :ROTARY_DIM]
        q_pass = q[..., ROTARY_DIM:]
        k_rot = k[..., :ROTARY_DIM]
        k_pass = k[..., ROTARY_DIM:]

        q1, q2 = q_rot[..., :ROTARY_DIM//2], q_rot[..., ROTARY_DIM//2:]
        q_rotated = torch.cat([-q2, q1], dim=-1)
        q_rot = q_rot * cos_cur + q_rotated * sin_cur

        k1, k2 = k_rot[..., :ROTARY_DIM//2], k_rot[..., ROTARY_DIM//2:]
        k_rotated = torch.cat([-k2, k1], dim=-1)
        k_rot = k_rot * cos_cur + k_rotated * sin_cur

        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Shift-left KV cache + append
        old_k = kv_cache[0:1]
        old_v = kv_cache[1:2]
        new_k_cache = torch.cat([old_k[:, :, 1:, :], k], dim=2)
        new_v_cache = torch.cat([old_v[:, :, 1:, :], v], dim=2)
        new_kv_cache = torch.cat([new_k_cache, new_v_cache], dim=0)

        # Repeat KV for GQA
        num_groups = ATTN_NUM_HEADS // ATTN_NUM_KV_HEADS
        k_full = new_k_cache.repeat_interleave(num_groups, dim=1)
        v_full = new_v_cache.repeat_interleave(num_groups, dim=1)

        # Attention with causal mask
        scale = ATTN_HEAD_DIM ** -0.5
        attn_weights = torch.matmul(q, k_full.transpose(-1, -2)) * scale
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        attn_out = torch.matmul(attn_weights, v_full)
        attn_out = attn_out.transpose(1, 2).reshape(1, 1, ATTN_NUM_HEADS * ATTN_HEAD_DIM)

        # Apply gate AFTER attention (attn_output_gate=true)
        attn_out = attn_out * torch.sigmoid(gate)

        hidden_states = self.o_proj(attn_out)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = rms_norm(hidden_states, self.post_attn_layernorm_weight, RMS_EPS)
        hidden_states = self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        hidden_states = residual + hidden_states

        return hidden_states, new_kv_cache


class Qwen35_4B_DecodeStep(nn.Module):
    """Full Qwen3.5-4B decode step — one token in, logits out."""
    def __init__(self, context_length=256):
        super().__init__()
        self.context_length = context_length
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.ssm_layers = nn.ModuleDict()
        self.attn_layers = nn.ModuleDict()
        for i in range(NUM_LAYERS):
            if i in ATTN_LAYERS:
                self.attn_layers[str(i)] = AttnLayer(context_length)
            else:
                self.ssm_layers[str(i)] = GDNLayer()
        self.final_norm_weight = nn.Parameter(torch.ones(HIDDEN_SIZE))


def load_full_weights(model: Qwen35_4B_DecodeStep):
    """Load all weights from HF Qwen3.5-4B."""
    print("Loading weights from safetensors...")
    sf_files = list(Path(MODEL_PATH).glob("*.safetensors"))
    weights = {}
    for f in sf_files:
        weights.update(load_file(str(f)))

    prefix = "model.language_model.layers."

    for i in range(NUM_LAYERS):
        lp = f"{prefix}{i}."

        if i in ATTN_LAYERS:
            layer = model.attn_layers[str(i)]
            layer.q_proj.weight.data = weights[lp + "self_attn.q_proj.weight"].float()
            layer.k_proj.weight.data = weights[lp + "self_attn.k_proj.weight"].float()
            layer.v_proj.weight.data = weights[lp + "self_attn.v_proj.weight"].float()
            layer.o_proj.weight.data = weights[lp + "self_attn.o_proj.weight"].float()
            layer.q_norm_weight.data = weights[lp + "self_attn.q_norm.weight"].float()
            layer.k_norm_weight.data = weights[lp + "self_attn.k_norm.weight"].float()
        else:
            layer = model.ssm_layers[str(i)]
            layer.in_proj_qkv.weight.data = weights[lp + "linear_attn.in_proj_qkv.weight"].float()
            layer.in_proj_z.weight.data = weights[lp + "linear_attn.in_proj_z.weight"].float()
            layer.in_proj_b.weight.data = weights[lp + "linear_attn.in_proj_b.weight"].float()
            layer.in_proj_a.weight.data = weights[lp + "linear_attn.in_proj_a.weight"].float()
            layer.out_proj.weight.data = weights[lp + "linear_attn.out_proj.weight"].float()
            conv_w = weights[lp + "linear_attn.conv1d.weight"].float()
            if conv_w.dim() == 3:
                conv_w = conv_w.squeeze(1)
            layer.conv_weight.data = conv_w
            layer.dt_bias.data = weights[lp + "linear_attn.dt_bias"].float()
            layer.A_log.data = weights[lp + "linear_attn.A_log"].float()
            layer.norm_weight.data = weights[lp + "linear_attn.norm.weight"].float()

        # Shared across both layer types
        target = model.attn_layers[str(i)] if i in ATTN_LAYERS else model.ssm_layers[str(i)]
        target.input_layernorm_weight.data = weights[lp + "input_layernorm.weight"].float()
        target.post_attn_layernorm_weight.data = weights[lp + "post_attention_layernorm.weight"].float()
        target.gate_proj.weight.data = weights[lp + "mlp.gate_proj.weight"].float()
        target.up_proj.weight.data = weights[lp + "mlp.up_proj.weight"].float()
        target.down_proj.weight.data = weights[lp + "mlp.down_proj.weight"].float()

    # Embedding (tied with lm_head)
    model.embed_tokens.weight.data = weights["model.language_model.embed_tokens.weight"].float()

    # Final norm
    model.final_norm_weight.data = weights["model.language_model.norm.weight"].float()

    print(f"Loaded weights for {NUM_LAYERS} layers + embedding + norm")
    return model


def build_rope_cache(context_length=256, base=10000000):
    """Build cosine/sine cache for RoPE."""
    inv_freq = 1.0 / (base ** (torch.arange(0, ROTARY_DIM, 2).float() / ROTARY_DIM))
    t = torch.arange(context_length).float()
    freqs = torch.outer(t, inv_freq)
    cos_cache = freqs.cos()
    sin_cache = freqs.sin()
    cos_cache = torch.cat([cos_cache, cos_cache], dim=-1)
    sin_cache = torch.cat([sin_cache, sin_cache], dim=-1)
    return cos_cache, sin_cache
