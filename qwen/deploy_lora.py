"""Deploy the trained q,v LoRA to the ANE as PARALLEL-DELTA mutable convs.

  proj = base(x) + delta(x)      base FROZEN in weight.bin, delta MUTABLE in adapter.bin
  delta = (alpha/r) * B @ A       folded to a full [out,in] conv per projection

Builds the base+delta model (`_out/qwen_lora.mlmodelc`), declares the 56 delta consts mutable
(BlobFileMutabilityInfo, all pointing at one adapter.bin), and packs a folded adapter.bin per task
(`_out/adapter_A.bin`, `_out/adapter_B.bin`). The base weight.bin is one file, byte-constant across
tasks. cos/sin/causal-mask are baked so the model has a single input x (embedded tokens).
"""
import os, re, shutil, subprocess, numpy as np, torch, torch.nn as nn, warnings
warnings.filterwarnings("ignore")
import coremltools as ct
from qwen_ane import Qwen, load, rope_tab, causal, D, NL, NH, NKV, HD

OUT = os.environ.get("QWEN_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out"))
T = 64
ad = torch.load(os.path.join(OUT, "lora_adapters.pt"), map_location="cpu"); SC = ad["ALPHA"] / ad["R"]

class QwenD(nn.Module):
    def __init__(s):
        super().__init__(); s.base = Qwen()
        s.qd = nn.ModuleList([nn.Conv2d(D, NH * HD, 1, bias=False) for _ in range(NL)])
        s.vd = nn.ModuleList([nn.Conv2d(D, NKV * HD, 1, bias=False) for _ in range(NL)])
        c, sn = rope_tab(T); s.register_buffer("cc", c); s.register_buffer("ss", sn); s.register_buffer("ng", causal(T))
    def forward(s, x):
        cos, sin, neg = s.cc, s.ss, s.ng
        for L, b in enumerate(s.base.blocks):
            r = x; h = b.n1(x)
            q = (b.q(h) + s.qd[L](h)).view(1, NH, HD, T); k = b.k(h).view(1, NKV, HD, T); v = (b.v(h) + s.vd[L](h)).view(1, NKV, HD, T)
            q = b.rmsh(q, b.qn); k = b.rmsh(k, b.kn)
            def roth(z): z1, z2 = z[:, :, :HD // 2], z[:, :, HD // 2:]; return torch.cat([-z2, z1], 2)
            q = q * cos + roth(q) * sin; k = k * cos + roth(k) * sin
            k = k.repeat_interleave(NH // NKV, 1); v = v.repeat_interleave(NH // NKV, 1)
            att = (q.transpose(-1, -2) @ k) / np.sqrt(HD) + neg; att = att.softmax(-1)
            o = (v @ att.transpose(-1, -2)).reshape(1, NH * HD, 1, T)
            x = r + b.o(o); r = x; h = b.n2(x)
            x = r + b.down(torch.nn.functional.silu(b.gate(h)) * b.up(h))
        return s.base.head(s.base.nf(x))

def build():
    m = QwenD().eval(); load(m.base)
    torch.manual_seed(0)
    for L in range(NL):  # small-random init so the delta consts are stored as BLOBFILE (repointed below)
        m.qd[L].weight.data.normal_(0, 1e-4); m.vd[L].weight.data.normal_(0, 1e-4)
    ml = ct.convert(torch.jit.trace(m, torch.zeros(1, D, 1, T)), inputs=[ct.TensorType(name="x", shape=(1, D, 1, T), dtype=np.float16)],
                    outputs=[ct.TensorType(name="logits", dtype=np.float16)], convert_to="mlprogram",
                    compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE, minimum_deployment_target=ct.target.macOS15)
    pk = os.path.join(OUT, "qwen_lora.mlpackage"); shutil.rmtree(pk, ignore_errors=True); ml.save(pk)
    md = os.path.join(OUT, "qwen_lora.mlmodelc"); shutil.rmtree(md, ignore_errors=True)
    subprocess.run(f"xcrun coremlcompiler compile '{pk}' '{OUT}' && mv '{OUT}/qwen_lora.mlmodelc' '{md}'", shell=True, capture_output=True)
    return md

def fold(A):  # 56 folded deltas in order [(L,qd),(L,vd)]
    ds = []
    for L in range(NL):
        ds.append(SC * (A[f"{L}.qB"] @ A[f"{L}.qA"])); ds.append(SC * (A[f"{L}.vB"] @ A[f"{L}.vA"]))
    return ds

def pack(ds, tag):
    class P(nn.Module):
        def __init__(s): super().__init__(); s.c = nn.ModuleList([nn.Conv2d(D, d.shape[0], 1, bias=False) for d in ds])
        def forward(s, x): return sum(cc(x).sum() for cc in s.c)
    P_ = P().eval()
    with torch.no_grad():
        for i, d in enumerate(ds): P_.c[i].weight.copy_(d[:, :, None, None])
    ml = ct.convert(torch.jit.trace(P_, torch.rand(1, D, 1, T)), inputs=[ct.TensorType(name="x", shape=(1, D, 1, T), dtype=np.float16)],
                    convert_to="mlprogram", compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15)
    pk = os.path.join(OUT, f"_pk_{tag}.mlpackage"); shutil.rmtree(pk, ignore_errors=True); ml.save(pk)
    o = os.path.join(OUT, f"_pk_{tag}"); shutil.rmtree(o, ignore_errors=True)
    subprocess.run(f"xcrun coremlcompiler compile '{pk}' '{o}'", shell=True, capture_output=True)
    mdir = subprocess.run(f"ls -d '{o}'/*.mlmodelc", shell=True, capture_output=True, text=True).stdout.strip()
    mil = open(os.path.join(mdir, "model.mil")).read(); offs = {}
    for cn in re.findall(r'c_\d+_weight_to_fp16', mil):
        i = mil.find(cn + " = const"); e = mil.find(";", i)
        offs[int(re.search(r'c_(\d+)_', cn).group(1))] = int(re.search(r'weight\.bin"\), offset = uint64\((\d+)\)', mil[i:e]).group(1))
    return open(os.path.join(mdir, "weights", "weight.bin"), "rb").read(), [offs[i] for i in range(len(ds))]

if __name__ == "__main__":
    md = build(); print("  built", md, flush=True)
    binA, offs = pack(fold(ad["A"]), "A"); binB, offs2 = pack(fold(ad["B"]), "B"); assert offs == offs2
    open(os.path.join(OUT, "adapter_A.bin"), "wb").write(binA); open(os.path.join(OUT, "adapter_B.bin"), "wb").write(binB)
    open(os.path.join(md, "weights", "adapter.bin"), "wb").write(binA)  # a valid baked default
    s = open(os.path.join(md, "model.mil")).read()
    consts = sorted(set(re.findall(r'qd_\d+_weight_to_fp16|vd_\d+_weight_to_fp16', s)),
                    key=lambda z: (int(re.search(r'_(\d+)_', z).group(1)), 0 if z.startswith("qd") else 1))
    assert len(consts) == 56, len(consts)
    for k, cn in enumerate(consts):
        i = s.find(cn + " = const"); e = s.find(";", i); stmt = s[i:e]; assert "weight.bin" in stmt, cn
        new = stmt.replace("@model_path/weights/weight.bin", "@model_path/weights/adapter.bin")
        new = re.sub(r'offset = uint64\(\d+\)', f'offset = uint64({offs[k]})', new, count=1)
        s = s[:i] + new + s[e:]
    mk = '{"coremltools-version", "9.0"}})]'
    s = s.replace(mk, mk[:-1] + ', BlobFileMutabilityInfo = tuple<string, dict<string, string>>(("Paths", {{"@model_path/weights/adapter.bin", "@model_path/weights/adapter.bin"}}))]')
    open(os.path.join(md, "model.mil"), "w").write(s)
    print(f"  packed {len(offs)} deltas -> adapter_A.bin / adapter_B.bin ({len(binA)//10**6} MB each); 56 delta consts declared mutable", flush=True)
