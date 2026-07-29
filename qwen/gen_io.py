"""Host-side helpers for gen_qwen.mm: write the fp16 embedding + prompt ids, and decode outputs.

  python3 gen_io.py prep "Once upon a time"   # -> _out/emb.f16, _out/prompt.txt
  python3 gen_io.py decode _out/out_A.txt     # -> prints the generated text
"""
import os, sys, glob, warnings, numpy as np
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer
from qwen_ane import _weights, MODEL_ID

OUT = os.environ.get("QWEN_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out"))
os.makedirs(OUT, exist_ok=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID)

if sys.argv[1] == "prep":
    prompt = sys.argv[2]
    emb = _weights()["model.embed_tokens.weight"].half().numpy()
    emb.tofile(os.path.join(OUT, "emb.f16"))
    ids = tok(prompt).input_ids
    open(os.path.join(OUT, "prompt.txt"), "w").write(" ".join(map(str, ids)))
    print(f"  prep: emb {emb.shape} + prompt ids {ids}")
elif sys.argv[1] == "decode":
    line = open(sys.argv[2]).read().strip()
    prompt_ids = [int(x) for x in open(os.path.join(OUT, "prompt.txt")).read().split()]
    gen_ids = [int(x) for x in line.replace("TOKENS:", "").split()]
    print(tok.decode(prompt_ids + gen_ids))
