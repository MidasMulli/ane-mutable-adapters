"""CODEBOOK LIVE DEMO — one unified web UI over the PROVEN engines (operator 2026-08-23).

Wraps, does NOT reinvent:
  - ane_vlm (V.encode/V.tower/V.decode_lm)      the on-ANE image decode (arm off=base, on=adapter)
  - deploy_codebook.sh                          torch ckpt -> conv mutable blob -> ANE stage (SWAP)
  - adapter_trainer.py + data_cbmix_L*/         the 1/5 held-out training regime (live loss curve)
  - ane_rate.py (xctrace)                       per-dispatch ANE capture (no entitlement/kext/reboot)

One page, four stacked panels: TRAIN (live loss curve) · GLYPH (drop or pick) · DECODE (base|swap|
adapter) · DISPATCH (xctrace chart). Resident ANE engine, decodes serialized (one ANE gen at a time).
Run:  sudo -v && python3 codebook_web.py            open http://127.0.0.1:8711
      (set V_ENV= if the ANE decode needs a different interpreter than the one you launch with)
(sudo -v primes the passwordless timestamp so deploy_codebook's staging works without a prompt.)
"""
import os, sys, json, time, threading, io, base64, subprocess, http.server, socketserver, urllib.parse, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
# VENDORED 2026-08-30: ane_vlm now ships at vendor/tower/ instead of a sibling `tower/` that
# was never part of the supplement. `{HERE}/tower` is kept on the path so an in-place
# development checkout still resolves, but vendor/ is searched first.
sys.path.insert(0, f"{HERE}/tower"); sys.path.insert(0, f"{HERE}/vendor/tower")
PORT = 8711
# ⛔ WAS A HARDCODED VIRTUALENV IN ONE DEVELOPER'S HOME. The decode subprocess needs an
# interpreter with torch + transformers + PIL; by default that is simply the one running
# this server. Override with V_ENV= when the demo is launched from a different env.
V_ENV = os.environ.get("V_ENV") or sys.executable
TRAIN_RUN = os.path.join(HERE, "runs_cbmix_live")           # the live 1/5 regime run (train_web-compatible report)
RENDERED = os.path.join(HERE, "data_codebook", "compose")   # the rendered glyph set for the picker
PROMPT = os.environ.get("EVAL_PROMPT", "Decode the symbols.")

_lock = threading.Lock()          # ANE: one generation at a time
_V = None                         # resident vision engine
_train_proc = None
_lm = {"proc": None, "arm": None}  # RESIDENT LM binary: model stays on the ANE, only re-binds on arm CHANGE

def load_engine():
    global _V
    if _V is None:
        import ane_vlm as V; _V = V
    return _V

def _lm_start():
    """Spawn q35_swap_img ONCE. It loads the model to the ANE and waits; READY names the bound arm."""
    V = _V
    p = subprocess.Popen(["/usr/bin/sudo", "-n", "./q35_swap_img_nb", V.CDIR, V.CTRUST,
                          "./embed.mlmodelc", "./lmhead.mlmodelc", ".", "3"],
                         cwd=V.DEMO, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=open(os.path.join(HERE, ".lm_resident.err"), "w"), text=True, bufsize=1)
    ready = p.stdout.readline()                       # "READY off 64.8 units=3"
    arm = ready.split()[1] if ready.startswith("READY") and len(ready.split()) > 1 else "off"
    _lm["proc"], _lm["arm"] = p, arm
    return arm

def resident_decode(img_path, arm):
    """Resident LM: bind cost is incurred ONLY when arm != currently-bound arm (base<->adapter)."""
    V = load_engine()
    with _lock:
        t = time.time()
        px, ids, p0, nimg = V.encode(V.load_rgb(img_path), PROMPT)
        vecs = V.tower(px, "BASE")
        if _lm["proc"] is None or _lm["proc"].poll() is not None:
            _lm_start()
        p = _lm["proc"]
        prev_arm = _lm["arm"]; changed = (arm != prev_arm)
        vecs.astype("float32").tofile(os.path.join(V.HERE, "_vlm_emb.f32"))
        stoparg = f"{len(V.STOPS)} " + " ".join(map(str, V.STOPS))
        cmd = (f"GENI {V.HERE}/_vlm_emb.f32 {p0} {nimg} {1 if arm=='on' else 0} {len(ids)} 16 "
               f"{stoparg} " + " ".join(map(str, ids)))
        p.stdin.write(cmd + "\n"); p.stdin.flush()
        toks, term, started, bind_ms = [], "?", None, None
        while True:
            l = p.stdout.readline()
            if not l: break
            if l.startswith("GENIBEGIN"):
                started = l.split("start=")[1].split()[0].strip() if "start=" in l else None
                if "bind_ms=" in l:
                    try: bind_ms = float(l.split("bind_ms=")[1].split()[0])
                    except Exception: pass
            elif l.startswith("TOK "): toks.append(int(l.split()[1]))
            elif l.startswith("GENIDONE"):
                term = l.split("term=")[1].strip() if "term=" in l else "?"; break
        _lm["arm"] = started or arm
        txt = V.TOK.decode(toks, skip_special_tokens=True).strip()
        # q35_swap_img_nb (gate PASS 2026-08-23): GENI binds ONLY on an arm change. Same-arm
        # repeat = bind_ms 0.0 (resident weights reused, tokens byte-identical to bind-every-time).
        # ⛔ CONTRACT: any re-deploy of staged bytes must go through _lm_force_swap -- the skip
        # is keyed on the arm flag, not on the bytes (see geni_rebind_gate.py header).
        bm = round(bind_ms, 1) if bind_ms is not None else None
        return txt, term, round((time.time() - t) * 1000), bm, changed, prev_arm


def _lm_force_swap(arm="on"):
    """Force a rebind of freshly-staged bytes on the resident LM. Returns swap ms or None."""
    with _lock:
        if _lm["proc"] is None or _lm["proc"].poll() is not None:
            return None                      # next GENI on a fresh spawn binds the new bytes anyway
        p = _lm["proc"]
        p.stdin.write(f"SWAP {arm}\n"); p.stdin.flush()
        while True:
            l = p.stdout.readline()
            if not l: return None
            if l.startswith("SWAPPED"):
                _lm["arm"] = arm
                try: return round(float(l.split()[2]), 1)
                except Exception: return None

def decode(img_path, arm):
    return resident_decode(img_path, arm)   # (txt, term, ms, bind_ms, changed, prev_arm)

# ---- dispatch capture (xctrace, per-action; honest v1: capture-on-demand, not continuous stream) ----
def capture_dispatch(img_path):
    """Run ONE decode under xctrace and return per-dispatch intervals (us) + rate."""
    script = f"""
import sys,os; H={HERE!r}; sys.path.insert(0,H); sys.path.insert(0,H+'/tower'); sys.path.insert(0,H+'/vendor/tower')
import ane_vlm as V
px,ids,p0,nimg=V.encode(V.load_rgb({img_path!r}),{PROMPT!r})
vecs=V.tower(px,'BASE'); V.decode_lm(vecs,ids,p0,nimg,'on',ngen=16)
"""
    sp = os.path.join(HERE, "_cbdemo_disp.py"); open(sp, "w").write(script)
    out = os.path.join(HERE, "_cbdemo_ne.trace")
    subprocess.run(["rm", "-rf", out], check=False)
    r = subprocess.run(["xcrun", "xctrace", "record", "--instrument", "Neural Engine",
                        "--output", out, "--launch", "--", V_ENV, sp],
                       capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        return {"error": "xctrace failed", "detail": r.stderr[-400:]}
    xml = os.path.join(HERE, "_cbdemo_ane.xml")
    subprocess.run(["xcrun", "xctrace", "export", "--input", out, "--xpath",
                    '/trace-toc/run[@number="1"]/data/table[@schema="ane-hw-intervals"]',
                    "--output", xml], capture_output=True, check=False)
    import xml.etree.ElementTree as ET
    cache = {}
    def val(el):
        if el is None: return None
        ref = el.get("ref")
        if ref: return cache.get(ref)          # xctrace id/ref reuse — the 3.8x trap (ane_rate.py)
        if el.get("id"): cache[el.get("id")] = el.text
        return el.text
    try:
        rows = ET.parse(xml).getroot().findall(".//row")
    except Exception as e:
        return {"error": "no dispatch table", "detail": str(e)}
    pairs = sorted(((int(val(rw.find("start-time"))), int(val(rw.find("duration"))))
                    for rw in rows if val(rw.find("start-time")) and val(rw.find("duration"))))
    if not pairs:
        return {"error": "no dispatches captured"}
    t0 = pairs[0][0]
    starts_ms = [round((s - t0) / 1e6, 3) for s, _ in pairs]         # ms from first dispatch
    dur_us = [round(d / 1000, 3) for _, d in pairs]                  # µs per dispatch
    span_ms = (pairs[-1][0] + pairs[-1][1] - t0) / 1e6
    busy_ms = sum(d for _, d in pairs) / 1e6
    sd = sorted(dur_us)
    return {"n": len(pairs), "starts_ms": starts_ms, "dur_us": dur_us,
            "span_ms": round(span_ms, 1), "busy_ms": round(busy_ms, 2),
            "rate_per_s": round(len(pairs) / (span_ms / 1000), 0) if span_ms else 0,
            "duty_pct": round(100 * busy_ms / span_ms, 1) if span_ms else 0,
            "mean_us": round(sum(dur_us) / len(dur_us), 1), "median_us": sd[len(sd) // 2],
            "max_us": max(dur_us)}

def rendered_set(n=24):
    """Serve a mix: single glyphs (L1, held-out length), five-glyph (L5, held-out), and 4-char combos."""
    import json as _j
    picks = []
    for L in ("L1", "L5"):
        f = os.path.join(HERE, f"data_cbmix_{L}", "compose.jsonl")
        if os.path.exists(f):
            for line in list(open(f))[:8]:
                picks.append(_j.loads(line)["image"])
    picks += sorted(glob.glob(os.path.join(RENDERED, "*.png")))[:8]
    return [os.path.relpath(p, HERE) for p in picks[:n]]

def train_report():
    p = os.path.join(TRAIN_RUN, "training_report.json")
    if not os.path.exists(p): return {"status": "no run yet"}
    try: return json.load(open(p))
    except Exception: return {"status": "partial"}

def img_b64(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()

PAGE = open(os.path.join(HERE, "codebook_web.html")).read() if os.path.exists(os.path.join(HERE, "codebook_web.html")) else "<!--PAGE-->"

class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, body, ct="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(200); self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        if u.path == "/":
            self._send(PAGE, "text/html")
        elif u.path == "/rendered":
            self._send(json.dumps([{"path": p, "b64": img_b64(os.path.join(HERE, p))} for p in rendered_set()]))
        elif u.path == "/train_status":
            self._send(json.dumps(train_report()))
        elif u.path == "/staged":
            md5 = subprocess.run(["/usr/bin/sudo", "-n", "md5", "-q",
                  "/private/var/db/AppleIntelligencePlatform/AppModelAssets/q35_08b_conv/gc0_on.bin"],
                  capture_output=True, text=True).stdout.strip()[:16]
            self._send(json.dumps({"staged_md5": md5}))
        else:
            self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = self.rfile.read(n)
        u = urllib.parse.urlparse(self.path)
        try:
            data = json.loads(body) if body else {}
        except Exception:
            data = {}
        if u.path == "/decode":
            img = data.get("img"); arm = data.get("arm", "off")
            if data.get("b64"):                       # dropped image -> temp file
                img = os.path.join(HERE, "_cbdemo_drop.png")
                open(img, "wb").write(base64.b64decode(data["b64"].split(",")[-1]))
            elif img: img = os.path.join(HERE, img)
            try:
                txt, term, ms, bind_ms, changed, prev = decode(img, arm)
                self._send(json.dumps({"text": txt, "term": term, "ms": ms, "bind_ms": bind_ms,
                                       "arm": arm, "changed": changed, "prev_arm": prev}))
            except Exception as e:
                import traceback; self._send(json.dumps({"error": str(e), "tb": traceback.format_exc()[-500:]}))
        elif u.path == "/dispatch":
            img = data.get("img") or rendered_set(1)[0]
            self._send(json.dumps(capture_dispatch(os.path.join(HERE, img))))
        elif u.path == "/swap":
            # deploy the FRESHLY-TRAINED 1/5 best checkpoint (not the v1 default)
            env = dict(os.environ); env["VERIFY"] = "0"          # fast stage for the demo (~20s, skips 180-img verify)
            rep = os.path.join(TRAIN_RUN, "training_report.json")
            if os.path.exists(rep):
                bc = json.load(open(rep)).get("best_ckpt")
                if bc: env["CK"] = os.path.join(HERE, bc) if not os.path.isabs(bc) else bc
            r = subprocess.run(["/bin/bash", os.path.join(HERE, "deploy_codebook.sh")],
                               capture_output=True, text=True, timeout=300, env=env)
            # staged bytes changed under the same suffix -> force the resident LM to rebind them
            # (the nb binary's same-arm skip is keyed on the arm flag, not the bytes)
            swap_ms = _lm_force_swap("on") if r.returncode == 0 else None
            self._send(json.dumps({"ok": r.returncode == 0, "ck": env.get("CK", "v1 default"),
                                   "swap_ms": swap_ms,
                                   "log": (r.stdout + r.stderr)[-1400:]}))
        elif u.path == "/train_start":
            global _train_proc
            if _train_proc and _train_proc.poll() is None:
                self._send(json.dumps({"error": "already training"})); return
            _train_proc = launch_training()
            self._send(json.dumps({"ok": True, "run": TRAIN_RUN}))
        else:
            self.send_error(404)

def launch_training():
    """Live 1/5 held-out regime: train on L2/L3/L4 fit, hold out L1/L5, via the canonical trainer."""
    train_jsonl = os.path.join(HERE, "_cbmix_train_live.jsonl")
    with open(train_jsonl, "w") as w:
        for L in ("L2", "L3", "L4"):
            fp = os.path.join(HERE, f"data_cbmix_{L}", "fit.jsonl")
            if os.path.exists(fp):
                for line in open(fp): w.write(line)
    held = os.path.join(HERE, "data_cbmix_L1", "compose.jsonl")
    snaps = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*"))
    model = snaps[0] if snaps else "Qwen/Qwen3.5-0.8B"
    cmd = [V_ENV, os.path.join(HERE, "..", "..", "ane-mutable-weights", "toolkit", "train", "adapter_trainer.py"),
           "--model", model, "--train", train_jsonl, "--heldout", held,
           "--out", TRAIN_RUN, "--image-field", "image", "--image-prompt", PROMPT,
           "--deploy-target", os.path.join(HERE, "_mlc_shadow", "gdnmut_gc0.mlmodelc"),
           "--epochs", "4", "--eval-every", "50", "--seed", "0"]
    log = open(os.path.join(HERE, "_cbmix_train_live.log"), "w")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=HERE)

if __name__ == "__main__":
    print(f"loading ANE engine ...", flush=True); load_engine(); print("engine resident.", flush=True)
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    BIND = os.environ.get("CB_BIND", "127.0.0.1")   # set CB_BIND=0.0.0.0 to serve on the LAN
    with socketserver.ThreadingTCPServer((BIND, PORT), H) as s:
        print(f"codebook demo -> http://{BIND}:{PORT}", flush=True); s.serve_forever()
