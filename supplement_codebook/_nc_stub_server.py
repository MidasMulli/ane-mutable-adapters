#!/usr/bin/env python3
"""A stub of the demo's HTTP surface, for the NEGATIVE CONTROLS only.

⛔ WHY THIS EXISTS. The first version of negative_controls.sh pointed demo_acceptance.py at a dead
port. Every control "held" -- and every one of them was refused by the server PRECONDITION at the
top of main(), which returns before G6 is ever reached. That is the arc's own recurring failure
(a control that never ran is indistinguishable from one that passed), reproduced inside the very
file written to prevent it. This stub answers /rendered and /decode so the acceptance script runs
its full length and the G6 branch is genuinely executed.

It is NOT a model and proves nothing about decoding. It exists so that a gate can be observed to
fire. Never import it from anything that measures.
"""
import json, os, sys, threading, http.server

GLYPHS = [{"path": p} for p in (os.environ.get("NC_GLYPHS") or "").split(",") if p]


class H(http.server.BaseHTTPRequestHandler):
    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/rendered"):
            return self._send(GLYPHS)
        self._send({})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        # A fixed, obviously-synthetic answer. The controls assert on WHICH GATE FIRES,
        # never on the value, so this must not look like a real decode.
        self._send({"text": "a", "bind_ms": 50.0 if req.get("arm") == "on" else 50.0,
                    "term": "eos", "ms": 1.0})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8799
    srv = http.server.HTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"stub up on {port}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
