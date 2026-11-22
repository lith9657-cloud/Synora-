#!/usr/bin/env python3
"""synora_server.py (fix19) — ZERO-DEPENDENCY server for the
Synora web interface (stdlib http.server only; numpy+tokenizers
needed for the neural layer, which degrades gracefully without).

Endpoints:
  GET  /            -> index.html (the 3D bilingual interface)
  GET  /api/health  -> {"ok": true, "engine": "..."}
  POST /api/chat    -> {"q": "..."} -> webmind.answer() reply dict
  POST /api/teach   -> {"q": "..."} -> learn pass (optional)

Layout: place synora_server.py NEXT TO the synora16_model_clean
package folder (or inside synora19_web/ beside it). Works on:
local machine, PythonAnywhere, Hugging Face Spaces (Flask-free),
Render free tier. See README_DEPLOY.md.
Errors: none (every request inside try)."""
import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [
    os.path.join(HERE, "synora16_model_clean"),
    os.path.join(os.path.dirname(HERE), "synora16_model_clean"),
    "/home/z/my-project/download/synora16_model_clean",
]
PKG = next((c for c in CANDIDATES if os.path.isdir(c)), None)
if PKG:
    sys.path.insert(0, PKG)
    sys.path.insert(0, os.path.join(PKG, "tools"))
    sys.path.insert(0, HERE)

_INDEX = os.path.join(HERE, "index.html")


def _engine():
    try:
        import webmind
        return webmind
    except Exception as ex:
        print("engine load failed:", ex, flush=True)
        return None


class Handler(SimpleHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type",
                         "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            path = self.path.split("?")[0]
            if path == "/api/health":
                wm = _engine()
                return self._json(200, {
                    "ok": wm is not None,
                    "engine": "synora-full" if wm else "demo",
                })
            if path in ("/", "/index.html"):
                with open(_INDEX, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type",
                                 "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            return super().do_GET()
        except Exception as ex:
            return self._json(500, {"error": str(ex)})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n).decode("utf-8"))
            q = str(req.get("q", ""))[:2000]
            if self.path == "/api/chat":
                wm = _engine()
                if wm is None:
                    return self._json(200, {
                        "reply": "Engine offline — running demo.",
                        "mode": "demo"})
                wm.fetch_source = lambda *a, **k: ("", "", 0)
                r = wm.answer(q)
                return self._json(200, {
                    "reply": r.get("reply", ""),
                    "mode": r.get("mode", ""),
                    "via": r.get("via", ""),
                    "polish": r.get("polish", {}),
                })
            if self.path == "/api/teach":
                wm = _engine()
                if wm is None:
                    return self._json(200, {"learned_pairs": 0})
                return self._json(200, wm.teach(q))
            return self._json(404, {"error": "unknown endpoint"})
        except Exception as ex:
            return self._json(500, {"error": str(ex)})

    def log_message(self, fmt, *args):
        sys.stderr.write("synora: " + (fmt % args) + "\n")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"SYNORA server on http://0.0.0.0:{port} "
          f"(engine: {'loaded' if _engine() else 'offline'})",
          flush=True)
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
