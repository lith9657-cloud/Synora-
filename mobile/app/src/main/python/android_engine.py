"""Android-local HTTP bridge for the existing Synora Python engine.

The server binds only to 127.0.0.1. Persistent memory and lexicon files
are redirected to app-private storage. The optional neural layer is
explicitly disabled until its PyTorch runtime and model assets are ported
and verified on Android.
"""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

_SERVER = None
_SERVER_LOCK = threading.Lock()


def start_server(home: str, port: int = 8765) -> bool:
    global _SERVER

    with _SERVER_LOCK:
        if _SERVER is not None:
            return True

        data_dir = os.path.join(home, "synora-data")
        os.makedirs(data_dir, exist_ok=True)

        os.environ["HOME"] = home
        os.environ["WEBMIND_OUTD"] = data_dir
        # The engine treats an explicitly invalid path as a disable switch.
        os.environ["WEBMIND_NEURAL"] = "android-neural-layer-not-installed"

        # APK contents are read-only: persist the learned lexicon privately.
        import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))

    import translation_core
        translation_core._LEX_PATH = os.path.join(data_dir, "lexicon_pairs.jsonl")

        import webmind

        index_path = os.path.join(home, "synora-index.html")

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status: int, payload: bytes,
                      content_type: str = "application/json; charset=utf-8") -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

            def _json(self, status: int, obj: dict) -> None:
                body = json.dumps(obj, ensure_ascii=False,
                                  default=str).encode("utf-8")
                self._send(status, body)

            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path in ("/", "/index.html"):
                    try:
                        with open(index_path, "rb") as stream:
                            self._send(200, stream.read(), "text/html; charset=utf-8")
                    except Exception as error:
                        self._json(500, {"error": str(error)})
                    return

                if path == "/api/health":
                    self._json(200, {
                        "ok": True,
                        "engine": "synora-android-python-core",
                        "runtime": "local",
                        "neural": "disabled-pytorch-not-ported",
                    })
                    return

                self._json(404, {"error": "not_found"})

            def do_POST(self) -> None:
                path = urlparse(self.path).path
                try:
                    length = min(int(self.headers.get("Content-Length", "0")), 16000)
                    request = json.loads(self.rfile.read(length).decode("utf-8"))
                    question = str(request.get("q", ""))[:2000]

                    if path == "/api/chat":
                        result = webmind.answer(question)
                        self._json(200, {
                            "reply": result.get("reply", ""),
                            "mode": result.get("mode", ""),
                            "via": result.get("via", ""),
                            "polish": result.get("polish", {}),
                            "understanding": result.get("understanding"),
                            "latency_s": result.get("latency_s"),
                            "source": result.get("source"),
                        })
                        return

                    if path == "/api/teach":
                        self._json(200, webmind.teach(question))
                        return

                    self._json(404, {"error": "unknown_endpoint"})
                except Exception as error:
                    self._json(500, {"error": str(error)})

            def log_message(self, fmt: str, *args) -> None:
                # Do not write user prompts into Android's system log.
                return

        server = ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)
        server.daemon_threads = True
        thread = threading.Thread(
            target=server.serve_forever,
            name="synora-local-api",
            daemon=True,
        )
        thread.start()
        _SERVER = server
        return True
