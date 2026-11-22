"""neural_bridge (phase-2): Synora's two-layer gateway.

Policy (measured, honest):
  1. LOCAL FIRST - persona, dialog, math, synthesis, creative, KB,
     memory and retrieval all get their chance BEFORE the bridge.
     The bridge fires ONLY when every local layer declined
     (mode == "no_answer").
  2. IDENTITY NEVER LEAVES HOME - identity/greeting questions are
     hard-blocked here too (belt and braces over the upstream
     persona gate): Synora's identity is answered by Synora.
  3. HONEST OFFLINE - when the neural service is unreachable or
     disabled, the dignified local decline survives untouched and
     the failure is recorded. Nothing is invented.
  4. TRANSPARENCY - every bridged reply is labeled mode="neural"
     so the interface can show which layer answered.

Implementation: z-ai CLI (`z-ai chat -p ... -o tmp.json`) with a
timeout, one retry, JSON parsing. No credentials are printed or
stored. Errors: none (callers get None + meta)."""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time

BRIDGE = {"enabled": os.environ.get("SYNORA_NEURAL", "1") != "0",
          "timeout_s": 45, "calls": 0, "ok": 0, "failed": 0,
          "rate_limited": 0, "pacing_s": 3.0, "backoff_s": 25.0,
          "last_error": "", "_last_call_ts": 0.0}

_IDENTITY_RE = re.compile(
    r"من أنت|من انت|ما اسمك|مين انت|عرف نفسك|who are you|"
    r"what is your name|introduce yourself", re.I)


def _cli(prompt: str, timeout: float):
    """One CLI completion → (text, status) with status in
    {"ok", "429", "fail"}. Errors: none."""
    fd, out = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        p = subprocess.run(
            ["z-ai", "chat", "-p", prompt, "-o", out],
            capture_output=True, timeout=timeout, text=True)
        err = (p.stderr or "") + (p.stdout or "")
        if p.returncode != 0 or "429" in err:
            if "429" in err:
                BRIDGE["rate_limited"] += 1
                return None, "429"
            BRIDGE["last_error"] = err.strip().splitlines()[-1][:140] \
                if err.strip() else "cli failed"
            return None, "fail"
        d = json.load(open(out, encoding="utf-8"))
        c = d.get("choices", [{}])[0].get("message", {}) \
            .get("content", "")
        if isinstance(c, str) and c.strip():
            return c.strip(), "ok"
        BRIDGE["last_error"] = "empty completion"
        return None, "fail"
    except Exception as e:
        BRIDGE["last_error"] = str(e)[:140]
        return None, "fail"
    finally:
        try:
            os.unlink(out)
        except Exception:
            pass


def ask_neural(question: str, system: str | None = None,
               timeout: float | None = None):
    """One neural completion. Returns (text, meta) with text None
    on failure - never a guessed answer. Errors: none."""
    if not BRIDGE["enabled"]:
        return None, {"why": "disabled"}
    # pacing: a fixed breather between calls keeps the shared
    # service under its rate limit (measured 429 flood: 126
    # unpaced calls burned the quota in seconds)
    gap = time.time() - BRIDGE["_last_call_ts"]
    if gap < BRIDGE["pacing_s"]:
        time.sleep(BRIDGE["pacing_s"] - gap)
    BRIDGE["calls"] += 1
    t0 = time.time()
    prompt = f"{system}\n\n{question}" if system else question
    txt, st = _cli(prompt, timeout or BRIDGE["timeout_s"])
    if txt is None and st == "429":
        time.sleep(BRIDGE["backoff_s"])
        txt, st = _cli(prompt, timeout or BRIDGE["timeout_s"])
    BRIDGE["_last_call_ts"] = time.time()
    dt = round(time.time() - t0, 2)
    if txt:
        BRIDGE["ok"] += 1
        return txt, {"available": True, "latency_s": dt,
                     "model": "glm-via-zai-bridge"}
    BRIDGE["failed"] += 1
    return None, {"available": False, "latency_s": dt,
                  "status": st, "error": BRIDGE["last_error"]}


def neural_fallback(question: str, raw_result: dict):
    """Called ONLY after every local layer declined. Returns an
    updated result dict (neural reply, or the honest local decline
    annotated with the bridge failure). Errors: none."""
    if _IDENTITY_RE.search(question or ""):
        raw_result["neural"] = {"available": False,
                                "why": "identity-protected"}
        return raw_result
    txt, meta = ask_neural(question)
    if not txt:
        raw_result["neural"] = dict(meta)
        return raw_result
    return {"mode": "neural", "layer": "neural",
            "reply": txt, "url": "", "http_calls": 0,
            "neural": dict(meta),
            "latency_s": meta.get("latency_s", 0)}


def stats() -> dict:
    return dict(BRIDGE)
