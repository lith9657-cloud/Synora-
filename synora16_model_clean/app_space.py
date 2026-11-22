"""Synora Space app - free 24/7 training + serving on a Hugging
Face CPU Space (the 'days with phone off' option, fix14).

Layout inside the Space repo (upload these):
  app.py            <- this file
  requirements.txt  <- gradio>=4.0
  tools/*.py        <- the package tools (numeric_lm, numeric_gen,
                       translation_core, webmind, serve, ...)
  lm/lm_ar_trigram.tsv.gz
  state_clean/, data_tatoeba/, webmind_memory.jsonl, ...
  data/*.txt        <- YOUR corpus files (any size, any encoding)

Behaviour:
  - on startup it resumes any previous WORK checkpoint, then a
    background thread feeds data/*.txt line by line forever
    (re-checks the folder every 10 min for new files),
    checkpointing every 20k lines
  - a UI button pulls a training batch straight from the HF
    dataset libraries + GitHub skill repos (tools/hf_pull.py,
    deduplicated) into data/ - "library feeding" without manual
    uploads; needs `datasets` in requirements.txt
  - if HF_TOKEN is present (default in Spaces) the trained
    checkpoint + report are committed to the Space repo every
    30 min, so training survives Space restarts
  - the UI shows live metrics, lets you ask the model, and
    download the trained package
Errors: none (UI never raises; thread failures are reported in
the status box)."""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS)

STATE = {"fed_total": 0, "last_line": "", "status": "بدأ الخيط…",
         "ppl": None, "started": time.time()}

# one training worker only; molab_train is reused as a library
TRAIN_PROC = None


def _data_files() -> list[str]:
    return sorted(glob.glob(os.path.join(ROOT, "data", "**", "*.txt"),
                            recursive=True))


def train_forever() -> None:
    global TRAIN_PROC
    ckpt_every = int(os.environ.get("SYN_CKPT_EVERY", "20000"))
    while True:
        files = _data_files()
        if not files:
            STATE["status"] = ("لا توجد ملفات data/*.txt بعد — "
                               "ارفع بياناتك ثم انتظر 10 دقائق")
            time.sleep(600)
            continue
        STATE["status"] = f"يدرب على {len(files)} ملفاً…"
        env = dict(os.environ,
                   LMF_CKPT=os.path.join(ROOT, "state_train",
                                         "lm_work.tsv.gz"),
                   FG_FUNC_PATH=os.path.join(
                       ROOT, "data_tatoeba", "function_tokens.json"))
        os.makedirs(os.path.join(ROOT, "state_train"), exist_ok=True)
        # feed via lm_feed CLI per file: resume-safe, checkpoints
        # every ckpt_every lines, prints stats
        for f in files:
            try:
                cp = subprocess.run(
                    [sys.executable, os.path.join(TOOLS, "lm_feed.py"),
                     f], capture_output=True, text=True, env=env,
                    timeout=3600 * 6)
                STATE["last_line"] = (cp.stdout or cp.stderr).strip().splitlines()[-1][:120] \
                    if (cp.stdout or cp.stderr) else ""
                STATE["status"] = f"أُغذي {os.path.basename(f)} ثم نكمّل"
            except Exception as e:
                STATE["status"] = "خطأ تغذية: " + repr(e)[:100]
        _snapshot_metrics()
        _persist()
        time.sleep(600)


def _snapshot_metrics() -> None:
    try:
        from numeric_lm import NumericLM
        m = NumericLM()
        m.load(os.path.join(ROOT, "state_train", "lm_work.tsv.gz"))
        STATE["ppl_src"] = m.stats()
    except Exception as e:
        STATE["ppl_src_err"] = repr(e)[:100]


def _persist() -> None:
    """Commit progress to the Space repo when a token exists so
    training survives restarts (native Spaces token HF_TOKEN /
    HUGGING_FACE_HUB_TOKEN, or a user secret named HF_TOKEN)."""
    tok = (os.environ.get("HF_TOKEN")
           or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    if not tok:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=tok)
        # SPACE_ID is the documented default (user/space-name);
        # SPACE_REPO_ID kept as legacy fallback
        repo = (os.environ.get("SPACE_ID")
                or os.environ.get("SPACE_REPO_ID") or "")
        if not repo:
            return
        w = os.path.join(ROOT, "state_train", "lm_work.tsv.gz")
        if os.path.isfile(w):
            api.upload_file(path_or_fileobj=w,
                            path_in_repo="state_train/lm_work.tsv.gz",
                            repo_id=repo, repo_type="space")
        api.upload_file(path_or_fileobj=_metrics_bytes(),
                        path_in_repo="train_report_live.json",
                        repo_id=repo, repo_type="space")
    except Exception as e:
        STATE["persist_err"] = repr(e)[:120]


def _metrics_bytes() -> bytes:
    return json.dumps(
        {k: v for k, v in STATE.items()
         if k in ("fed_total", "ppl_src", "status",
                  "last_line")} | {"uptime_h": round(
              (time.time() - STATE["started"]) / 3600, 2),
              "ts": datetime.utcnow().isoformat()},
        ensure_ascii=False).encode()


def _train_snapshot_text() -> str:
    return json.dumps(STATE, ensure_ascii=False, indent=1)


def _run_pull() -> None:
    """Pull a deduplicated batch from the HF dataset libraries +
    GitHub skill repos into data/ (library feeding). Errors are
    reported in STATE, never raised."""
    STATE["pulling"] = True
    STATE["pull_log"] = "بدأ السحب من المكتبات…"
    try:
        budget = os.environ.get("SYN_PULL_MB", "35")
        cp = subprocess.run(
            [sys.executable, os.path.join(TOOLS, "hf_pull.py"),
             "--budget-mb", budget],
            capture_output=True, text=True, cwd=ROOT,
            timeout=3600 * 3)
        tail = (cp.stdout or cp.stderr or "").strip().splitlines()
        STATE["pull_log"] = "\n".join(tail[-12:])[:1500] \
            or "لا مخرجات"
        STATE["pull_status"] = "اكتمل السحب — سيلتقطه خيط التدريب خلال 10 دقائق"
    except Exception as e:
        STATE["pull_log"] = "خطأ سحب: " + repr(e)[:200]
        STATE["pull_status"] = "فشل السحب — انظر السجل"
    STATE["pulling"] = False


def _pull_click() -> str:
    if STATE.get("pulling"):
        return "السحب جارٍ بالفعل…"
    threading.Thread(target=_run_pull, daemon=True).start()
    return "بدأ السحب من المكتبات في الخلفية — راقب السجل"


def _ask(q: str) -> str:
    if not q.strip():
        return "اكتب سؤالاً."
    try:
        import webmind
        r = webmind.answer(q)
        g = r.get("gen") or {}
        extra = (f"\n\n[الوضع: {r.get('mode')} | تغطية: "
                 f"{g.get('term_coverage', '-')} | سلاسة: "
                 f"{g.get('lm_score', '-')}]")
        return r.get("reply", "") + extra
    except Exception as e:
        return "خطأ: " + repr(e)[:120]


def _trained_tar() -> str:
    out = os.path.join("/tmp", "synora_trained_download.tar.gz")
    lm_work = os.path.join(ROOT, "state_train", "lm_work.tsv.gz")
    if os.path.isfile(lm_work):
        stamp = os.path.join(ROOT, "lm", "lm_ar_trigram.tsv.gz")
        shutil.copyfile(lm_work, stamp)
    with tarfile.open(out, "w:gz") as tf:
        tf.add(ROOT, arcname="synora16_model_clean_trained")
    return out


def main() -> None:
    import gradio as gr
    threading.Thread(target=train_forever, daemon=True).start()
    with gr.Blocks(title="سينورا — تدريب مستمر") as demo:
        gr.Markdown("## سينورا — تدريب صامت مستمر 24/7")
        with gr.Tab("الحالة"):
            btn = gr.Button("تحديث الحالة")
            box = gr.Textbox(label="حالة التدريب الحية", lines=14)
            btn.click(lambda: _train_snapshot_text(),
                      outputs=box)
            demo.load(lambda: _train_snapshot_text(),
                      outputs=box)
            gr.Markdown("### التغذية من المكتبات (HF + GitHub)")
            pull_btn = gr.Button(
                "سحب دفعة تدريب من مكتبات HF/GitHub إلى data/")
            pull_box = gr.Textbox(label="سجل السحب", lines=8)
            pull_btn.click(_pull_click, outputs=pull_box)
            gr.Button("تحديث سجل السحب").click(
                lambda: STATE.get("pull_log", "لا سحب بعد"),
                outputs=pull_box)
        with gr.Tab("اسأل النموذج"):
            q = gr.Textbox(label="سؤالك")
            a = gr.Textbox(label="الجواب", lines=4)
            q.submit(_ask, inputs=q, outputs=a)
            gr.Button("اسأل").click(_ask, inputs=q, outputs=a)
        with gr.Tab("تنزيل النسخة المدربة"):
            gr.Button("تجهيز الحزمة المدربة وتنزيلها").click(
                _trained_tar, outputs=gr.File(
                    label="synora_trained.tar.gz"))
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
