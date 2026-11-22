"""hf_pull - pull a training batch directly from HF datasets and
GitHub skill repos into data/ (the "library feeding" path: no
manual file uploads needed).

Inside a Hugging Face Space: works with the runtime token for
public datasets; requires the `datasets` package (add it to
requirements.txt). Anonymous works for public datasets too.

CLI:
  python3 hf_pull.py [--budget-mb 35] [--only name1,name2]

Writes data/lib_<name>.txt (globally deduplicated), state_train/
lib_dedup.set, state_train/lib_manifest.json. Errors: per-source
try/except -> manifest, never raises."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
STATE = os.path.join(ROOT, "state_train")
os.makedirs(DATA, exist_ok=True)
os.makedirs(STATE, exist_ok=True)
WS = re.compile(r"\s+")

MSG_KEYS = ("content", "text")
SKIP_KEYS = ("uuid", "id", "hash", "date", "url", "license",
             "image", "emoji", "emojis")


def walk_texts(obj, depth=0):
    if depth > 6:
        return
    if isinstance(obj, str):
        if len(obj) >= 3 and "base64," not in obj[:80]:
            yield obj
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in MSG_KEYS:
                yield from walk_texts(v, depth + 1)
            elif k in SKIP_KEYS:
                continue
            else:
                yield from walk_texts(v, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from walk_texts(v, depth + 1)


def ex_messages(row):
    for k in ("system", "messages", "initial_messages",
              "trajectory", "query", "patch"):
        yield from walk_texts(row.get(k))


def ex_fields(*names):
    def f(row):
        for n in names:
            yield from walk_texts(row.get(n))
    return f


CURRICULUM = [
    # (name, hf repo, config, split, extractor, budget_mb)
    ("denovoswe_trj_filtered",
     "AweAI-Team/DeNovoSWE-Trajectory-Filtered", None, "train",
     ex_messages, 3),
    ("denovoswe_trj_raw", "AweAI-Team/DeNovoSWE-Trajectory-Raw",
     None, "train", ex_messages, 2),
    ("denovoswe_repos", "AweAI-Team/DeNovoSWE", None, "train",
     ex_fields("readme_content", "repo_structure", "document"), 2),
    ("nemotron_swe_v1", "nvidia/Nemotron-SWE-v1", None, "r2e_gym",
     ex_messages, 2),
    ("nemotron_sft_swe_v3", "nvidia/Nemotron-SFT-SWE-v3",
     None, "train", ex_messages, 3),
    ("terminal_traj", "m-a-p/TerminalTraj", None, "train",
     ex_messages, 2),
    ("opencode_instruct", "nvidia/OpenCodeInstruct", None,
     "train", ex_fields("input", "output"), 3),
    ("glaive_code", "glaiveai/glaive-code-assistant", None,
     "train", ex_fields("question", "answer"), 3),
    ("codeparrot_clean", "codeparrot/codeparrot-clean", None,
     "train", ex_fields("content"), 3),
    ("oasst2", "OpenAssistant/oasst2", None, "train",
     ex_fields("text"), 3),
    ("marketing_13k", "suehuynh/marketing-instruct-13k", None,
     "train", ex_fields("enhanced_prompt", "enhanced_completion"),
     2),
    ("fineweb_sample", "HuggingFaceFW/fineweb", "sample-10BT",
     "train", ex_fields("text"), 5),
    ("fineinstructions",
     "fineinstructions/fineinstructions_nemotron", None, "train",
     ex_fields("instantiated_instruction", "answer", "text"), 2),
]
GH_REPOS = [
    ("gh_stitch_landing", "cskwork/stitch-landing-skill"),
    ("gh_web_design", "jseanj/web-design-skill"),
    ("gh_premium_saas", "MiddleSugar7000/premium-saas-web-skill"),
]
DEDUP = os.path.join(STATE, "lib_dedup.set")


def load_seen() -> set:
    if os.path.isfile(DEDUP):
        with open(DEDUP, "rb") as f:
            return set(f.read().split(b"\n")[:-1])
    return set()


def save_seen(seen: set) -> None:
    tmp = DEDUP + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b"\n".join(sorted(seen)) + b"\n")
    os.replace(tmp, DEDUP)


def pull_github(seen: set, stats: dict) -> None:
    out_path = os.path.join(DATA, "lib_gh_skills.txt")
    added = 0
    with open(out_path, "a", encoding="utf-8") as out:
        for name, slug in GH_REPOS:
            blob = None
            for br in ("main", "master"):
                url = (f"https://codeload.github.com/{slug}/"
                       f"tar.gz/refs/heads/{br}")
                try:
                    with urllib.request.urlopen(url,
                                                timeout=60) as r:
                        blob = r.read()
                    break
                except Exception:
                    continue
            if blob is None:
                stats.setdefault("errors", {})[name] = \
                    f"{slug} unreachable"
                continue
            with tarfile.open(fileobj=io.BytesIO(blob)) as t:
                for m in t.getmembers():
                    if not (m.name.endswith(".md")
                            or m.name.endswith(".txt")):
                        continue
                    if m.size > 300_000 or m.size < 10:
                        continue
                    try:
                        txt = t.extractfile(m).read().decode(
                            "utf-8", errors="replace")
                    except Exception:
                        continue
                    for line in txt.splitlines():
                        line = WS.sub(" ", line).strip()
                        if len(line) < 3 or len(line) > 4000:
                            continue
                        h = hashlib.sha1(
                            line.encode("utf-8")).digest()
                        if h in seen:
                            continue
                        seen.add(h)
                        out.write(line + "\n")
                        added += 1
    stats["gh_skills"] = {"unique_lines": added,
                          "status": "ok"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-mb", type=int, default=35)
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    try:
        from datasets import load_dataset
    except Exception as e:
        print("HINT: add 'datasets' to requirements.txt "
              "(Space rebuild needed).", repr(e)[:120],
              flush=True)
        sys.exit(0)
    seen = load_seen()
    stats = {"invocation_budget_mb": args.budget_mb,
             "sources": {}, "errors": {}}
    spent = 0.0
    for name, repo, cfg, split, ex, budget in CURRICULUM:
        if only and name not in only:
            continue
        if spent >= args.budget_mb:
            break
        st = {"repo": repo, "budget_mb": budget, "rows": 0,
              "unique_lines": 0, "dup_skipped": 0, "mb": 0.0,
              "status": "ok", "error": ""}
        t0 = time.time()
        try:
            ds = load_dataset(repo, cfg, split=split,
                              streaming=True)
            with open(os.path.join(DATA, f"lib_{name}.txt"), "a",
                      encoding="utf-8") as out:
                for row in ds:
                    st["rows"] += 1
                    if time.time() - t0 > 300:
                        st["status"] = "time_cap"
                        break
                    for txt in ex(row):
                        for line in txt.splitlines():
                            line = WS.sub(" ", line).strip()
                            if len(line) < 3 or len(line) > 4000:
                                continue
                            h = hashlib.sha1(
                                line.encode("utf-8")).digest()
                            if h in seen:
                                st["dup_skipped"] += 1
                                continue
                            seen.add(h)
                            out.write(line + "\n")
                            st["unique_lines"] += 1
                            st["mb"] += (len(line) + 1) / 1e6
                        if st["mb"] >= budget:
                            break
                    if st["mb"] >= budget:
                        break
        except Exception as e:
            st["status"] = "error"
            st["error"] = repr(e)[:160]
        spent += st["mb"]
        stats["sources"][name] = st
        print(f"{name}: {st['status']} rows={st['rows']} "
              f"uniq={st['unique_lines']} "
              f"({round(st['mb'], 2)}MB) "
              f"dup={st['dup_skipped']}", flush=True)
        save_seen(seen)
        with open(os.path.join(STATE, "lib_manifest.json"),
                  "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=1)
    if not only:
        pull_github(seen, stats)
        save_seen(seen)
    with open(os.path.join(STATE, "lib_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(f"DONE spent={round(spent, 2)}MB unique_total="
          f"{len(seen)}", flush=True)


if __name__ == "__main__":
    main()
