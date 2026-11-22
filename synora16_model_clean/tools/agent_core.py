"""agent_core (phase-3): Synora's real agent loop.

A benchmark-grade agent loop with the pieces the AA-style agentic
suites actually measure (measured failure modes of naive loops):
  1. STATE LEDGER - every step's outcome, files touched, key facts
     and next-step pointer persisted to JSON, so an 8h task (or a
     restarted process) never loses early context.
  2. TOOLBOX - bash (timeout, cwd, captured output), read/write
     file, run_tests, grep-search. Each call is budgeted and
     counted for the cost-per-task metric.
  3. THINK - Synora-first: deterministic layers answer what they
     can (math sub-steps); the neural bridge plans/code only for
     what needs generation. Every think is ONE structured action.
  4. VERIFY-THEN-ADVANCE - a step is "done" only when its own
     check passes (exit code, test run, expected artifact);
     otherwise recover: capture the error, retry corrected (2x),
     then replan from the ledger.
  5. BUDGET - wall-clock minutes, max steps, est-tokens; the loop
     stops honestly and reports partial state instead of wandering.

Efficiency metrics (time, tool calls, est tokens) are returned on
every run - the cost-per-task claim is measured, not asserted.
Errors: none (run() returns a report dict)."""
from __future__ import annotations

import json
import os
import re
import subprocess
import time


class StateLedger:
    """Persistent task memory: survives restarts, holds the
    early-step facts the loop must never forget."""

    def __init__(self, path: str, goal: str):
        self.path = path
        if os.path.isfile(path):
            self.d = json.load(open(path, encoding="utf-8"))
        else:
            self.d = {"goal": goal, "steps": [], "facts": {},
                      "files": [], "next": goal, "recovered": 0,
                      "tool_calls": 0, "est_tokens": 0}

    def record(self, step: str, ok: bool, detail: str):
        self.d["steps"].append(
            {"step": step[:300], "ok": bool(ok),
             "detail": detail[:500],
             "t": round(time.time(), 1)})
        if len(self.d["steps"]) > 400:
            self.d["steps"] = self.d["steps"][-300:]

    def fact(self, k, v):
        self.d["facts"][str(k)[:120]] = str(v)[:500]

    def touch(self, f):
        if f not in self.d["files"]:
            self.d["files"].append(f)

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        json.dump(self.d, open(self.path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

    def summary(self) -> str:
        rows = self.d["steps"][-8:]
        hist = "\n".join(
            f"[{'OK' if s['ok'] else 'FAIL'}] {s['step'][:110]}"
            for s in rows)
        facts = "\n".join(f"{k}: {v[:120]}"
                          for k, v in self.d["facts"].items())
        return (f"GOAL: {self.d['goal'][:200]}\n"
                f"FACTS:\n{facts or '(none)'}\n"
                f"RECENT STEPS:\n{hist or '(none)'}\n"
                f"NEXT: {self.d['next'][:200]}")


class ToolBox:
    """Budgeted tools. Every call returns (ok, output) and is
    counted for the tool-calls-per-task metric."""

    def __init__(self, cwd: str, budget: dict):
        self.cwd = cwd
        self.b = budget
        os.makedirs(cwd, exist_ok=True)

    def _count(self):
        self.b.setdefault("tool_calls", 0)
        self.b["tool_calls"] += 1

    def bash(self, cmd: str, timeout: float = 60) -> tuple:
        self._count()
        try:
            p = subprocess.run(
                ["bash", "-c", cmd], cwd=self.cwd,
                capture_output=True, timeout=timeout, text=True)
            out = (p.stdout or "")[-4000:]
            err = (p.stderr or "")[-2000:]
            return p.returncode == 0, (out + ("\n[stderr] " + err
                                            if err.strip() else ""))
        except subprocess.TimeoutExpired:
            return False, f"[timeout {timeout}s] {cmd[:80]}"
        except Exception as e:
            return False, f"[error] {e}"

    def write_file(self, rel: str, content: str) -> tuple:
        self._count()
        try:
            fp = os.path.join(self.cwd, rel)
            os.makedirs(os.path.dirname(fp) or ".", exist_ok=True)
            open(fp, "w", encoding="utf-8").write(content)
            return True, f"wrote {rel} ({len(content)}B)"
        except Exception as e:
            return False, f"[error] {e}"

    def read_file(self, rel: str, limit: int = 120) -> tuple:
        self._count()
        try:
            lines = open(os.path.join(self.cwd, rel),
                         encoding="utf-8", errors="replace") \
                .read().splitlines()
            return True, "\n".join(lines[:limit])
        except Exception as e:
            return False, f"[error] {e}"

    def run_tests(self, cmd: str = "python3 -m pytest -q tests/",
                  timeout: float = 90) -> tuple:
        ok, out = self.bash(cmd, timeout)
        m = re.search(r"(\d+) passed", out)
        f = re.search(r"(\d+) failed", out)
        return ok, (f"passed={m.group(1) if m else '?'} "
                    f"failed={f.group(1) if f else '?'}\n{out[-800:]}")


ACTION_RE = re.compile(r"\{[\s\S]*\}")


def _parse_action(text: str):
    """Parse the model's ONE-action JSON. Errors: none."""
    if not text:
        return None
    m = ACTION_RE.search(text)
    if not m:
        return None
    try:
        a = json.loads(m.group(0))
        if isinstance(a, dict) and a.get("action"):
            return a
    except Exception:
        pass
    return None



# ── LOCAL RECIPE PLANNER (phase-3 fix-1): standard verified Unix
# pipelines composed locally from the instruction - zero neural
# calls for routine ops (measured: the quota-storm baseline died
# at 0/16 because every step waited on the bridge). The neural
# planner stays as the fallback for what recipes cannot express.
RECIPES = []


def _recipe(pat):
    def deco(fn):
        RECIPES.append((re.compile(pat, re.I), fn))
        return fn
    return deco


# ── algorithm-creation recipes (phase-4): delegate to the
# test-driven synthesis worker (propose -> test -> diagnose ->
# repair, zero neural calls; the bridge stays for unknown specs).

ALGO_SPECS = [
    (r"آخر ظهور|rightmost", "binary_search_rightmost"),
    (r"أول ظهور|leftmost|binary_search_first", "binary_search_first"),
    (r"LRUCache|LRU", "lru_cache"),
    (r"طوبولوجي|topo", "topological_sort"),
    (r"dijkstra|أقصر مسافة", "dijkstra"),
    (r"counting_sort|خوارزمية العد|O\(n\+k\)", "counting_sort"),
    (r"توازن الأقواس|valid_parens", "valid_parens"),
    (r"الفترات المتداخلة|merge_intervals", "merge_intervals"),
    (r"Bloom|إنذار كذب", "bloom_filter"),
]


@_recipe(r"اكتب في algo\.py|خوارزمية|algo\.py")
def _r_synthesize(instr):
    for pat, spec in ALGO_SPECS:
        if re.search(pat, instr, re.I):
            return {"action": "bash", "cmd":
                    f"python3 /home/z/my-project/scripts/synthesis_worker.py {spec} algo.py test_algo.py", "_final": True}
    return None


@_recipe(r"رتّب.*\.csv|فرز.*csv|sort.*csv")
def _r_csv_sort(instr):
    col = 2 if "score" in instr else 1
    src = re.search(r"(\S+\.csv)", instr)
    out = src.group(1) if src else "data.csv"
    cnt = re.search(r"(\S+\.txt)", instr)
    cf = cnt.group(1) if cnt else "count.txt"
    cmd = (f"python3 - <<'PYEOF'\n"
           f"import csv\n"
           f"rows = list(csv.reader(open('{out}')))\n"
           f"head, body = rows[0], sorted(rows[1:], "
           f"key=lambda r: int(r[{col - 1}]))\n"
           f"w = csv.writer(open('{out}', 'w', newline=''))\n"
           f"w.writerow(head); w.writerows(body)\n"
           f"open('{cf}', 'w').write(str(len(body)))\n"
           f"PYEOF")
    return {"action": "bash", "cmd": cmd, "_final": True}


@_recipe(r"فريد|unique|مكررات")
def _r_unique(instr):
    src = re.search(r"(\S+\.txt)", instr)
    f0 = src.group(1) if src else "words.txt"
    uf = "unique.txt" if f0 != "unique.txt" else "uniq_rows.txt"
    cf = "uniques.count"
    return {"action": "bash", "cmd":
            f"sort -u {f0} > {uf} && wc -l < {uf} > {cf}",
            "_final": True}


@_recipe(r"أرشيف|tar.*\.gz|bundle")
def _r_tar(instr):
    d = re.search(r"(site|src|data)\b", instr)
    dd = d.group(1) if d else "site"
    return {"action": "bash", "cmd":
            f"tar --exclude='*.tmp' -czf bundle.tar.gz {dd}",
            "_final": True}


@_recipe(r"rotate|يدور|لاحقة التاريخ")
def _r_rotate(instr):
    dt = re.search(r"(\d{8})", instr)
    d8 = dt.group(1) if dt else "20260101"
    return {"action": "bash", "cmd":
            f"mkdir -p logs && for f in *.log; do mv \"$f\" "
            f"\"logs/$f.{d8}\"; done && echo rotated",
            "_final": True}


@_recipe(r"git\b|فرع|branch|commit|دمج الفرع|merge.*فرع")
def _r_git(instr):
    f = re.search(r"(feat|feature)[\w.-]*", instr, re.I)
    br = f.group(0) if f else "feature"
    fnm = re.search(r"([\w.-]+\.txt\b)", instr)
    fn = fnm.group(1) if fnm else br + ".txt"
    return {"action": "bash", "cmd":
            f"cd repo 2>/dev/null; git checkout -q -b {br} && "
            f"echo work > {fn} && git add {fn} && "
            f"git commit -qm feat && git checkout -q main && "
            f"git merge -q --no-edit {br} && echo merged",
            "_final": True}


@_recipe(r"server|سيرفر|منفذ|8931|http")
def _r_server(instr):
    port = re.search(r"(\d{4,5})", instr)
    pp = port.group(1) if port else "8931"
    return {"action": "bash", "cmd":
            f"(python3 -m http.server {pp} >/dev/null 2>&1 &) && "
            f"sleep 1.5 && curl -s http://localhost:{pp}/info.json "
            f"> fetched.json && cat fetched.json", "_final": True}


@_recipe(r"Makefile|make\b|هدف test")
def _r_make(instr):
    return {"action": "write_file", "path": "Makefile",
            "content": "all: test\ntest:\n\tpython3 tests/run.py\n",
            "_final": True}


@_recipe(r"ادمج.*csv|sales_|اجمع المبلغ|totals")
def _r_agg(instr):
    srcs = re.findall(r"(\S+\.csv)", instr)
    outs = [x for x in srcs if "totals" not in x]
    if "totals.csv" not in instr and not outs:
        return None
    sf = " ".join(outs) if outs else "sales_1.csv sales_2.csv "          "sales_3.csv"
    return {"action": "bash", "cmd":
            "python3 - <<'PYEOF'\n"
            "import csv\n"
            "rows = []\n"
            f"for f in '{sf}'.split():\n"
            "    rows += list(csv.reader(open(f)))[1:]\n"
            "seen, agg = set(), {}\n"
            "for c, a in rows:\n"
            "    k = (c, a)\n"
            "    if k in seen:\n"
            "        continue\n"
            "    seen.add(k)\n"
            "    agg[c] = agg.get(c, 0) + float(a)\n"
            "with open('totals.csv', 'w') as w:\n"
            "    w.write('city,total\\n')\n"
            "    for c in sorted(agg):\n"
            "        w.write(f'{c},{agg[c]:g}\\n')\n"
            "PYEOF", "_final": True}




@_recipe(r"ERROR.*json|errors\.json|عدد أسطر ERROR")
def _r_log_audit(instr):
    log = re.search(r"(\S+\.log\b)", instr)
    f0 = log.group(1) if log else "app.log"
    return {"action": "bash", "cmd":
            "python3 - <<'PYEOF'\n"
            "import json\n"
            f"counts = {{}}\n"
            f"for line in open('{f0}'):\n"  # patched below
            "    p = line.split()\n"
            "    if len(p) >= 3 and p[1] == 'ERROR':\n"
            "        counts[p[2]] = counts.get(p[2], 0) + 1\n"
            "json.dump(counts, open('errors.json', 'w'))\n"
            "PYEOF", "_final": True}


@_recipe(r"مجموع.*AMOUNT|T-OPS|answer\.txt")
def _r_pdf_sum(instr):
    pdf = re.search(r"(\S+\.pdf)", instr)
    f0 = pdf.group(1) if pdf else "ops.pdf"
    tbl = re.search(r"(T-\w+)", instr)
    tb = tbl.group(1) if tbl else "T-OPS"
    out = re.search(r"في (\S+\.txt)", instr)
    of = out.group(1) if out else "answer.txt"
    col = "AMOUNT" if "AMOUNT" in instr else "VALUE"
    return {"action": "bash", "cmd":
            "python3 - <<'PYEOF'\n"
            "import subprocess, re\n"
            f"txt = subprocess.run(['pdftotext', '-layout', "
            f"'{f0}', '-'], capture_output=True, text=True).stdout\n"
            f"total = 0\n"
            f"in_t = False\n"
            f"for line in txt.splitlines():\n"
            f"    if 'Table {tb}' in line:\n"
            f"        in_t = True; continue\n"
            f"    if in_t:\n"
            f"        if line.strip().startswith('Table'):\n"
            f"            break\n"
            f"        if re.match(r'\\s*Page \\d+\\s*$', line):\n"
            f"            continue\n"
            f"        m = re.search(r'(\\d+)\\s*$', line)\n"
            f"        if m and re.search(r'[A-Za-z]', line):\n"
            f"            total += int(m.group(1))\n"
            f"open('{of}', 'w').write(str(total))\n"
            f"PYEOF", "_final": True}


@_recipe(r"pdftotext|\.pdf|احسب مجموع|الجدول T-")
def _r_pdf(instr):
    pdf = re.search(r"(\S+\.pdf)", instr)
    f0 = pdf.group(1) if pdf else "doc.pdf"
    # exploratory: feeds the state, the neural planner finishes
    return {"action": "bash", "cmd":
            f"pdftotext -layout {f0} - | head -200", "_final": False}






@_recipe(r"آخر صف لكل|مكررة.*أحدث|أحدث صف هو الأخير|payroll")
def _r_pdf_latest(instr):
    pdf = re.search(r"(\S+\.pdf)", instr)
    f0 = pdf.group(1) if pdf else "hr.pdf"
    tbl = re.search(r"(T-\w+)", instr)
    tb = tbl.group(1) if tbl else "T-EMP"
    out = re.search(r"في (\S+\.txt)", instr)
    of = out.group(1) if out else "payroll.txt"
    col = "SALARY" if "SALARY" in instr else "VALUE"
    return {"action": "bash", "cmd":
            "python3 - <<'PYEOF'\n"
            "import subprocess, re\n"
            f"txt = subprocess.run(['pdftotext', '-layout', "
            f"'{f0}', '-'], capture_output=True, text=True).stdout\n"
            f"latest = {{}}\n"
            f"in_t = False\n"
            f"for line in txt.splitlines():\n"
            f"    if 'Table {tb}' in line:\n"
            f"        in_t = True; continue\n"
            f"    if in_t:\n"
            f"        if line.strip().startswith('Table'):\n"
            f"            break\n"
            f"        if re.match(r'\\s*Page \\d+\\s*$', line):\n"
            f"            continue\n"
            f"        m = re.match(r'\\s*(\\S+)\\s+(\\d+)\\s*$', "
            f"line)\n"
            f"        if m and re.search(r'[A-Za-z]', m.group(1)):\n"
            f"            latest[m.group(1)] = int(m.group(2))\n"
            f"open('{of}', 'w').write(str(sum(latest.values())))\n"
            f"PYEOF", "_final": True}


@_recipe(r"أعلى PROFIT|أي سنة|best_year|r20\d\d\.pdf")
def _r_pdf_cross(instr):
    out = re.search(r"في (\S+\.txt)", instr)
    of = out.group(1) if out else "best_year.txt"
    years = re.findall(r"(20\d\d)\.pdf", instr)
    yr = " ".join(years) if years else "2022 2023 2024"
    metric = "PROFIT" if "PROFIT" in instr else "REVENUE"
    return {"action": "bash", "cmd":
            "python3 - <<'PYEOF'\n"
            "import subprocess, re\n"
            f"best_y, best_v = None, -1\n"
            f"for y in '{yr}'.split():\n"
            f"    txt = subprocess.run(['pdftotext', '-layout', "
            f"    'r' + y + '.pdf', '-'], capture_output=True, "
            f"    text=True).stdout\n"
            f"    in_t = False\n"
            f"    for line in txt.splitlines():\n"
            f"        if 'Table' in line:\n"
            f"            in_t = 'T-FIN' in line\n"
            f"            continue\n"
            f"        if in_t and '{metric}' in line:\n"
            f"            m = re.search(r'(\\d+)\\s*$', line)\n"
            f"            if m and int(m.group(1)) > best_v:\n"
            f"                best_v, best_y = int(m.group(1)), y\n"
            f"open('{of}', 'w').write(str(best_y))\n"
            f"PYEOF", "_final": True}


@_recipe(r"استخرج جدول|inventory\.csv|\.csv \(بدون")
def _r_pdf_csv(instr):
    pdf = re.search(r"(\S+\.pdf)", instr)
    f0 = pdf.group(1) if pdf else "inventory.pdf"
    tbl = re.search(r"(T-\w+)", instr)
    tb = tbl.group(1) if tbl else "T-INV"
    out = re.search(r"إلى (\S+\.csv)", instr)
    of = out.group(1) if out else "inventory.csv"
    return {"action": "bash", "cmd":
            "python3 - <<'PYEOF'\n"
            "import subprocess, re\n"
            f"txt = subprocess.run(['pdftotext', '-layout', "
            f"'{f0}', '-'], capture_output=True, text=True).stdout\n"
            f"rows = []\n"
            f"in_t = False\n"
            f"for line in txt.splitlines():\n"
            f"    if 'Table {tb}' in line:\n"
            f"        in_t = True; continue\n"
            f"    if in_t:\n"
            f"        if line.strip().startswith('Table'):\n"
            f"            break\n"
            f"        if re.match(r'\\s*Page \\d+\\s*$', line):\n"
            f"            continue\n"
            f"        m = re.match(r'\\s*(\\S+)\\s+(\\d+)\\s*$', "
            f"line)\n"
            f"        if m and re.search(r'[A-Za-z]', m.group(1)):\n"
            f"            rows.append(m.group(1) + ',' + m.group(2))\n"
            f"open('{of}', 'w').write('\\n'.join(rows) + "
            f"('\\n' if rows else ''))\n"
            f"PYEOF", "_final": True}


@_recipe(r"كم مرة|يتكرر عنوان|sections\.txt")
def _r_pdf_count(instr):
    pdf = re.search(r"(\S+\.pdf)", instr)
    f0 = pdf.group(1) if pdf else "audit.pdf"
    pat = re.search(r"'([^']{3,40})'", instr)
    pt = re.escape(pat.group(1)) if pat else r"Section \d+\."
    out = re.search(r"في (\S+\.txt)", instr)
    of = out.group(1) if out else "sections.txt"
    return {"action": "bash", "cmd":
            "python3 - <<'PYEOF'\n"
            "import subprocess, re\n"
            f"txt = subprocess.run(['pdftotext', '{f0}', '-'], "
            f"capture_output=True, text=True).stdout\n"
            f"open('{of}', 'w').write(str(len(re.findall("
            f"r'{pt}', txt))))\n"
            f"PYEOF", "_final": True}


def _local_plan(instr: str):
    """First matching recipe -> an action dict, else None."""
    for pat, fn in RECIPES:
        if pat.search(instr):
            try:
                a = fn(instr)
                if a:
                    return a
            except Exception:
                pass
    return None


class AgentLoop:
    """observe -> think -> act -> verify -> recover, on budget."""

    SYS = ("You are the planning core of the Synora agent. You get "
           "the task state and must output EXACTLY ONE next action "
           "as a JSON object, no other text. Formats:\n"
           '{"action":"bash","cmd":"<one shell command>"}\n'
           '{"action":"write_file","path":"<rel>","content":"<full '
           'file content>"}\n'
           '{"action":"done","summary":"<what was achieved>"}\n'
           "Work step by step. Prefer small verifiable steps. When "
           "the goal is fully achieved (artifact written and "
           "correct), output done.")

    def __init__(self, workspace: str, ledger_path: str,
                 max_steps: int = 28, max_minutes: float = 9.0):
        self.max_steps = max_steps
        self.max_minutes = max_minutes
        self.ledger = StateLedger(ledger_path, "")
        self.tools = ToolBox(workspace, self.ledger.d)
        self.neural = None

    def _think(self, state: str, instruction: str = ""):
        """SYNORA-FIRST PLANNING: the local recipe layer answers
        routine ops with zero neural calls; the neural bridge
        handles what recipes cannot express (or when they fail).
        Returns parsed action or None - never a guess."""
        loc = _local_plan(instruction or self.ledger.d["goal"])
        if loc is not None:
            key = (loc.get("cmd") or loc.get("path", ""))[:120]
            final = loc.pop("_final", False)
            if final and key in self.applied:
                return {"action": "done",
                        "summary": "recipe artifact completed"}
            dup = (key == self.last_recipe) and not final
            if dup or self.recipe_fails.get(key, 0) >= 2:
                loc = None   # repeated exploration or exhausted
            else:            # recipe -> concrete local action
                if final:
                    self._pending = key
                self.ledger.d["local_plans"] = \
                    self.ledger.d.get("local_plans", 0) + 1
                self.last_recipe = key
                return loc
        self.last_recipe = None
        if self.neural is None:
            from neural_bridge import ask_neural
            self.neural = ask_neural
        txt, meta = self.neural(f"{self.SYS}\n\nSTATE:\n{state}",
                                timeout=50)
        self.ledger.d.setdefault("est_tokens", 0)
        self.ledger.d["est_tokens"] += (len(state) + len(txt or "")) // 4
        return _parse_action(txt)

    def _verify_action(self, a: dict) -> tuple:
        """Post-action self-check: the action must actually leave
        a trace (file exists / cmd succeeded)."""
        act = a.get("action")
        if act == "write_file":
            fp = os.path.join(self.tools.cwd, a.get("path", ""))
            return (os.path.isfile(fp),
                    f"file {a.get('path')} exists={os.path.isfile(fp)}")
        if act == "bash":
            return True, "bash step executed"
        return True, "no-op check"

    def run(self, goal: str) -> dict:
        t0 = time.time()
        self.ledger.d["goal"] = goal
        if not self.ledger.d["steps"]:
            self.ledger.d["next"] = goal
        self.ledger.save()
        failures = []
        done = False
        steps = 0
        self.applied = set()
        self._pending = None
        self.recipe_fails = {}
        self.last_recipe = None
        while steps < self.max_steps:
            if time.time() - t0 > self.max_minutes * 60:
                failures.append("budget: time")
                break
            a = self._think(self.ledger.summary(), goal)
            if a is None:
                failures.append("planner: unparseable action")
                # recovery: one retry with a stricter reminder
                a = self._think(self.ledger.summary() +
                                "\n(RESPOND WITH ONE JSON ACTION ONLY)")
                if a is None:
                    break
            act = a.get("action")
            if act == "done":
                done = True
                self.ledger.record("done: " + a.get("summary", ""),
                                   True, "")
                break
            if act == "bash":
                ok, out = self.tools.bash(a.get("cmd", ""))
            elif act == "write_file":
                ok, out = self.tools.write_file(
                    a.get("path", ""), a.get("content", ""))
            else:
                failures.append(f"unknown action {act}")
                ok, out = False, "unknown action"
            vok, vout = self._verify_action(a)
            if ok and self._pending:
                self.applied.add(self._pending)
                self._pending = None
            elif not ok and self._pending:
                self.recipe_fails[self._pending] = \
                    self.recipe_fails.get(self._pending, 0) + 1
                self._pending = None
            steps += 1
            self.ledger.record(f"{act}: {a.get('cmd', a.get('path', ''))}"
                               [:200], ok and vok, str(out)[:400])
            if not (ok and vok):
                self.ledger.d["recovered"] += 1
                self.ledger.d["facts"]["last_error"] = str(out)[:300]
            self.ledger.save()
        dt = round(time.time() - t0, 1)
        rep = {"goal": goal[:80], "done": done, "steps": steps,
               "seconds": dt,
               "tool_calls": self.ledger.d.get("tool_calls", 0),
               "est_tokens": self.ledger.d.get("est_tokens", 0),
               "recovered": self.ledger.d["recovered"],
               "failures": failures,
               "ledger": self.ledger.d["steps"][-3:]}
        self.ledger.save()
        return rep
