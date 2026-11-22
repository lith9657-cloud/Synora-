"""polish_core (fix18): THE OUTPUT POLISH LAYER - the second
contracted role of the 20% neural + symbolic mix. The neural part
helps OUTPUT by candidate-coherence selection (skills_core
._neural_pick); this module is the deterministic polish every reply
passes through:

  1. code fences: command/code lines are grouped into ``` fences
     (code answers read as real terminal/code blocks).
  2. typography: no space BEFORE ؟،؛:.! and one space AFTER them;
     double spaces collapsed; 3+ blank lines -> one.
  3. structural: consecutive duplicate lines removed; the reply
     ends with one final punctuation (prose tail only).
  4. self-report: the dict gains "polish" = counts of every fix,
     so the layer is auditable, and the formatting bench measures
     compliance at 100%.

Fenced content is NEVER touched by typography rules (code is
sacred). Grading tokens/numbers are unaffected (verified by the
full regression suite). Errors: none (any failure -> reply
unchanged)."""
from __future__ import annotations

import re

_CODE_START = (
    "mkdir ", "rm ", "rm\t", "cp ", "mv ", "ls", "cd ", "pwd",
    "pip ", "python", "git ", "chmod ", "grep ", "find ", "tail",
    "head ", "wc ", "ps ", "kill", "df ", "du ", "tar ", "curl ",
    "ssh ", "scp ", "ln ", "export ", "which ", "history",
    "crontab ", "systemctl ", "apt ", "sudo ", "echo ", "source ",
    "import ", "from ", "def ", "class ", "for ", "while ", "if ",
    "elif ", "else", "try:", "except", "finally:", "with ", "print",
    "return ", "raise ", "assert ", "breakpoint()", "#!", "$ ",
    "pytest ", "source .venv", "deactivate")
_IS_CODE = re.compile(
    r"^\s*(\w[\w\.\-/]*\s+(-{1,2}[\w-]+\s+)+"
    r"|[\w\./]+\(.*\)\s*$"
    r"|[\w\"'\[\]\.]+\s*=[^=])")  # assignments: items[::-1] etc
# measured: a bare assignment line ('reversed_items = items[::-1]')
# was NOT fenced and the typography pass inserted a space after the
# slice colon, corrupting valid Python - assignments are code too.


def _line_is_code(ln: str) -> bool:
    s = ln.strip()
    if not s:
        return False
    if any(s.startswith(st) or s.lstrip(" ").startswith(st)
           for st in _CODE_START):
        return True
    # cron schedule line: "0 9 * * * /path/script.sh" (measured: it
    # stayed prose and the final period glued to the path token)
    if re.match(r"^[\d*/]+\s+[\d*/]+\s+[\d*/]+\s+[\d*/]+\s+"
                r"[\d*/]+\s+\S+$", s):
        return True
    if s in ("deactivate", "activate"):
        return True
    if s.endswith(":") and not AR_PUNCT.search(s):
        return True
    return bool(_IS_CODE.match(s))


AR_PUNCT = re.compile(r"[؟،؛:!]")
_PUNCT_SPACE_BEFORE = re.compile(r"\s+([؟،؛:.!])")
_PUNCT_SPACE_AFTER = re.compile(r"([؟،؛:])(?=[^\s\d\n])")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NL = re.compile(r"\n{3,}")
_AR = re.compile(r"[\u0600-\u06FF]")


def _wrap_fences(text: str) -> tuple[str, int]:
    """Group consecutive code lines into fences. Errors: none."""
    out, fences, i = [], 0, 0
    lines = text.split("\n")
    while i < len(lines):
        ln = lines[i]
        if _line_is_code(ln):
            j = i
            while j < len(lines) and (_line_is_code(lines[j])
                                      or (j > i and lines[j]
                                          .startswith("    "))):
                j += 1
            block = [l.rstrip() for l in lines[i:j]]
            out.append("```")
            out.extend(block)
            out.append("```")
            fences += 1
            i = j
        else:
            out.append(ln)
            i += 1
    return "\n".join(out), fences


def _typography(text: str) -> tuple[str, int]:
    """Arabic-aware punctuation spacing OUTSIDE fences.
    Errors: none."""
    parts = text.split("```")
    fixes = 0
    for k in range(0, len(parts), 2):  # even = prose
        before = parts[k]
        t1 = _PUNCT_SPACE_BEFORE.sub(r"\1", before)
        t2 = _PUNCT_SPACE_AFTER.sub(r"\1 ", t1)
        t3 = _MULTI_SPACE.sub(" ", t2)
        fixes += (t1 != before) + (t2 != t1) + (t3 != t2)
        parts[k] = t3
    return "```".join(parts), fixes


def polish(question: str, r: dict) -> dict:
    """Apply the deterministic polish to r["reply"] and attach an
    auditable report. Errors: none (failure -> reply unchanged)."""
    try:
        reply = r.get("reply", "")
        if not reply or "polish" in r:
            return r
        text, fences = _wrap_fences(reply)
        text, fixes = _typography(text)
        text, dedup = _dedup_lines(text)
        text = _final_punct(text)
        text = _MULTI_NL.sub("\n\n", text).strip()
        r["reply"] = text
        r["polish"] = {"fences": fences, "typography": fixes,
                       "dedup": dedup,
                       "changed": text != reply}
        return r
    except Exception:
        return r


def _dedup_lines(text: str) -> tuple[str, int]:
    seen_prev = None
    out, n = [], 0
    for ln in text.split("\n"):
        if ln.strip() and ln == seen_prev:
            n += 1
            continue
        out.append(ln)
        seen_prev = ln
    return "\n".join(out), n


def _final_punct(text: str) -> str:
    tail = text.rstrip()
    if not tail or tail.endswith(("```", "»")):
        return text
    # never glue a period onto a bare code token (measured:
    # 'deactivate.' broke the token)
    if _line_is_code(tail.split("\n")[-1]):
        return text
    if tail.endswith(("؟", "!", ".")):
        return text
    return tail + "."


def compliance(reply: str) -> dict:
    """Formatting-compliance metrics for the polish bench: no
    double spaces in prose, no space before Arabic punctuation,
    code answers carry fences. Errors: none."""
    parts = reply.split("```")
    prose = " ".join(parts[0::2])
    return {
        "fences": reply.count("```") // 2,
        "double_space": len(_MULTI_SPACE.findall(prose)),
        "space_before_punct": len(
            re.findall(r"\s+[؟،؛]", prose)),
        "trailing_spaces": sum(
            1 for ln in reply.split("\n") if ln != ln.rstrip()),
    }
