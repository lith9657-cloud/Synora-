"""paraphrase_core (fix16): NON-NEURAL surface generalization for
Synora. Light Arabic stemmer + prefix/suffix-stem token matching +
IDF weighting over KB entries + equivalence groups + paraphrase
intent rules (identity possessive forms). NO networks, NO vectors:
counting and rules only - the Synora red line.
Data rule: this module adds zero runtime neural computation; it is
pure string/statistics logic like the rest of the counting core."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent

_DIA = re.compile(r"[\u064B-\u0652\u0640]")
_TOK = re.compile(r"[\u0600-\u06FFa-zA-Z0-9]+")
# Arabic-range punctuation (؟ U+061F, ، U+060C, ؛ U+061B, . U+06D4)
# GLUES to tokens ("بناك؟" failed .endswith("ك")); strip it plus
# anything non-word.
_CLEAN = re.compile(
    r"[\u060C\u061B\u061F\u066B\u066C\u06D4]|[^\w\u0600-\u06FF\s]")

# function/question/request words - never discriminative for KB
# (بايثون is NOT stop: it separates python entries from bash ones)
_STOP = set("""ما ماذا من هو هي في على عن كيف هل اين متى لماذا
الذي التي هذا هذه ذلك تلك او ام اي ايهما يا الى مع ثم لكن لكنه
يكون كان كانت معرفه استخدام مستخدم امر امري طريقه طرق اعطني
اخبرني احتاج اريد محتاج ممكن اكتب صمم اقترح نظم صغ احكي اخطر
تخيل جرب اكمل اعد علمني فقط بدون مع لو فضلك بدون حاسب
استخدم اعطنيها""".split())

# small closed equivalence groups (morphological pairs the light
# stemmer cannot bridge)
_EQUIV = [
    ("انشاء", "انشئ", "انشا"),
    ("اعرف", "تعريف", "يعرف", "عرّف"),
    ("حذف", "ازاله", "امسح"),
    ("بحث", "ايجاد", "تتبع"),
    ("فرز", "ترتيب"),
    ("طباعه", "اخراج"),
    # fix18 (measured): dialect noun/verb pairs the light stemmer
    # cannot bridge - 'أستبدل' question vs 'استبدال' anchor failed
    # the shared-stem guard at sim 0.595
    ("استبدال", "استبدل", "استبدال"),
    ("تنسيق", "انسق"),
]
_EQ_SETS = [set(g) for g in _EQUIV]


def stem(w: str) -> str:
    """Light Arabic stemmer: diacritics + hamza norm, one prefix,
    one suffix (conservative: keep >= 3 chars)."""
    w = _DIA.sub("", w)
    w = (w.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
          .replace("ة", "ه").replace("ى", "ي").replace("ئ", "ء"))
    for p in ("بال", "وال", "فال", "كال", "لل", "ال", "و", "ف",
              "ب", "ك", "ل"):
        if w.startswith(p) and len(w) - len(p) >= 3:
            w = w[len(p):]
            break
    for s in ("ات", "ون", "ين", "يه", "ها", "هم", "كم", "هن",
              "ان", "ه", "ي"):
        if w.endswith(s) and len(w) - len(s) >= 3:
            w = w[: len(w) - len(s)]
            break
    return w


def _same_group(a: str, b: str) -> bool:
    for g in _EQ_SETS:
        if a in g and b in g:
            return True
    return False


def tok_match(a: str, b: str, minlen: int = 3) -> bool:
    """Stem-level match: equality, prefix/suffix containment, or
    equivalence group (measured: انشاء~انشئ, اعرف~تعريف)."""
    if a == b:
        return True
    if _same_group(a, b):
        return True
    if len(a) >= minlen and len(b) >= minlen:
        if a.startswith(b) or b.startswith(a):
            return True
        if a.endswith(b) or b.endswith(a):
            return True
    return False


class KBIndex:
    """IDF index over KB entries (q/re/v/obj/keys fields)."""

    def __init__(self, entries: list[dict]):
        self.entries = entries
        self.docs: list[set[str]] = []
        self.df: dict[str, int] = {}
        for e in entries:
            toks: set[str] = set()
            for field in ("q", "re", "v", "obj", "keys"):
                raw = e.get(field)
                if not raw:
                    continue
                for w in _TOK.findall(_CLEAN.sub(" ", str(raw))):
                    st = stem(w)
                    if len(st) >= 2 and st not in _STOP:
                        toks.add(st)
            self.docs.append(toks)
            for t in toks:
                self.df[t] = self.df.get(t, 0) + 1
        self.N = max(len(entries), 1)

    def idf(self, t: str) -> float:
        return 1.0 + math.log(self.N / (1 + self.df.get(t, 0)))

    def score(self, q_stems: list[str]) -> list[float]:
        """Per q-stem: count the BEST matching doc token ONCE
        (measured: مجلد+مجلدا double-counted to 6.24 in ties)."""
        out = []
        for toks in self.docs:
            s = 0.0
            for q in q_stems:
                cand = [self.idf(t) for t in toks
                        if tok_match(t, q)]
                if cand:
                    s += max(cand)
            out.append(s)
        return out


def kb_rank(question: str, index: KBIndex):
    """(entry, best, runner_up) with floor + MINIMUM-2-MATCHED-STEM
    rule. The 2-stem rule killed the measured hijack: 'من الذي بنى
    الأهرامات؟' matched the WHERE-are-pyramids fact on the single
    shared stem اهرام - a real question must clear >= 2 stems."""
    q_text = _CLEAN.sub(" ", question)
    q_stems = [stem(t) for t in _TOK.findall(q_text)
               if t not in _STOP and len(t) >= 2]
    if not q_stems:
        return None
    scores = index.score(q_stems)
    if not scores:
        return None
    best_i = max(range(len(scores)), key=lambda i: scores[i])
    best = scores[best_i]
    if best < 4.5:
        return None
    matched = sum(1 for q in q_stems
                  if any(tok_match(t, q)
                         for t in index.docs[best_i]))
    if matched < 2:
        return None
    rest = sorted(scores, reverse=True)
    second = rest[1] if len(rest) > 1 else 0.0
    return index.entries[best_i], best, second


def kb_lookup(question: str, index: KBIndex) -> dict | None:
    """Scored lookup: best entry if clearly ahead (floor 4.5 +
    1.12x runner-up margin; 1.2 rejected a CORRECT winner when two
    sibling entries both matched, measured grep 19.94 vs find
    17.22). None -> caller falls through."""
    r = kb_rank(question, index)
    if r is None:
        return None
    entry, best, second = r
    if best < 1.12 * max(second, 0.1):
        return None
    return entry


# ── identity paraphrase rule (fix16) ──────────────────────────
# possessive ك-forms: "من بناك/صنعك/طورك/مبرمجك..." - the ك guard
# prevents "من صنع هذا الملف؟" from hijacking (no ك on the verb).
_ID_STEMS = {"صنع", "بنا", "طور", "برمج", "صمم", "انشا", "انشئ",
             "كتب", "اخترع", "صانع", "مبرمج", "باني", "مطور",
             "مؤسس", "مخترع", "منشئ"}
_QMARK = {"من", "بمن", "اسم"}


def identity_hit(question: str) -> bool:
    toks = _TOK.findall(
        _CLEAN.sub(" ", re.sub(r"[\u064B-\u0652]", "", question)))
    if len(toks) > 7:
        return False
    if not any(t in _QMARK for t in toks):
        return False
    for t in toks:
        # ONLY the possessive ك guard: بناك/صنعك/مبرمجك/كتبك...
        # (the old ه-suffix hijacked "من كتبه؟" about a book)
        if t.endswith("ك") and len(t) > 3:
            st = stem(t[:-1])
            if any(tok_match(st, s) for s in _ID_STEMS):
                return True
    return False
