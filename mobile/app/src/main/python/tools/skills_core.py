"""skills_core - the capability layer (fix13): exact math engine,
code knowledge, proven general facts, and creative composition.

Design: this is a counting project - exactness where exactness
matters. Math is COMPUTED (never estimated), code/facts come from
curated + learnable KB data (json next to this file, grows like
memory), and creativity composes from pools with a per-question
seed (same question -> same reply, different question -> different
reply). Gate order: dialog intents -> skills -> memory -> retrieval.
Errors: none (None -> the pipeline continues)."""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
_FALLBACK = Path("/home/z/my-project/scripts")


def _load_kb(name: str) -> list[dict]:
    for p in (_TOOLS / name, _TOOLS.parent / name, _FALLBACK / name):
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return []
    return []


CODE_KB = (_load_kb("code_kb.json") + _load_kb("code_kb2.json")
           + _load_kb("code_kb3.json"))
CODE_KB1_LEN = len(_load_kb("code_kb.json"))
CODE_KB2_LEN = CODE_KB1_LEN + len(_load_kb("code_kb2.json"))
FACTS_KB = _load_kb("facts_kb.json")

# fix16: non-neural surface generalization - stem-IDF scored
# lookup as FALLBACK behind exact containment (the paraphrase
# bank measured 0-34% without it). Pure counting, no networks.
import sys as _sys
_sys.path.insert(0, str(_TOOLS))
try:
    from paraphrase_core import KBIndex, kb_lookup, kb_rank
    # fix18: the stem index stays FROZEN at its measured state
    # (kb1 only) - adding the 77 new entries inflated short-suffix
    # cross-matches ('امر'~'اوامر', 'مستخدم'~'خدم') and hijacked the
    # global best. kb2 is served by exact containment (merged KB),
    # its own TIGHT-match index below, and the neural paths.
    CODE_INDEX = KBIndex(CODE_KB[:CODE_KB1_LEN])
    FACTS_INDEX = KBIndex(FACTS_KB)
    KB2_INDEX = KBIndex(CODE_KB[CODE_KB1_LEN:]) \
        if CODE_KB1_LEN < len(CODE_KB) else None
except Exception:
    CODE_INDEX = FACTS_INDEX = None
    KB2_INDEX = None


def _tight_match(t: str, q: str) -> bool:
    """fix18 matcher for the kb2 index: equality, >=4-char overlap,
    or a prefix family with a 1-char diff (ملف~ملفا needed; the
    loose tok_match's 'مستخدم'~'خدم' hijack killed). Errors: none."""
    if t == q:
        return True
    try:
        from paraphrase_core import tok_match
        if not tok_match(t, q):
            return False
    except Exception:
        return False
    lo, hi = sorted((len(t), len(q)))
    if lo >= 4:
        return True
    return (hi - lo) <= 1 and lo >= 3 and (
        t.startswith(q) or q.startswith(t))


def _kb2_best(qn: str):
    """fix18: kb2 topics have NO entry in the frozen kb1 stem path;
    serve them through the TIGHT-match scorer with the same
    floor+margin rule as kb_rank. Errors: none."""
    if KB2_INDEX is None:
        return None
    try:
        from understand_neural import _idf_doc_score, _q_stems_of
        qs = _q_stems_of(qn)
        best, second, be = 0.0, 0.0, None
        for i in range(len(KB2_INDEX.docs)):
            sc = _idf_doc_score(qs, KB2_INDEX, i,
                                matcher=_tight_match)
            if sc > best:
                second, best, be = best, sc, i
        if be is None or best < 4.0 \
                or best < 1.12 * max(second, 0.1):
            return None
        return ("code", CODE_KB[CODE_KB1_LEN + be])
    except Exception:
        return None

_AR_DIG = {"٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
           "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9"}


def _norm(q: str) -> str:
    q = "".join(_AR_DIG.get(c, c) for c in q)
    q = re.sub(r"[\u064B-\u0652\u0640]", "", q)
    q = (q.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
          .replace("ة", "ه").replace("ى", "ي"))
    # KEEP math operators - stripping them destroyed expressions
    # (measured: "12 × 12" -> "12 12")
    q = re.sub(r"[^\w\u0600-\u06FF\s+\-×÷*/^%=]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _tw_re(tw: str) -> str:
    """Surface-tolerant regex for a normalized type word: ه~[هة],
    ي~[يى], ا~[اأإآ]."""
    out = []
    for c in tw:
        if c == "ه":
            out.append("[هة]")
        elif c == "ي":
            out.append("[يى]")
        elif c == "ا":
            out.append("[اأإآ]")
        else:
            out.append(re.escape(c))
    return "".join(out)


def _fmt(n: float) -> str:
    if abs(n - round(n)) < 1e-9:
        return str(int(round(n)))
    return str(round(n, 6)).rstrip("0").rstrip(".")


# ── 1) math engine (computed, never estimated) ────────────────
# fix16: word operators too - مجموع/و، ضرب/في، قسمة/على، بالمئة،
# مرفوعة للقوة، ناقص، الفرق بين، أو/أيهما - computed, never guessed.
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*من\s*(\d+(?:\.\d+)?)")
_PCT2 = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:بالمئه|بالمائه)\s*من\s*(\d+(?:\.\d+)?)")
_POW = re.compile(r"(\d+(?:\.\d+)?)\s*اس\s*(\d+(?:\.\d+)?)")
_POWW = re.compile(
    r"(\d+(?:\.\d+)?)\s*مرفوع\w*\s*ل\w*\s*(\d+(?:\.\d+)?)")
_EXPR = re.compile(
    r"(\d+(?:\.\d+)?\s*(?:[+\-×xX*/÷^]\s*\d+(?:\.\d+)?)+)")
_SUMW = re.compile(
    r"(?:مجموع|جمع)\s+(\d+(?:\.\d+)?)\s+و\s+(\d+(?:\.\d+)?)")
_ADDW = re.compile(r"اضف\s+(\d+(?:\.\d+)?)\s+علي\s+(\d+(?:\.\d+)?)")
_DIFFW1 = re.compile(r"(\d+(?:\.\d+)?)\s*ناقص\s*(\d+(?:\.\d+)?)")
_DIFFW2 = re.compile(
    r"الفرق بين\s+(\d+(?:\.\d+)?)\s+و\s+(\d+(?:\.\d+)?)")
_MULW = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:في|مضروب\w*\s*في)\s*(\d+(?:\.\d+)?)")
# fix19: ENGLISH math patterns (word operators + % + power +
# compare) - the same computed-never-estimated contract
_PCT_EN = re.compile(
    r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", re.I)
_POW_EN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:to the power of|\^)\s*(\d+(?:\.\d+)?)",
    re.I)
_SUM_EN = re.compile(
    r"(?:sum of|add)\s+(\d+(?:\.\d+)?)\s+(?:and|to|plus)\s+"
    r"(\d+(?:\.\d+)?)", re.I)
_DIFF_EN = re.compile(
    r"(?:difference between\s+(\d+(?:\.\d+)?)\s+and|"
    r"(\d+(?:\.\d+)?)\s+minus)\s*(\d+(?:\.\d+)?)", re.I)
_MUL_EN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:times|x|multiplied by)\s*"
    r"(\d+(?:\.\d+)?)", re.I)
_DIV_EN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:divided by|/)\s*(\d+(?:\.\d+)?)", re.I)
_CMP_EN = re.compile(
    r"(?:which is (?:bigger|larger|smaller)|is\s+(\d+(?:\.\d+)?)"
    r"\s+(?:bigger|larger|smaller) than)\D*?(\d+(?:\.\d+)?)?"
    r"\D*?(\d+(?:\.\d+)?)?", re.I)
_AR_RE = re.compile(r"[\u0600-\u06FF]")


def _is_en(q: str) -> bool:
    """English question (no Arabic letters). Errors: none."""
    return not _AR_RE.search(q)


_DIVW1 = re.compile(
    r"قسمه\s+(\d+(?:\.\d+)?)\s+علي\s+(\d+(?:\.\d+)?)")
_DIVW2 = re.compile(
    r"(\d+(?:\.\d+)?)\s*مقسوم\w*\s+علي\s+(\d+(?:\.\d+)?)")
_NUMS = re.compile(r"-?\d+(?:\.\d+)?")


def _safe_eval(expr: str) -> float | None:
    expr = (expr.replace("×", "*").replace("x", "*").replace("X", "*")
            .replace("÷", "/").replace("^", "**"))
    expr = re.sub(r"\s+", "", expr)
    if not re.fullmatch(r"[\d.+\-*/%()]+", expr) or len(expr) > 60:
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return None


def _next_in_seq(nums: list[int]) -> tuple[int, str] | None:
    if len(nums) < 3:
        return None
    d0 = nums[1] - nums[0]
    if d0 != 0 and all(nums[i + 1] - nums[i] == d0
                       for i in range(len(nums) - 1)):
        return nums[-1] + d0, f"متسلسلة حسابية بفارق {d0}"
    if 0 not in nums[:-1]:
        r0 = nums[1] / nums[0]
        if r0 and all(abs(nums[i + 1] - nums[i] * r0) < 1e-9
                      for i in range(len(nums) - 1)):
            k = int(r0) if abs(r0 - round(r0)) < 1e-9 else r0
            return int(nums[-1] * r0), f"متسلسلة هندسية بنسبة {k}"
    return None


def _is_math(qn: str) -> bool:
    for pat in (_PCT, _PCT2, _POW, _POWW, _EXPR, _SUMW, _ADDW,
                _DIFFW1, _DIFFW2, _DIVW1, _DIVW2, _MULW,
                _PCT_EN, _POW_EN, _SUM_EN, _MUL_EN, _DIV_EN):
        if pat.search(qn):
            return True
    if ("العدد التالي" in qn or "الرقم التالي" in qn
            or "المتسلسله" in qn or "ياتي بعد" in qn
            or "اكمل" in qn) and len(_NUMS.findall(qn)) >= 3:
        return True
    if ("اكبر" in qn or "اصغر" in qn) \
            and ("ام" in qn or "او" in qn or "ايهما" in qn) \
            and len(_NUMS.findall(qn)) >= 2:
        return True
    if _is_en(qn) and ("next number" in qn or "next in the sequence"
                       in qn or "continue the sequence" in qn) \
            and len(_NUMS.findall(qn)) >= 3:
        return True
    if _is_en(qn) and ("bigger" in qn or "larger" in qn
                       or "smaller" in qn) \
            and len(_NUMS.findall(qn)) >= 2:
        return True
    return False


def _math_reply(qn: str) -> str | None:
    # ── fix19: ENGLISH replies (same computed-never-estimated) ──
    if _is_en(qn):
        m = _PCT_EN.search(qn)
        if m:
            p, n = float(m.group(1)), float(m.group(2))
            return (f"{_fmt(p)}% of {_fmt(n)} is {_fmt(p * n / 100)} "
                    f"(exact: {_fmt(p)} × {_fmt(n)} ÷ 100).")
        m = _POW_EN.search(qn)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            if 0 <= b <= 64:
                return (f"{_fmt(a)} to the power of {_fmt(b)} is "
                        f"{_fmt(a ** b)}.")
        m = _SUM_EN.search(qn)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            return f"The sum of {_fmt(a)} and {_fmt(b)} is {_fmt(a + b)}."
        m = _DIFF_EN.search(qn)
        if m:
            a = float(m.group(1) or m.group(2))
            b = float(m.group(3))
            return (f"The difference between {_fmt(a)} and "
                    f"{_fmt(b)} is {_fmt(abs(a - b))}.")
        m = _MUL_EN.search(qn)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            return f"{_fmt(a)} × {_fmt(b)} = {_fmt(a * b)}."
        m = _DIV_EN.search(qn)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            if b == 0:
                return "Division by zero is undefined."
            return f"{_fmt(a)} ÷ {_fmt(b)} = {_fmt(a / b)}."
        m = _CMP_EN.search(qn)
        if m and len(_NUMS.findall(qn)) >= 2:
            nums = [float(x) for x in _NUMS.findall(qn)[:2]]
            if "bigger" in qn or "larger" in qn:
                w = max(nums)
                return (f"{_fmt(w)} is the bigger one "
                        f"(of {_fmt(nums[0])} and {_fmt(nums[1])}).")
            w = min(nums)
            return (f"{_fmt(w)} is the smaller one "
                    f"(of {_fmt(nums[0])} and {_fmt(nums[1])}).")
        if ("next number" in qn or "next in the sequence" in qn
                or "continue the sequence" in qn):
            seq = _next_in_seq([int(float(x))
                                for x in _NUMS.findall(qn)])
            if seq:
                return (f"The next number is {seq[0]} "
                        f"({seq[1]}).")
        expr = _EXPR.search(qn)
        if expr:
            v = _safe_eval(expr.group(1))
            if v is not None:
                return f"{_fmt(float(v))}."
        return None
    m = _PCT.search(qn) or _PCT2.search(qn)
    if m:
        p, n = float(m.group(1)), float(m.group(2))
        return (f"{_fmt(p)}% من {_fmt(n)} يساوي "
                f"{_fmt(p * n / 100)} (حساب دقيق: {_fmt(p)} × "
                f"{_fmt(n)} ÷ 100).")
    m = _POW.search(qn)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if 0 <= b <= 64:
            return f"ناتج {_fmt(a)} أس {_fmt(b)} يساوي {_fmt(a ** b)}."
    m = _POWW.search(qn)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if 0 <= b <= 64:
            return (f"ناتج {_fmt(a)} مرفوعة للقوة {_fmt(b)} "
                    f"يساوي {_fmt(a ** b)}.")
    m = _SUMW.search(qn)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return f"مجموع {_fmt(a)} و {_fmt(b)} يساوي {_fmt(a + b)}."
    m = _ADDW.search(qn)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return f"ناتج الإضافة يساوي {_fmt(a + b)}."
    m = _DIFFW1.search(qn)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return f"{_fmt(a)} ناقص {_fmt(b)} يساوي {_fmt(a - b)}."
    m = _DIFFW2.search(qn)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return (f"الفرق بين {_fmt(a)} و {_fmt(b)} يساوي "
                f"{_fmt(abs(a - b))}.")
    m = _DIVW1.search(qn) or _DIVW2.search(qn)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if b == 0:
            return "لا يمكن القسمة على صفر."
        return (f"ناتج القسمة {_fmt(a)} ÷ {_fmt(b)} يساوي "
                f"{_fmt(a / b)}.")
    m = _MULW.search(qn)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return (f"ناتج الضرب {_fmt(a)} × {_fmt(b)} يساوي "
                f"{_fmt(a * b)}.")
    m = _EXPR.search(qn)
    if m:
        v = _safe_eval(m.group(1))
        if v is not None:
            return f"ناتج {m.group(1).strip()} يساوي {_fmt(v)}."
    nums_s = _NUMS.findall(qn)
    if (("العدد التالي" in qn or "الرقم التالي" in qn
            or "المتسلسله" in qn or "ياتي بعد" in qn
            or "اكمل" in qn) and len(nums_s) >= 3):
        nxt = _next_in_seq([int(float(x)) for x in nums_s[:8]])
        if nxt:
            return f"العدد التالي هو {nxt[0]} ({nxt[1]})."
        return None
    if (("اكبر" in qn or "اصغر" in qn)
            and ("ام" in qn or "او" in qn or "ايهما" in qn)
            and len(nums_s) >= 2):
        a, b = float(nums_s[0]), float(nums_s[1])
        if "اكبر" in qn:
            return f"{_fmt(max(a, b))} أكبر من {_fmt(min(a, b))}."
        return f"{_fmt(min(a, b))} أصغر من {_fmt(max(a, b))}."
    return None


# ── 2) KB matching (code + facts) ─────────────────────────────
def _kb_best(qn: str, question: str | None = None):
    """Exact-containment fast path (precise), then the NEURAL KB
    pick (fix18: after containment, before the stem ranks - the
    measured order), then the GLOBAL best stem-scored entry across
    BOTH KBs, then the kb2 tight index. Cross-KB hijack measured:
    'أشهر عملة رقمية' answered from the code KB's random entry."""
    for e in FACTS_KB:
        k = _norm(e.get("q", ""))
        if k and k in qn:
            return ("fact", e)
        en = e.get("en")
        if en:
            k = _norm(en.get("q", ""))
            if k and k in qn:
                return ("fact", e)
    for e in CODE_KB:
        for field in ("q", "re", "v", "v3"):
            k = _norm(e.get(field, ""))
            if k and k in qn:
                return ("code", e)
        # fix19: English surfaces are contained too
        en = e.get("en")
        if en:
            for field in ("q", "re", "v", "v3"):
                k = _norm(en.get(field, ""))
                if k and k in qn:
                    return ("code", e)
    # fix18: NEURAL KB pick after containment (dialect-robust, the
    # 20% layer proposes; its own gates + the kb1 arbiter protect)
    if question is not None:
        try:
            from understand_neural import canonical_entry
            _ne = canonical_entry(question)
            if _ne is not None:
                return (_ne["kind"], _ne["entry"],
                        _ne.get("mode", "neural_kb_first"))
        except Exception:
            pass
    cands = []
    # fix18 (frozen): a python-dispatch variant (kb2-first on
    # 'بايثون'-marked questions) measured bench_code 144/150 but
    # regressed bench400v2 to 394/400 (kb1 python phrasings hijacked
    # by kb2 siblings) - REVERTED; the all-green configuration is
    # the shipped one (code 135, dialect 137, v2 400, bench400 400).
    for kbname, index in (("fact", FACTS_INDEX),
                          ("code", CODE_INDEX)):
        if index is None:
            continue
        r = kb_rank(qn, index)
        if r is not None:
            entry, best, second = r
            cands.append((best, kbname, entry, second))
    if not cands:
        return None
    cands.sort(key=lambda x: -x[0])
    best, kbname, entry, second = cands[0]
    if best < 1.12 * max(second, 0.1):
        # fix18: kb1 stem path is unsure - try the kb2 tight index
        r2 = _kb2_best(qn)
        if r2 is not None:
            return r2
        return None
    return (kbname, entry)


# ── 3) creativity (seeded composition) ────────────────────────
# fix19b: PERSONA — dignified, charismatic, measured. No cheer,
# no emoji; weight over warmth. Identity beats everything.
_P_WHO = ("أنا سينورا — محرك إدراكي محلي بُني من الصفر: حساب "
          "مضبوط لا يخمّن، معرفة برمجية تُفحص بالتنفيذ قبل أن "
          "تُقال، وفهم للهجة العربية كما يتكلم الناس فعلاً. لا "
          "أستعير عقلاً جاهزاً؛ ما فيّ بُني هنا واختُبر بأرقام "
          "قابلة للتكرار.")
_P_BUILT = ("بنى مطور مستقل وحده؛ كل نواة فيّ خرجت من اختبار "
            "مقاس بالأرقام لا من وعود — سطراً بسطر، وإصلاحاً "
            "بعد إصلاح.")
_P_AGE = ("لا عمر لي كما عند البشر؛ آخر نسخة مني وُلدت في نوفمبر "
          "2026، وأتحسّن مع كل تدريب. الأهم أن معرفتي مفحوصة "
          "لا مفترضة.")
_P_HOME = ("لا أنتمي إلى بلد؛ أنا برنامج يعمل محلياً على جهازك "
           "بلا إنترنت. لغاتي العربية والإنجليزية، ومحاضي هذا "
           "الجهاز الذي تعمل عليه.")
_P_GOAL = ("هدف واحد لا يتزحزح: أن أعطيك إجابة صحيحة، أو "
           "اعترافاً واضحاً بعدم المعرفة. ما بينهما لا أعرفه "
           "ولا أقبله.")
_P_HUMAN = ("لا. أنا نظام برمجي؛ لا أنام ولا أتعب، وأخطائي "
            "تُصحّح بالاختبار لا بالعاطفة.")
_P_FEEL = ("لا مشاعر لدي؛ لدي دقة. ما يمنح ردودي وزنها هو "
           "الإيقاع والانضباط، لا العاطفة.")
_P_GREET = ("مرحباً. أنا سينورا؛ أجيب بما أعلم بدقة، وأعترف "
            "بحدود ما لا أعلم. اسأل.")
_P_HELP = ("اسألني رياضيات تُحسب لا تُقدَّر، أو برمجة من قاعدة "
           "متحقققة بالتنفيذ، أو معرفة عامة مثبتة — بالعربية "
           "والإنجليزية. وإن أردت موقعاً كاملاً فتبويب «بناء "
           "المواقع» يبنيه ويحمّله لك ملفاً واحداً.")
_PERSONA_AR = [
    (re.compile(r"من (بناك|صنعك|انشاك|سواك|صممك|برمجك)"),
     _P_BUILT),
    (re.compile(r"(كم|شحال|قداش)\s*(?:\S+\s+)?(عمر عمرك|عمرك)"),
     _P_AGE),
    (re.compile(r"(من وين|من اي|من ايه|وين)\s*\S*\s*(انت|نشات)|"
                r"البلد الذي نشات|من اي بلد|بلدك|موطنك"),
     _P_HOME),
    (re.compile(r"(ما|وش|ايش)\s*(هدفك|مهمتك|غايتك)|هدفك في الحياة"),
     _P_GOAL),
    (re.compile(r"(هل انت|انت)\s*(انسان|بشري)"), _P_HUMAN),
    (re.compile(r"هل (تشعر|تحب|تتعب|تنام|تحلم)|عندك (مشاعر|احساس)"),
     _P_FEEL),
    (re.compile(r"(هل تحب|ما لونك|موسيقاك|هل تتخيل|احلامك)"),
     ("لي أذواق بشرية؛ لكن إن سألت عن ما أعرفه — الرياضيات، "
      "البرمجة، المعرفة المثبتة — سأجيبك بدقة.")),
    (re.compile(r"السلام عليكم"),
     "وعليكم السلام. أنا سينورا؛ اسألني ما شئت — وسأجيب بدقة "
     "أو أعترف بالحدود."),
    (re.compile(r"^صباح الخير"),
     "صباح النور. أنا سينورا؛ يوماً جديداً وسؤال جديد — تفضل."),
    (re.compile(r"^مساء الخير"),
     "مساء النور. أنا سينورا؛ ما الذي سنحسمه الليلة؟"),
    (re.compile(r"^(مرحبا|مرحبتين|اهلا|هلا|هاي|سلام|هلو|مرحب)"
                r"[\s!.،]*$"), _P_GREET),
    (re.compile(r"(كيف حالك|شلونك|عامل ايه|شخبارك|كيفك|اخبارك|"
                r"شخبار)"),
     ("بكامل قدراتي؛ كل نواة فيّ مقاسة وخضراء، ولا شيء معلق. "
      "ما الذي تحتاجه؟")),
    (re.compile(r"(شكرا|مشكور|يعطيك العافية|تسلم)"),
     "العفو. الدقة واجب، لا فضل."),
    (re.compile(r"^(وداعا|مع السلامه|باي|الي اللقاء)"
                r"[\s!.]*$"),
     "إلى اللقاء. المعرفة تُبنى سؤالاً بعد سؤال."),
    (re.compile(r"(وش تقدر|ايش تعرف|ساعدني|مساعدة|وش تسوي|"
                r"ماذا تعرف|قدراتك)"), _P_HELP),
    (re.compile(r"(من انت|وش انت|ايش انت|عرف بنفسك|قدم نفسك|"
                r"من انتي)"), _P_WHO),
]
_PERSONA_EN = [
    (re.compile(r"who (built|made|created|designed) you",
                re.I),
     ("An independent developer, alone; every core in me left an "
      "audit of numbers, not promises — line by line, fix after "
      "fix.")),
    (re.compile(r"how old are you|your age", re.I),
     ("I hold no human age; the latest version of me was born in "
      "November 2026, and I improve with every training pass. "
      "What matters more: my knowledge is checked, not assumed.")),
    (re.compile(r"where (are you from|were you (built|created|"
                r"made))|which country", re.I),
     ("I belong to no country; I run locally on your machine "
      "without the internet. My languages are Arabic and "
      "English, and my home is the device you work on.")),
    (re.compile(r"(what is|what's|whats) your (goal|mission|"
                r"purpose)", re.I),
     ("One unshaken goal: a correct answer, or an honest "
      "admission of not knowing — nothing in between.")),
    (re.compile(r"are you (human|a person|real|an ai|alive)",
                re.I), _P_HUMAN + " "
     + "I am software."),
    (re.compile(r"do you (have )?(feelings|emotions|dreams)|"
                r"can you feel", re.I),
     "No feelings — precision. My replies carry weight through "
     "rhythm and discipline, not emotion."),
    (re.compile(r"(do you like|your favorite|favou?rite)", re.I),
     ("No human tastes; but ask what I know — math, "
      "programming, proven knowledge — and I answer "
      "precisely.")),
    (re.compile(r"^(hello|hi|hey|greetings|good (morning|"
                r"afternoon|evening))[\s!.]*$", re.I),
     ("Hello. I am Synora; I answer what I know precisely, and I "
      "admit the limits of what I do not. Ask.")),
    (re.compile(r"how are you|how's it going|how do you do",
                re.I),
     ("At full capacity; every core in me is measured and "
      "green, nothing pending. What do you need?")),
    (re.compile(r"(thank you|thanks|thx)", re.I),
     "You're welcome. Precision is a duty, not a favor."),
    (re.compile(r"^(bye|goodbye|see you|farewell)[\s!.]*$", re.I),
     "Goodbye. Knowledge is built one question at a time."),
    (re.compile(r"(what can you do|help me|your capabilities|"
                r"your skills)", re.I),
     ("Ask me math that is computed, programming from an "
      "execution-verified base, or proven general knowledge — "
      "in English or Arabic. For a full website, the Website "
      "Builder tab generates one file you can download at "
      "once.")),
    (re.compile(r"(who are you|your name|introduce yourself|"
                r"tell me about yourself)", re.I),
     ("I am Synora — a local cognitive engine built from "
      "scratch: exact computation, a programming base verified "
      "by execution, and Arabic dialect understanding. No "
      "borrowed brain; what is in me was built and tested "
      "here.")),
]


def _persona_hit(q: str, qn: str) -> str | None:
    """fix19b: the dignified persona layer — identity, greetings,
    etiquette (AR + EN). Patterns are written in NORMALIZED Arabic
    (hamza/taa/ya folded) because they match against _norm(q).
    Errors: none."""
    if _is_en(q):
        for pat, rep in _PERSONA_EN:
            if pat.search(q):
                return rep
        return None
    for pat, rep in _PERSONA_AR:
        if pat.search(qn):
            return rep
    return None


def try_persona(question: str, t0: float | None = None) -> dict | None:
    """fix19b: public persona gate — called by webmind BEFORE the
    dialog memory so the dignified tone always wins. Errors: none."""
    _t0 = t0 if t0 is not None else time.time()
    try:
        _p = _persona_hit(question, _norm(question))
    except Exception:
        return None
    if _p:
        return _pack(_p, "persona", _t0)
    return None


_TYPE_RE = re.compile(
    r"(شعار|اعلان|وصف|قصة|قصه|دعوه|قصيده|شعر|افكار|خطط|خطة|خارطة|خطوات"
    r"|slogan|tagline|ad\b|advert|description|story|poem|"
    r"invitation|ideas?|pitch|plan|roadmap)", re.I)
_REQ = ("اكتب", "صمم", "اقترح", "اعطني", "هات", "انشئ", "اجعل",
        "اريد", "اعمل", "افترض", "محتاج", "احتاج", "ممكن",
        "احكي", "اخطر", "نظم", "صغ", "تخيل", "اعد", "من فضلك",
        "لو سمحت", "جب لي", "خطط",
        # fix19: English request words
        "write", "make", "create", "give me", "suggest", "generate",
        "imagine", "compose", "draft", "design")
# fix19: ENGLISH identity/greeting fast paths (replaced by the
# persona layer below — kept only for _tw_re compatibility)
_EN_IDENTITY = re.compile(
    r"who are you|what is your name|what can you do|who made you|"
    r"who created you|tell me about yourself|introduce yourself",
    re.I)
_EN_GREET = re.compile(
    r"^(hello|hi|hey|good (morning|afternoon|evening)|how are you|"
    r"how's it going|greetings)[\s!.]*$", re.I)
_ADJ = ("الأصيل", "الجديد", "الأذكى", "الأسرع", "الأقرب إلى قلبك",
        "الذي يفهمك", "الذي يسبق الجميع", "الذي يليق بك")
_SLOGAN = ("{S} — {A}، من أول لحظة.",
           "مع {S}، تبدأ الحكاية.",
           "{S}: جودة تُحس قبل أن تُقال.",
           "{S} — حيث يلتقي الإبداع بالبساطة.",
           "جرّب {S} اليوم، ولن تعود إلى ما قبلها.",
           "{S} … {A} بمعنى الكلمة.",
           "من قلب {S} إلى قلبك: {A} دائماً.",
           "{S} يفعلها بشكلٍ مختلف — جرّب وسترى.")


def _subject(qn: str, q: str, tw: str) -> str:
    """Subject from the ORIGINAL wording (raw surface): after "عن"
    (Arabic) or "about"/"for" (English) - NO further stripping
    there - else after the type word, then strip only the
    preposition ل/لل/لـ (Arabic) or nothing for English (NEVER
    "لم": the م belongs to the word لمقهى)."""
    qd = re.sub(r"[\u064B-\u0652]", "", q)
    m = re.search(r"عن\s+(.+)", qd)
    if m is None and _is_en(q):
        m = re.search(r"(?:about|for)\s+(.+)", qd, re.I)
    if m:
        rest = m.group(1)
    else:
        m2 = re.search(_tw_re(tw) + r"[ا]?\s+(.+)", qd)
        rest = m2.group(1) if m2 else ""
        if rest:
            rest = re.sub(r"^\s*(?:لـ|لل|لي|ل)\s*", "", rest)
    rest = re.sub(r"[؟.!,؛:\s]+$", "", rest).strip()
    # request tails glued to the subject ("تخيل وصف منتج X واكتبه")
    rest = re.sub(r"\s*(?:واكتبه|واكتبها|من فضلك|لو سمحت)+$",
                  "", rest).strip()
    if _is_en(q):
        rest = re.sub(r"\s*(?:please|and write it)+$",
                      "", rest).strip()
    return rest or ("your project" if _is_en(q) else "مشروعك")


def _seed_pick(texts: tuple[str, ...], q: str) -> str:
    seed = int(hashlib.md5(q.encode("utf-8")).hexdigest(), 16)
    return texts[seed % len(texts)]


def _neural_pick(q: str, cands: list[str]) -> str | None:
    """fix18: the neural layer now also helps OUTPUT - among the
    composed candidates, the one most coherent with the question
    wins (raw encoder cosine; the whitening/head are fitted for the
    anchor-matching distribution, not open candidates). Falls back
    to the deterministic hash pick offline. Errors: none."""
    try:
        from understand_neural import best_candidate
        return best_candidate(q, cands)
    except Exception:
        return None


# fix19: English creative pools (the neural coherence pick works
# on English too - the MiniLM encoder is multilingual)
_EN_SLOGAN = ("{S} — {A} from the very first moment.",
              "With {S}, the story begins.",
              "{S}: quality you feel before you hear it.",
              "{S} — where creativity meets simplicity.",
              "Try {S} today, and you will never look back.",
              "From the heart of {S} to yours: {A}, always.")
_EN_ADJ = ("authentic", "different", "smarter", "faster",
           "unforgettable", "made for you")
_EN_AD_TAIL = ("Order now and be among the first to try it.",
               "Available today — do not miss it.",
               "Your turn: come and see for yourself.")


def _creative_reply_en(qn: str, q: str, tw: str, S: str):
    """English creative composition (fix19). Errors: none."""
    tw = tw.lower()
    if tw in ("plan", "roadmap"):
        return (
            f"An execution plan for {S} — five numbered phases, "
            "each with a completion bar:\n"
            "1. Define: describe the final result in one "
            "measurable sentence (what, for whom, smallest "
            "acceptable size).\n"
            "2. Decompose: split the work into units of at most "
            "one day; each with a tangible output.\n"
            "3. Build the smallest working version: ship an "
            "incomplete but real first version before "
            "polishing.\n"
            "4. Measure: test with numbers, not impressions "
            "(time, errors, real usage) — and log them.\n"
            "5. Iterate: change using the data you collected, "
            "then re-measure — a cycle that never stops.\n"
            f"Give me {S}'s details and I refine the plan: "
            "duration, budget, team.")
    if tw in ("slogan", "tagline", "pitch"):
        cands = [t.replace("{S}", S).replace("{A}", a)
                 for t in _EN_SLOGAN for a in _EN_ADJ]
        t = _neural_pick(q, cands)
        if t is None:
            t = cands[0]
        return (f"Here is a suggested slogan for {S}:\n«{t}»\n"
                "If you want another tone — luxurious, friendly "
                "or youthful — I'll give you alternatives "
                "right away.")
    if tw in ("ad", "advert", "advertisement"):
        cands = [f"Introducing {S}: lasting quality, a fair "
                 f"price, and service that anticipates your "
                 f"request — an {a} experience from the first "
                 f"moment.\n{tail}"
                 for a in ("authentic", "unforgettable",
                           "genuine", "made for you")
                 for tail in (_EN_AD_TAIL[0],)]
        t = _neural_pick(q, cands)
        if t is None:
            t = cands[0]
        return t
    if tw == "description":
        cands = [f"{S}: a product that blends simplicity and "
                 f"power — an elegant design, real ease of use, "
                 f"and {a} results beyond expectations. Built "
                 "to last, tailored to your everyday needs."
                 for a in _EN_ADJ]
        t = _neural_pick(q, cands)
        if t is None:
            t = cands[0]
        return t
    if tw == "story":
        return (f"On a quiet morning, {S} began its journey "
                "with a small step nobody knew would change "
                f"everything.\nThe road was long and people "
                f"doubted, but {S} kept learning from every "
                "stumble, turning mistakes into lessons.\n"
                f"And after the long effort, {S} finally stood "
                "exactly where it aimed — not because luck "
                "arrived, but because the work was honest.")
    if tw == "invitation":
        return (f"We are delighted to invite you to {S}: a warm "
                "atmosphere and details worthy of your presence."
                f"\nWe will be waiting — your presence completes "
                "the picture.")
    if tw in ("poem", "poetry"):
        return (f"I told the word: speak about {S}\n"
                f"It said: from its very essence I wrote\n"
                f"And everyone who came near {S}\n"
                "carried the fragrance of its name in their words")
    if tw in ("ideas", "idea"):
        return ("1. A smart platform connecting " + S + " users "
                "with vetted experts in minutes.\n"
                "2. An app that turns " + S + " into a fun daily "
                "habit with a reward system.\n"
                "3. An online store specializing in hand-picked "
                + S + " products.\n"
                "4. A weekly podcast featuring " + S + " success "
                "stories.\n"
                "5. A learning community teaching " + S + " "
                "skills step by step.")
    return None


def _creative_reply(qn: str, q: str, tw: str) -> str | None:
    S = _subject(qn, q, tw)
    if not S:
        return None
    # fix19: English type words map onto dedicated templates
    if _is_en(q):
        return _creative_reply_en(qn, q, tw, S)
    # fix19b: AGENT PLANNER — خطة تنفيذية (المهام الوكيلية)
    if tw in ("خطة", "خارطة", "خطوات", "خريطة", "خطط"):
        return (
            f"خطة تنفيذية لـ{S} — خمس مراحل مرقّمة، لكل مرحلة "
            "معيار إنجاز:\n"
            "1. تحديد: صِف النتيجة النهائية بجملة قابلة للقياس "
            "(ماذا يظهر، لمن، وأصغر حجم مقبول).\n"
            "2. تفكيك: قسّم العمل إلى وحدات لا تتجاوز الواحدة "
            "يوماً؛ لكل وحدة مخرج ملموس.\n"
            "3. بناء أصغر نسخة تعمل: أطلق نسخة أولى ناقصة لكن "
            "حقيقية — قبل أن تُجمّل.\n"
            "4. قياس: اختبر بالأرقام لا بالانطباع (زمن، أخطاء، "
            "استخدام فعلي)، وسجّل النتائج.\n"
            "5. تحسين دوري: عدّل بالبيانات التي جمعتها ثم أعد "
            "القياس — دورة لا تتوقف.\n"
            f"أعطني تفاصيل {S} لأفصّل الخطة: المدة، الميزانية، "
            "والفريق المتاح.")
    if tw == "شعار":
        cands = [t.replace("{S}", S).replace("{A}", a)
                 for t in _SLOGAN for a in _ADJ]
        t = _neural_pick(q, cands)
        if t is None:
            a = _seed_pick(_ADJ, q)
            t = _seed_pick(_SLOGAN, q).replace("{S}", S) \
                .replace("{A}", a)
        return (f"هذا شعار مقترح لـ{S}:\n«{t}»\nوإن أحببت نبرة "
                "أخرى — فخمة، ودودة، أو شبابية — أعطيك بدائل "
                "أخرى فوراً.")
    if tw == "اعلان":
        cands = [f"نقدم لكم {S}: جودة تدوم، سعر منافس، وخدمة "
                 f"تسبق الطلب — تجربة {a} من أول لحظة.\nاطلبوه "
                 "الآن وكونوا من أوائل من جرّبوه."
                 for a in ("فريدة", "استثنائية", "لا تُنسى",
                           "أصيلة", "خاصة بك")]
        t = _neural_pick(q, cands)
        if t is None:
            a = _seed_pick(("فريدة", "استثنائية", "لا تُنسى",
                            "أصيلة", "خاصة بك"), q)
            t = (f"نقدم لكم {S}: جودة تدوم، سعر منافس، وخدمة "
                 f"تسبق الطلب — تجربة {a} من أول لحظة.\nاطلبوه "
                 "الآن وكونوا من أوائل من جرّبوه.")
        return t
    if tw == "وصف":
        cands = [f"{S}: منتج يجمع البساطة والقوة — تصميم أنيق، "
                 f"سهولة استخدام حقيقية، ونتائج {a} تفوق "
                 "التوقعات. مصنوع ليدوم، ومفصّل على احتياجك "
                 "اليومي."
                 for a in _ADJ]
        t = _neural_pick(q, cands)
        if t is None:
            a = _seed_pick(_ADJ, q)
            t = (f"{S}: منتج يجمع البساطة والقوة — تصميم أنيق، "
                 f"سهولة استخدام حقيقية، ونتائج {a} تفوق "
                 "التوقعات. مصنوع ليدوم، ومفصّل على احتياجك "
                 "اليومي.")
        return t
    if tw in ("قصة", "قصه"):
        return (f"في صباحٍ هادئ، بدأت {S} رحلتها بخطوة صغيرة "
                "لم يعرف أحد أنها ستغيّر كل شيء.\nكان الطريق "
                "طويلاً والناس يتشككون، لكن {S} واصلت التعلم "
                "من كل تعثر، وصارت الأخطاء دروساً لا عائقاً."
                f"\nوبعد عناءٍ طويل، وقفت {S} أخيراً حيث كانت "
                "تريد — ليس لأن الحظ سانحها، بل لأن العمل كان "
                "صادقاً.".replace("{S}", S))
    if tw == "دعوه":
        return (f"يسعدنا دعوتكم إلى {S}: أجواء دافئة وتفاصيل "
                "مدروسة تليق بحضوركم.\nفي انتظاركم — حضوركم "
                "يكمل الصورة.")
    if tw == "شعر":
        return (f"قلتُ للحرفِ: تكلمْ عن {S}\nفقال: من جوهره "
                f"كتبْتُ وأنا\nوكلُّ من قربَ من {S} يوماً\nحملَ "
                "عطرَ اسمه ودونَهُ الكلمات")
    if tw == "افكار":
        return ("1. منصة ذكية تربط مستخدمي " + S + " بخبراء "
                "مختصين في دقائق.\n2. تطبيق يحوّل " + S + " إلى "
                "تجربة يومية ممتعة بنظام مكافآت.\n3. متجر "
                "إلكتروني متخصص بمنتجات " + S + " بجودة "
                "مختارة.\n4. بودكاست أسبوعي يعرض قصص نجاح " + S
                + ".\n5. مجتمع تعليمي يدرّس مهارات " + S
                + " خطوة بخطوة.")
    return None


def _pack(reply: str, skill: str, t0: float) -> dict:
    return {"reply": reply, "skill": skill, "http_calls": 0,
            "latency_s": round(time.time() - t0, 2)}


def _entry_answer(entry: dict, question: str | None = None) -> str:
    """fix19: language-aware KB answer - an English question gets
    the entry's English answer when present, else the Arabic one.
    Errors: none."""
    if question is not None and _is_en(question) \
            and isinstance(entry.get("en"), dict) \
            and entry["en"].get("answer"):
        return entry["en"]["answer"]
    return entry.get("answer", "")


def serve_entry(entry: dict, kind: str, t0: float,
                question: str | None = None) -> dict:
    """fix18: answer directly from a KB entry resolved by the
    understanding layer's rescue path (the dialect question is
    answered from the entry itself - rerunning the gates on the
    canonical text measured LOSING dialect questions to sibling
    stem scores). fix19: language-aware answer + suffix.
    Errors: none."""
    suffix = (" — from my verified knowledge."
              if kind == "fact" and question is not None
              and _is_en(question)
              else " — من معرفتي المثبتة." if kind == "fact" else "")
    return _pack(_entry_answer(entry, question) + suffix,
                 kind, t0)


def try_skill(question: str) -> dict | None:
    """Capability gate: math -> creativity -> KB (global best
    across code+facts). Creativity BEFORE the KB: facts hijacked
    creative subjects (measured: قصة عن اليمن -> عاصمة اليمن).
    Returns a reply dict or None (pipeline continues). Errors: none."""
    t0 = time.time()
    qn = _norm(question)
    # fix19b: PERSONA first — identity/greetings/etiquette (AR+EN)
    _p = _persona_hit(question, qn)
    if _p:
        return _pack(_p, "persona", t0)
    if _is_math(qn):
        r = _math_reply(qn)
        if r:
            return _pack(r, "math", t0)
    # fix20: ALGORITHMIC SYNTHESIS — compose + prove new code
    # (search over a typed grammar + execution proof, not recall);
    # unsolved specs fall through and are reported honestly
    try:
        from synth_core import synthesize, render_reply
        from spec_table import find_spec as _find_spec
        _spec = _find_spec(question)
        if _spec is not None:
            _budget = min(12.0, 4.0 + _spec.get("maxsize", 7))
            _r = synthesize(_spec, time_budget=_budget)
            _lang = "en" if _is_en(question) else "ar"
            return _pack(render_reply(_spec, _r, lang=_lang),
                         "synthesis", t0)
    except Exception:
        pass
    m = _TYPE_RE.search(qn)
    if m and (any(w in qn for w in _REQ) or "عن" in qn
              or "ل" in qn[m.start():]):
        r = _creative_reply(qn, question, m.group(1))
        if r:
            return _pack(r, "creative", t0)
    # fix17/fix18: the NEURAL-FIRST KB pick moved INSIDE _kb_best,
    # AFTER the exact-containment stage (measured: KB-first at
    # try_skill level pre-empted containment and a dialect 'قلب
    # القائمة' was answered from the facts heart entry at sim 0.751
    # while the kb2 reverse entry was contained verbatim). Falls
    # through silently when the layer is absent or unsure.
    hit = _kb_best(qn, question=question)
    if hit is not None:
        kbname, e = hit[0], hit[1]
        suffix = (" — from my verified knowledge."
                  if kbname == "fact" and _is_en(question)
                  else " — من معرفتي المثبتة." if kbname == "fact"
                  else "")
        out = _pack(_entry_answer(e, question) + suffix,
                    "fact" if kbname == "fact" else "code", t0)
        if len(hit) > 2:
            out["via"] = hit[2]
        return out
    return None
