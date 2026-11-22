"""dialog_core - the dialog layer (fix11): intent gate + solid
identity + direct smart replies + naming engine.

Design decision (documented): questions ABOUT the assistant or the
conversation (identity, creator, greeting, thanks, creative help)
must be answered DIRECTLY and honestly - never routed to external
sources, never learned as "facts". External retrieval stays for
real world questions only. The classifier is lexical-scoring over
normalized token patterns (no neural parts), and multi-intent
questions are answered as ONE assembled reply (identity + naming
in the same breath). Errors: none (unknown -> None -> the normal
pipeline)."""
from __future__ import annotations

import re
import time

try:
    from paraphrase_core import identity_hit as _identity_hit
except Exception:
    _identity_hit = None

_TOK = re.compile(r"[\w\u0600-\u06FF]+")
_AR = re.compile(r"[\u0600-\u06FF]")

_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")
_SUFFIXES = ("هما", "كما", "هن", "ها", "هم", "كم", "نا",
             "ات", "ون", "ين", "ان", "يه", "تي")
_GREET = ("مرحبا", "اهلا", "هلا", "السلام", "صباح", "مساء",
          "hi", "hello", "hey")
_QWORDS = ("من", "ما", "كيف", "هل", "لماذا", "متى", "اين",
           "ماذا", "وش", "شنو", "when", "what", "where", "why",
           "how", "who", "is", "are", "does", "can")


def toks(text: str, deep: bool = False) -> list[str]:
    """Same normalization family as webmind.content_tokens (kept
    local to avoid a circular import). Errors: none."""
    out = []
    for t in _TOK.findall(text):
        t = t.strip(".,!?؛:«»()\"'؟؟—ـَُِّْ")
        if len(t) < 2:
            continue
        t = t.lower()
        t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
        t = t.replace("ة", "ه").replace("ى", "ي").replace("ـ", "")
        if t.startswith("ال") and len(t) >= 5:
            t = t[2:]
        if deep:
            for p in _PREFIXES:
                if t.startswith(p) and len(t) - len(p) >= 3:
                    t = t[len(p):]
                    break
            for s in _SUFFIXES:
                if t.endswith(s) and len(t) - len(s) >= 3:
                    t = t[:-len(s)]
                    break
        out.append(t)
    return out


def ar_ratio(text: str) -> float:
    chars = [c for c in text if c.strip()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if _AR.match(c)) / len(chars)


def _norm(text: str) -> str:
    return " ".join(toks(text))


# ── identity (hard internal facts, honest and counted) ────────
# fix17: composition changed by owner decision - 20% neural
# understanding layer (open-source MiniLM paraphrase encoder,
# refined on the project's own data) / 60% numeric-evolutionary
# core / 10% symbolic gates. The self-description says so.
IDENTITY_AR = (
    "أنا سينورا (Synora16/WebMind): نموذج عربي هجين — قلب عدّي "
    "تطوري يولّد الإجابات بالعدّ والحساب والذاكرة، وطبقة فهم "
    "عصبية صغيرة مفتوحة المصدر (مُشفّر متعدد اللغات مُقلّص حجمه "
    "واضُقل تدريب رأسه على بياناتي الخاصة) تفهم صيغ السؤال "
    "ولهجاته وتوجّهها، وقواعد رمزية للحساب الدقيق وخطاب الهوية. "
    "درّبني مهندس الذكاء الاصطناعي اليمني ليث أحمد خديش مباشرة "
    "على مكتبات Hugging Face وGitHub الحقيقية (مسارات برمجة "
    "وأوامر طرفية، حوار مفتوح، تسويق، نص عام)، وإزالة تكرار "
    "صارمة ببصمة كل سطر.")
CREATOR_AR = (
    "بناني وطورني مهندس الذكاء الاصطناعي اليمني ليث أحمد خديش "
    "بنفسه: هو من قرر فلسفتي — أن أعرف ما أعرف بثقة، وأبحث في "
    "مصادري فقط عندما يحتاج السؤال ذلك، وأقول «لا أعلم» بصدق إن "
    "لم أجد جواباً موثوقاً.")
BECOMING_AR = (
    "وأما ماذا سأكون: أتحول جولة بعد جولة من عدّاد انتقالات إلى "
    "مرجع عربي يجمع فهم نية السؤال، ومعرفة هويته، وذاكرة تنمو مع "
    "كل محادثة — كل تغذية جديدة تكبّر مداركي.")
AD_AR = (
    "تفضل — هذا نصي الإعلاني عن نفسي:\n\n"
    "«سينورا — عقل عربي هجين: قلبٌ يعدّ، وطبقة تفهم لهجتك.\n"
    "قلبي عدّي تطوري: يجيب من معرفته ويحسب بدقة ويتعلم من كل "
    "محادثة بلا نماذج جاهزة تقلّده. وطبقة الفهم العصبية الصغيرة "
    "مفتوحة المصدر المدمجة فيه تعرف عربيك أياً كانت صيغته. بناه "
    "ودرّبه مهندس الذكاء الاصطناعي اليمني ليث أحمد خديش من "
    "مكتبات Hugging Face وGitHub الحقيقية.\n"
    "اسأله؟ يجيب مباشرة إذا يعرف، ويبحث في مصادره إذا احتاج، ويقول "
    "بصدق «لا أعلم» إذا لم يجد — لا فبرك ولا ترقيع.\n"
    "سينورا: ذكاء يمني أصيل يجيب، يبحث، يتعلّم — ويذكر صانعه في "
    "كل جواب.\n"
    "جرّبه اليوم، واكتشف عربياً يفهمك من صانعه العربي.»")
AD_EN = (
    "Here is my own ad copy:\n\n"
    "\"Synora - a hybrid Arabic mind: a numeric-evolutionary core "
    "that counts, computes and learns, plus a small open-source "
    "neural understanding layer that reads your dialect. No "
    "copycat ready-made generation. Built and trained by the "
    "Yemeni AI engineer Laith Ahmed Khadeesh from real Hugging "
    "Face and GitHub libraries. Ask it: it answers directly when "
    "it knows, searches its own sources when it needs, and "
    "honestly says 'I don't know' when it doesn't. Synora: "
    "authentic Yemeni AI that answers, searches, learns - and "
    "credits its maker in every answer.\"")
CAPAB_AR = (
    "أستطيع مساعدتك في: الإجابة عن معلومات واقعية بعد البحث في "
    "مصادري المسجلة، المساعدة البرمجية من ما تعلمته من مسارات "
    "الوكيل والمكتبات، اقتراح الأسماء والأفكار، والحوار العربي "
    "المباشر.")

NAMES = [
    ("بصيرة", "Baseera",
     "الأقرب لجوهر الذكاء الاصطناعي: رؤية تفهم قبل أن تشرح؛ قصير "
     "وعربي أصيل ويناسب علامة عالمية."),
    ("مِغزا", "Mighza",
     "مشتق من «المغز» الدماغي: يربط الآلة بالعقل البشري في كلمة "
     "سهلة النطق وفريدة في سجل الأعمال."),
    ("نِبراس", "Nibras",
     "المنارة التي تُرشد: يعطي انطباع الاستدلال والتوجيه، مثالي "
     "لمنتجات المساعدة الذكية."),
    ("مِدارك", "Mudarak",
     "جمع «مدارك»: يوحي بإدراك متعدد الطبقات يفهم النص والسياق "
     "معاً."),
    ("وَميض", "Wamid",
     "سرعة الفكرة وتلألؤها: اسم حاد قصير سهل الحفظ والنطق."),
]


def _identity_reply(show_creator: bool, show_becoming: bool,
                    show_caps: bool, ar: bool) -> str:
    if not ar:
        s = ("I am Synora (Synora16/WebMind): a hybrid Arabic "
             "model - a numeric-evolutionary core that answers by "
             "counting, computing and memory, plus a small "
             "open-source neural understanding layer (refined on "
             "my own data) that reads question forms and dialects, "
             "and symbolic rules for exact math and identity. "
             "Trained directly by the Yemeni AI engineer Laith "
             "Ahmed Khadeesh on Hugging Face and GitHub "
             "libraries.")
        if show_creator:
            s += (" My builder is Laith Ahmed Khadeesh, the "
                  "Yemeni AI engineer - he set my philosophy: "
                  "answer what I know, search only when the "
                  "question truly needs it, and say \"I don't "
                  "know\" honestly.")
        if show_caps:
            s += (" I can help with factual answers from my "
                  "registered sources, coding help from my agent-"
                  "trajectory training, name and idea suggestions, "
                  "and direct Arabic conversation.")
        return s
    parts = [IDENTITY_AR]
    if show_creator:
        parts.append(CREATOR_AR)
    if show_becoming:
        parts.append(BECOMING_AR)
    if show_caps:
        parts.append(CAPAB_AR)
    return " ".join(parts)


def _naming_reply(q: str, ar: bool) -> str:
    tn = _norm(q)
    subject = "مشروعك"
    if any(w in tn for w in ("شركت", "شركه")):
        subject = "شركتك المتخصصة في الذكاء الاصطناعي"
    elif any(w in tn for w in ("تطبيق", "منتج", "منصه", "موقع")):
        subject = "منتجك الجديد"
    elif any(w in tn for w in ("مشروع",)):
        subject = "مشروعك التقني"
    if not ar:
        s = ("Gladly! For your AI company, my top five name "
             "proposals:")
        for i, (a, e, why) in enumerate(NAMES, 1):
            s += f"\n{i}. {e} ({a}) - {why}"
        s += ("\nTell me the market and tone you prefer and I "
              "will generate a fresh list.")
        return s
    lines = [f"يسعدني ذلك! بالنسبة للاسم {subject}، هذه أقوى "
             "خمسة اقتراحات أعدتها لك:"]
    for i, (a, e, why) in enumerate(NAMES, 1):
        lines.append(f"{i}. «{a}» ({e}) — {why}")
    lines.append("وإن أحببت اتجاهاً آخر — أسماء علمية باردة "
                 "(سديم، مدار)، أو ودودة قريبة (حكمة، رفيق) — قل "
                 "لي الاتجاه والسوق المستهدف وأعطيك قائمة جديدة "
                 "فوراً.")
    return "\n".join(lines)


def _has(qn: str, *needles: str) -> bool:
    return any(n in qn for n in needles)


def classify(question: str) -> dict:
    """Detect dialog intents (a SET, not one winner) with hard
    guards against false positives: creator patterns require a
    reference to the assistant; greeting requires no question
    word; naming requires request markers + subject. Errors:
    none."""
    qn = _norm(question)
    ar = ar_ratio(question) >= 0.3
    raw = question.strip().lower()
    qn_sp = f" {qn} "  # word-boundary containment (بايثون vs باي!)
    intents = []
    # identity: who are you / what are you / your name - short
    # needles REQUIRE word boundaries ("ما انت" must not fire on
    # "ما انتظر")
    if (_has(qn, "عارف نفسك", "عرف بنفسك", "ما اسمك", "ما هويتك",
             "من سينورا", "you are who", "who are you",
             "what are you", "your name")
            or " من انت " in qn_sp or " ما انت " in qn_sp
            or " من ذا " in qn_sp):
        intents.append("identity")
    # fix16: identity paraphrases - "من بناك/صنعك/مبرمجك/طورك/كتبك"
    # rule-based possessive-ك matching over stem groups (no networks)
    if "identity" not in intents and _identity_hit is not None \
            and _identity_hit(question):
        intents.append("identity")
    # creator: من بنىك/بناك/طورك/صنعك - the bare "بناك/بنىك" forms
    # are inherently assistant-directed (the ك = you); the open
    # "من بنى ..." forms require an assistant reference so factual
    # questions ("من الذي بنى الأهرامات؟") stay factual.
    if (_has(qn, "من بناك", "من بنىك", "من طورك", "من صنعك",
             "من برمجك", "بنيتك", "طورك", "بناك", "بنىك",
             "who built you", "who created you", "who made you",
             "who developed you")
            or (_has(qn, "من بنى", "من طور", "من الذي بنى",
                     "من الذي طور", "من الذي صمم")
                and ("انت" in qn or "سينورا" in qn))):
        intents.append("creator")
    # becoming: ماذا تكون / ما ستكون
    if ("ماذا تكون" in qn and ("انت" in qn or "سينورا" in qn)):
        intents.append("becoming")
    # capability
    if (_has(qn, "ماذا تستطيع", "بتعمل ايه", "ما قدراتك",
             "ماذا تعرف ان تعمل", "what can you do",
             "capabilit")):
        intents.append("capability")
    # naming help: request markers + subject markers
    if ar and _has(qn, "ابتكار اسم", "اقتراح اسم", "اقترح اسم",
                   "اسمة", "تسميه", "اسم شركت", "اسم لشركت",
                   "اسم لمشروع", "اسم لمنتج", "اسم لتطبيق",
                   "اختر اسم", "افكر في اسم"):
        intents.append("naming")
    if (not ar and _has(raw, "name for my company",
                        "name my company", "name for my startup",
                        "company name ideas",
                        "suggest a name")):
        intents.append("naming")
    # writing (fix13 narrowed): SELF-promo only ("يروج لك"، "عن
    # نفسك"، "كنموذج") - generic copy requests (ads for a product,
    # descriptions...) go to the skills_core creativity engine.
    if (_has(qn, "اكتب", "صمم لي", "انشئ لي", "اعمل لي")
            and _has(qn, "اعلان", "اعلاني", "نص", "منشور")
            and _has(qn, "يروج لك", "روج لك", "عن نفسك", "كنموذج",
                     "لنفسك", "تعريف بنفسك", "اد عليك")):
        intents.append("writing")
    # greeting only when it is a greeting, not a question
    first = qn.split(" ")[0] if qn else ""
    if first in _GREET and not any(w in qn for w in _QWORDS
                                   if w in ("من", "ما", "كيف",
                                            "هل", "لماذا", "متى",
                                            "اين", "ماذا")):
        intents.append("greeting")
    if (_has(qn, "كيف حالك", "شخبارك", "ازيك", "عامل ايه",
             "how are you")):
        intents.append("howareyou")
    if _has(qn, "شكرا", "مشكور", "thanks", "thank you"):
        intents.append("thanks")
    # farewell with word boundaries: "باي" alone, NEVER inside
    # "بايثون" (measured: 17 python questions answered goodbye)
    if (_has(qn, "مع السلامه", "وداعا", "goodbye")
            or " باي " in qn_sp or " bye " in qn_sp):
        intents.append("farewell")
    return {"intents": intents, "ar": ar}


def try_answer(question: str) -> dict | None:
    """Direct smart reply for dialog intents; None when the
    question is factual and must go through the retrieval
    pipeline. Errors: none."""
    t0 = time.time()
    c = classify(question)
    intents = c["intents"]
    if not intents:
        return None
    ar = c["ar"]
    parts = []
    if "greeting" in intents and len(intents) == 1:
        parts.append("مرحباً. أنا سينورا؛ أجيب بما أعلم بدقة، "
                     "وأعترف بحدود ما لا أعلم. اسأل.")
    if "identity" in intents or "creator" in intents \
            or "becoming" in intents:
        parts.append(_identity_reply(
            "creator" in intents, "becoming" in intents,
            "capability" in intents or not intents,
            ar))
    if "capability" in intents and "identity" not in intents:
        parts.append(CAPAB_AR)
    if "naming" in intents:
        parts.append(_naming_reply(question, ar))
    if "writing" in intents:
        parts.append(AD_EN if not ar else AD_AR)
    if "howareyou" in intents:
        parts.append("بكامل قدراتي؛ كل نواة فيّ مقاسة وخضراء، "
                     "ولا شيء معلق. ما الذي تحتاجه؟")
    if "thanks" in intents:
        parts.append("العفو. الدقة واجب، لا فضل.")
    if "farewell" in intents:
        parts.append("إلى اللقاء. المعرفة تُبنى سؤالاً بعد سؤال.")
    if not parts:
        return None
    return {"mode": "dialog",
            "reply": "\n\n".join(p for p in parts if p),
            "intents": intents,
            "http_calls": 0,
            "latency_s": round(time.time() - t0, 2)}
