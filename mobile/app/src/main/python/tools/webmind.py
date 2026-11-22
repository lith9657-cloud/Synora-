"""WebMind v2 (fix10): web knowledge + cross-language gloss
translation + learned memory. Numeric only - no neural parts, no
hand-coded routing lists (scopes are registry config scored with
BM25 overlaps; the bilingual lexicon is LEARNED data, see
translation_core.py).

Pipeline per question:
  1. MEMORY FIRST (containment scoring, >=2 tokens, tau 0.25) -
     zero network, provenance carried.
  2. ROUTE numerically: language ratio (AR/EN) + BM25 overlap of
     question tokens against each registry source's scope text.
     Cross-language sources carry translate:true and are chosen
     when their domain scores best (e.g. Arabic MEDICAL ->
     MedlinePlus EN, the richest coverage, then gloss-translated).
  3. FETCH (search + extract; RSS titles for news; PyPI JSON for
     code libraries). Cross-language: the search query is built by
     translating content tokens via the learning lexicon
     (Wikidata AR->EN labels on demand, cached forever).
  4. EXTRACT + rank sentences (overlap + >=2 gate) on the SOURCE
     language, compress (strip parens/citations).
  5. TRANSLATE gloss-wise when needed (term_coverage measured,
     reported per answer).
  6. LEARN: write to memory (answer language) + lexicon growth.

Honest limits (measured): gloss translation is term-faithful but
not fluent paraphrase (fix7/8 documented); marketing-library
sources need a licensed provider (documented, not simulated);
network rate limits degrade gracefully.

Errors: none (degrade to honest no_answer)."""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from translation_core import Lexicon, gloss_translate, lexicon_stats
from dialog_core import try_answer as _dialog_try
from skills_core import try_skill as _skill_try

_LM = None


def _lm():
    """Lazy NumericLM singleton (loads the 2.1 MB checkpoint once;
    grows via lm_feed without restarting). Errors: none (missing
    checkpoint -> None -> gloss fallback)."""
    global _LM
    if _LM is None:
        try:
            from numeric_lm import NumericLM
            import os
            m = NumericLM()
            # portable (fix14): package layout keeps the
            # checkpoint in lm/, the dev sandbox in data/
            for p in (f"{OUTD}/data/lm_ar_trigram.tsv.gz",
                      f"{OUTD}/lm/lm_ar_trigram.tsv.gz"):
                if os.path.isfile(p):
                    m.load(p)
                    break
            else:
                raise FileNotFoundError("lm checkpoint")
            _LM = m
        except Exception:
            _LM = False
    return _LM or None

def _pkg_root() -> str:
    """Portable layout resolution (fix14): when this file lives
    in a package tools/ dir, the package root carries the state
    (memory/lexicon/lm); in the dev sandbox the build dir is
    used. WEBMIND_OUTD overrides both. Errors: none."""
    import os
    env = os.environ.get("WEBMIND_OUTD")
    if env:
        return env
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isfile(os.path.join(root, "webmind_memory.jsonl")):
        return root
    return "/home/z/my-project/build/synora_final"


OUTD = _pkg_root()
MEM_PATH = f"{OUTD}/webmind_memory.jsonl"
UA = {"User-Agent": "SynoraWebMind/1.0 (research demo; "
                    "contact: research@example.org)"}
TAU_MEM = 0.55  # v1 0.25: template questions ("ما احدث اخبار X؟")
                # carry ~50% shared tokens with ANY news entry and
                # let tech memory answer politics (measured in
                # phase-2 run 1); 0.42 keeps real replays (0.5-0.6)
                # and rejects template collisions (0.4)
_AR = re.compile(r"[\u0600-\u06FF]")
_EN = re.compile(r"[A-Za-z]")
_TAG = re.compile(r"<[^>]+>")
_PAREN = re.compile(r"\([^)]*\)")
_CITE = re.compile(r"\[[0-9]+\]|\[note[^\]]*\]")
_SENT = re.compile(r"(?<=[.!؟?])\s+|\n+")


def content_tokens(text: str, deep: bool = False) -> list[str]:
    """WebMind tokenizer: Arabic IR normalization + punctuation
    stripping (the U+061F glue bug, measured in fix10), optionally
    DEEP morphology (stem_ar) for retrieval scoring. Lexicon
    lookups and cross-language queries keep raw forms.
    Errors: none."""
    out = []
    for t in re.findall(r"[\w\u0600-\u06FF]+", text):
        t = t.strip(".,!?؛:«»()\"'؟؟—ـَُِّْ")
        if len(t) < 3:
            continue
        t = t.lower()
        t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
        t = t.replace("ة", "ه").replace("ى", "ي").replace("ـ", "")
        if t.startswith("ال") and len(t) >= 5:
            t = t[2:]
        if deep:
            t = stem_ar(t)
        out.append(t)
    return out


_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")
_SUFFIXES = ("هما", "كما", "هن", "ها", "هم", "كم", "نا",
             "ات", "ون", "ين", "ان", "يه", "تي")


def stem_ar(t: str) -> str:
    """Deep Arabic morphology: one article+conjunction prefix and
    one possessive/plural suffix per token, with a >=3-char stem
    floor. Retrieval only; measured in evals_data/
    morph_retrieval.json (MRR before/after). Errors: none."""
    for p in _PREFIXES:
        if t.startswith(p) and len(t) - len(p) >= 3:
            t = t[len(p):]
            break
    for s in _SUFFIXES:
        if t.endswith(s) and len(t) - len(s) >= 3:
            t = t[:-len(s)]
            break
    return t


def ar_ratio(text: str) -> float:
    chars = [c for c in text if c.strip()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if _AR.match(c)) / len(chars)


def _get(url: str, timeout: int = 20) -> str:
    """Polite fetch: 0.6 s spacing, one 403/429 retry with backoff.
    Errors: raises on persistent failure (callers degrade)."""
    time.sleep(0.6)
    for attempt in (1, 2):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            if attempt == 1 and ("403" in str(e) or "429" in str(e)):
                time.sleep(2.0)
                continue
            raise


def compress(sent: str, cap: int = 240) -> str:
    s = _CITE.sub("", _PAREN.sub("", sent))
    s = re.sub(r"\s+", " ", s).strip(" .;:-")
    return s[:cap].rsplit(" ", 1)[0] + "." if s else ""


def split_sents(text: str) -> list[str]:
    out = []
    for s in _SENT.split(text):
        s = s.strip()
        if 45 <= len(s) <= 900 and not s.startswith(("(", "*")):
            out.append(s)
    return out


# fix11e/f: time-question machinery (متى/when questions must be
# answered with the DATED sentence, and the date itself - usually
# in parentheses - must survive compression)
_QSTOP = {"متي", "من", "ما", "ماذا", "هل", "كيف", "اين", "لماذا",
          "ذي", "هو", "هي", "when", "what", "where", "why", "how",
          "who", "was", "were", "is", "are", "the"}
_MONTHYEAR_RE = re.compile(
    r"(?:\d{1,2}\s)?(?:يناير|فبراير|مارس|ابريل|أبريل|مايو|يونيو|"
    r"يوليو|اغسطس|أغسطس|سبتمبر|اكتوبر|أكتوبر|نوفمبر|ديسمبر)\s+\d{4}"
    r"|(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+\d{1,2},?\s+\d{4}")
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


def compress_keep(sent: str, cap: int = 320) -> str:
    """compress() variant that KEEPS parentheses - birth dates live
    inside them ('من مواليد 14 يونيو 1946'); a WHEN-answer must not
    strip its own date. Errors: none."""
    s = _CITE.sub("", sent)
    s = re.sub(r"\s+", " ", s).strip(" .;:-")
    return s[:cap].rsplit(" ", 1)[0] + "." if s else ""


def _time_q(question: str) -> bool:
    """WHEN-question detector (متى/when/تاريخ). Errors: none."""
    return (any(w in content_tokens(question) for w in
                ("متي", "تاريخ"))
            or "when" in question.lower())


# ─────────────────────── source registry v2 ───────────────────────
REGISTRY = {
    "wiki_ar": {
        "lang": "ar", "kind": "general",
        "scope": ("المعلومات العامة المعرفة التاريخ الجغرافيا العلوم "
                  "التقنية الأشخاص المدن الأحداث المفاهيم العلم"),
    },
    "wiki_en": {
        "lang": "en", "kind": "general",
        "scope": ("general knowledge history geography science "
                  "technology people places concepts"),
    },
    "medline_en": {
        "lang": "en", "kind": "medical", "translate": True,
        "scope_ar": ("المرض الامراض الاعراض العلاج الدواء الطب الصحة "
                     "التشخيص المريض الوباء الوراثة"),
        "scope": ("health medicine disease symptoms treatment drugs "
                  "diagnosis medical topics illness therapy"),
        "search": ("https://wsearch.nlm.nih.gov/ws/query?db="
                   "healthTopics&term={q}&retmax=3"),
    },
    "news_ar": {
        "lang": "ar", "kind": "news",
        "scope": ("الاخبار العاجلة السياسة الاقتصاد الحوادث التقرير "
                  "القضايا الاحداث"),
        "rss": "https://feeds.bbci.co.uk/arabic/rss.xml",
    },
    "news_tech_en": {
        "lang": "en", "kind": "news", "translate": True,
        "scope": ("technology news gadgets software hardware "
                  "computers phones startups"),
        "scope_ar": ("التقنية الاخبار التقنية الهواتف الحاسوب "
                     "البرمجيات الذكاء الاصطناعي"),
        "rss": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    },
    "news_sport_en": {
        "lang": "en", "kind": "news", "translate": True,
        "scope": ("sport football soccer cricket tennis olympics "
                  "match league scores"),
        "scope_ar": ("الرياضة الرياضية المباراة الكرة الدوري "
                     "الاولمبياد الاخبار الرياضة"),
        "rss": "https://feeds.bbci.co.uk/sport/rss.xml",
    },
    "news_politics_en": {
        "lang": "en", "kind": "news", "translate": True,
        "scope": ("politics government election parliament policy "
                  "diplomacy international relations"),
        "scope_ar": ("السياسة الانتخابات البرلمان الحكومة الدبلوماسية "
                     "العلاقات الدولية الاخبار السياسية"),
        "rss": "https://feeds.bbci.co.uk/news/world/rss.xml",
    },
    "pypi_en": {
        "lang": "en", "kind": "code", "translate": True,
        "scope": ("python package library module pip install code "
                  "programming api documentation"),
        "scope_ar": ("المكتبة المكتبات البرمجية البايثون الحزمة "
                     "البرمجة الكود"),
        "search": "https://pypi.org/pypi/{q}/json",
    },
}


def wiki_api(lang: str, action: str) -> str:
    return ("https://{l}.wikipedia.org/w/api.php?action={a}&format="
            "json&redirects=1".format(l=lang, a=action))


def route(question: str) -> tuple[str, float]:
    """Numeric routing: language ratio + BM25-style overlap of the
    question's tokens with each source's scope text in the
    question's language. Cross-language sources (translate:true)
    are eligible when their domain wins - they translate after.
    Errors: none."""
    qt = set(content_tokens(question, deep=True))
    is_ar = ar_ratio(question) >= 0.3
    best, best_s = ("wiki_ar" if is_ar else "wiki_en"), 0.0
    for name, src in REGISTRY.items():
        scope = src.get("scope_ar" if is_ar else "scope", "")
        if not scope:
            continue
        st = set(content_tokens(scope, deep=True))
        s = len(qt & st) / max(len(qt), 1)
        if s > best_s:
            best, best_s = name, s
    return best, best_s


def cross_query(question: str, target_lang: str,
                lex: Lexicon, budget: list[int]) -> str:
    """Source-language search query: only ENTITY-backed terms (a
    Wikidata label exists) - word-soup queries were measured to
    break the medical source. Max 4 terms. Errors: none."""
    toks = content_tokens(question)
    raws = [t for t in re.findall(r"[\u0600-\u06FF]{3,}", question)]
    out = []
    for t in sorted(set(toks), key=len, reverse=True):
        if len(out) >= 4:
            break
        if target_lang == "en":
            raw = next((r for r in raws
                        if content_tokens(r) and content_tokens(r)[0]
                        in (t,)), None)
            en = lex.a2e.get(t) or _wikidata_ar2en(
                t, lex, budget, raw=raw)
            if en:
                out.append(en)
        else:
            ar = lex.e2a.get(t)
            if ar:
                out.append(ar)
    return " ".join(out[:4]) or " ".join(toks[:4]) or question[:60]


def _wikidata_ar2en(term: str, lex: Lexicon, budget: list[int],
                    raw: str | None = None) -> str | None:
    """AR term -> EN label via Wikidata with TWO numeric gates:
    (1) CONTAINMENT - the normalized term must appear in the
    entity's AR labels/aliases; (2) SITELINKS >= 5 language wikis -
    real concepts are multilingual, obscure junk items are not
    (measured poison: a stub labeled zahaimar with 0 sitelinks).
    Searches the raw surface form first, normalized fallback.
    Errors: none."""
    if budget[0] <= 0:
        return None
    budget[0] -= 1
    best: tuple[int, str] | None = None
    for q in dict.fromkeys([raw or term, term]):
        try:
            u = ("https://www.wikidata.org/w/api.php?action="
                 "wbsearchentities&search={q}&language=ar&format="
                 "json&limit=3".format(q=urllib.parse.quote(q)))
            d = json.loads(_get(u))
        except Exception:
            return None
        for h in d.get("search", []):
            try:
                eid = h["id"]
                u2 = ("https://www.wikidata.org/w/api.php?action="
                      "wbgetentities&ids={i}&props=labels|aliases|"
                      "sitelinks&languages=ar|en&format=json"
                      .format(i=eid))
                d2 = json.loads(_get(u2))
                ent = d2.get("entities", {}).get(eid, {})
                ns = len(ent.get("sitelinks", {}))
                if ns < 5:
                    continue
                ar_names = [ent.get("labels", {}).get("ar", {})
                            .get("value", "")]
                ar_names += [a.get("value", "") for a in
                             ent.get("aliases", {}).get("ar", [])]
                ar_words = [w for n in ar_names
                            for w in content_tokens(n)]
                # word-level containment (substring matched junk)
                if term not in ar_words:
                    continue
                lbl = (ent.get("labels", {}).get("en", {})
                       .get("value"))
                # among gate-passing candidates prefer the most
                # multilingual one (disease beats its eponymous
                # doctor - measured: alois alzheimer won before)
                if lbl and (best is None or ns > best[0]):
                    best = (ns, lbl)
            except Exception:
                continue
        if d.get("search"):
            break
    if best is not None:
        lex.learn(best[1], term, via="wikidata_ar2en")
        return best[1]
    return None


def _wiki_search_extract(lang: str, query: str,
                         qtok: set | None = None) -> tuple[str, str]:
    """Search + extract. fix11e: pick the hit whose TITLE matches
    the question's tokens - the search API sometimes ranks an
    unrelated article first (measured: "متى ولد الرئيس ترامب؟" hit
    "ألعاب الفيديو في 2023"). Errors: none (empties on failure)."""
    q = urllib.parse.quote(query)
    data = json.loads(_get(wiki_api(lang, "query")
                           + "&list=search&srlimit=5&srsearch=" + q))
    hits = data.get("query", {}).get("search", [])
    if not hits:
        return "", ""
    ti = 0
    if qtok:
        for i, h in enumerate(hits):
            if set(content_tokens(h.get("title", ""), deep=True)) \
                    & qtok:
                ti = i
                break
    title = urllib.parse.quote(hits[ti]["title"])
    ex = json.loads(_get(
        wiki_api(lang, "query")
        + "&prop=extracts&exintro&explaintext&titles=" + title))
    text = ""
    for p in ex.get("query", {}).get("pages", {}).values():
        text = p.get("extract", "")
    if not text.strip():
        try:
            sm = json.loads(_get(
                "https://{l}.wikipedia.org/api/rest_v1/page/"
                "summary/{t}".format(l=lang, t=title)))
            text = sm.get("extract", "")
        except Exception:
            pass
    return hits[ti].get("title", ""), text


def _medline_search_extract(query: str) -> tuple[str, str]:
    q = urllib.parse.quote(query[:120])
    xml = _get(REGISTRY["medline_en"]["search"].format(q=q))
    docs = re.findall(r'<document[^>]*url="([^"]+)"[^>]*>', xml)
    if not docs:
        return "", ""
    url = docs[0]
    html = _get(url)
    m = re.search(r'id="topic-summary"[^>]*>(.*?)</div>', html, re.S)
    if m and len(m.group(1)) > 300:
        body = _TAG.sub(" ", m.group(1))
    else:
        body = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ",
                      html, flags=re.S | re.I)
        body = _TAG.sub(" ", body)
    return url, re.sub(r"\s+", " ", body)[:6000]


def _rss_titles(url: str) -> list[dict]:
    """Parse RSS items (title/link/date) - numeric regex. Errors:
    none (empty on failure)."""
    try:
        xml = _get(url)
    except Exception:
        return []
    items = []
    for it in re.findall(r"<item>(.*?)</item>", xml, re.S)[:8]:
        t = re.search(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?"
                      "</title>", it, re.S)
        l = re.search(r"<link>(.*?)</link>", it)
        d = re.search(r"<pubDate>(.*?)</pubDate>", it)
        items.append({"title": (t.group(1).strip() if t else ""),
                      "link": (l.group(1).strip() if l else ""),
                      "date": (d.group(1).strip() if d else "")})
    return items


def _pypi_summary(question: str) -> tuple[str, str]:
    """Package name = longest Latin token; official PyPI JSON API.
    Returns (pkg_url, summary). Errors: none (empties on failure)."""
    lat = re.findall(r"[A-Za-z][A-Za-z0-9_.-]{2,}", question)
    if not lat:
        return "", ""
    pkg = max(lat, key=len).strip(".-").lower()
    try:
        d = json.loads(_get(
            REGISTRY["pypi_en"]["search"].format(
                q=urllib.parse.quote(pkg))))
        info = d.get("info", {})
        return (info.get("project_url", ""),
                (info.get("summary") or "") + ". "
                + "latest version: "
                + str(info.get("version", "")))
    except Exception:
        return "", ""


def fetch_source(name: str, query: str,
                 question: str = "") -> tuple[str, str, int]:
    """(url_or_title, page_text, calls) with retry-on-empty.
    Errors: none (empties on failure)."""
    try:
        if name.startswith("wiki"):
            qtok = (set(content_tokens(question, deep=True))
                    if question else None)
            title, text = _wiki_search_extract(name[-2:], query, qtok)
            for _ in range(2):
                if text.strip():
                    break
                time.sleep(2.0)
                title, text = _wiki_search_extract(name[-2:], query,
                                                   qtok)
            got = (text.strip()
                   and bool(set(content_tokens(title, deep=True))
                            & qtok))
            if qtok and not got:
                # fix11e retry: the page is empty OR its title shares
                # NOTHING with the question (wrong article ranked
                # first - measured on ترامب): drop interrogative/
                # function words, then fall back to the entity token
                # alone.
                q2 = " ".join(t for t in content_tokens(question)
                              if t not in _QSTOP)[:120]
                cands = [q2, q2.split()[-1] if q2 else ""]
                for cand in cands:
                    if not cand.strip() or cand == query:
                        continue
                    t2, x2 = _wiki_search_extract(name[-2:], cand,
                                                  qtok)
                    if x2.strip():
                        title, text = t2, x2
                        break
            return title, text, 2
        if name == "medline_en":
            title, text = _medline_search_extract(query)
            for _ in range(2):
                if text.strip():
                    break
                time.sleep(2.0)
                title, text = _medline_search_extract(query)
            return title, text, 2
        if name.startswith("news_"):
            items = _rss_titles(REGISTRY[name]["rss"])
            text = " ".join(i["title"] + ". " for i in items
                            if i["title"])
            return REGISTRY[name]["rss"], text, 1
        if name == "pypi_en":
            # the package name must come from the USER's original
            # wording (the translated query renames packages:
            # requests -> request -> library, measured)
            url, text = _pypi_summary(question or query)
            return url, text, 1
        return "", "", 0
    except Exception as e:
        print(f"  fetch failed: {e}", flush=True)
        return "", "", 1


def rank_sents(question: str, text: str, k: int = 3) -> list[str]:
    """Numeric selection: overlap + >=2-token gate. Errors: none."""
    sents = split_sents(text)
    if not sents:
        return []
    qt = set(content_tokens(question, deep=True))
    scored = []
    for s in sents:
        ov = qt & set(content_tokens(s, deep=True))
        if len(ov) >= 2:
            scored.append((len(ov), s))
    scored.sort(key=lambda kv: -kv[0])
    return [compress(s) for _, s in scored[:k]]


# ─────────────────────── learned memory ───────────────────────
TTL_NEWS_S = 48 * 3600


def memory_hit(question: str, now: float | None = None
               ) -> tuple[dict | None, dict]:
    """Containment scoring (>=2 overlap tokens, tau 0.25) with a df
    gate (skip tokens shared by >= max(2, 60%) of entries) and a
    LIFECYCLE (phase-1 defect-7): news entries older than 48 h are
    EXPIRED (skipped and counted - stale headlines previously
    replayed as current), superseded entries are skipped, tokens
    deep-stemmed both sides. Errors: none."""
    now = now or time.time()
    expired = 0
    stats = {"expired_skipped": 0, "entries": 0}
    if not os.path.exists(MEM_PATH):
        return None, stats
    qt = content_tokens(question, deep=True)
    if not qt:
        return None, stats
    qset = set(qt)
    # df counts EVERY row (expired included, superseded excluded):
    # stale news entries still prove that ahdath/akhbar-style
    # words are templates - counting only fresh rows let a tech
    # entry answer a politics question (measured in phase-1 run 1)
    all_dsets = []
    for line in open(MEM_PATH, encoding="utf-8"):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("superseded_by"):
            continue
        d = set(content_tokens(
            e["q"] + " " + e.get("title", "") + " "
            + " ".join(e["sents"]), deep=True))
        if d:
            all_dsets.append((e, d))
    entries, dsets = [], []
    for e, d in all_dsets:
        if e.get("kind") == "news" and e.get("ts_epoch"):
            if now - e["ts_epoch"] > TTL_NEWS_S:
                expired += 1
                continue
        entries.append(e)
        dsets.append(d)
    n = len(entries)
    stats = {"expired_skipped": expired, "entries": n}
    if not n:
        return None, stats
    df: Counter = Counter()
    for _, d in all_dsets:
        for t in d:
            df[t] += 1
    # df gate: skip tokens shared by >= max(2, 60%) of entries -
    # always on, with a floor of 2 so single/dual entries still
    # reject generic words (ahdath/akhbar linked topics when
    # measured)
    import math as _m
    df_thr = max(2, _m.ceil(0.6 * n))
    best, best_sc = None, 0.0
    for e, d in zip(entries, dsets):
        dc = Counter(content_tokens(
            e["q"] + " " + e.get("title", "") + " "
            + " ".join(e["sents"]), deep=True))
        ov = {t for t in (qset & d) if df[t] < df_thr}
        if len(ov) < 2:
            continue
        sc = sum((dc[t] * 2.5) / (dc[t] + 1.5 * 1.0) for t in ov)
        sc /= len(qset)
        if sc > best_sc:
            best, best_sc = e, sc
    if best is not None and best_sc >= TAU_MEM:
        return best, stats
    return None, stats


def learn(question: str, url: str, title: str,
          sents: list[str], kind: str = "general") -> None:
    """Append to memory + SUPERSEDE (phase-1 defect-7): an existing
    entry whose question tokens overlap >=0.6 Jaccard with the new
    one is marked superseded_by (newest wins; the old row is kept
    for audit, never silently deleted). Errors: none."""
    e = {"q": question, "url": url, "title": title,
         "sents": sents, "kind": kind,
         "ts_epoch": round(time.time()),
         "learned_at": time.strftime("%Y-%m-%d %H:%M")}
    qtok = set(content_tokens(question))
    rows = []
    if os.path.exists(MEM_PATH):
        for line in open(MEM_PATH, encoding="utf-8"):
            try:
                old = json.loads(line)
            except Exception:
                continue
            if not old.get("superseded_by"):
                oset = set(content_tokens(old["q"]))
                if oset and qtok:
                    j = len(qtok & oset) / len(qtok | oset)
                    if j >= 0.6:
                        old["superseded_by"] = e["ts_epoch"]
            rows.append(old)
    rows.append(e)
    with open(MEM_PATH, "w", encoding="utf-8") as f:
        for old in rows:
            f.write(json.dumps(old, ensure_ascii=False) + "\n")


def answer_raw(question: str, k: int = 3,
               learn_mode: str = "fast", gen: bool = True) -> dict:
    """Main entry: memory -> route -> fetch -> extract -> GENERATE
    (numeric composition; defect-1) or translate -> learn, with
    measured metrics. gen=True composes a NOVEL reply with the
    anchor-hopping numeric generator over NumericLM, falling back
    to gloss when the LM is absent. learn_mode="fast" (default,
    phase-1 defect-3): the live path uses ONLY the cached lexicon
    (budget 0). teach()/teach_terms() perform learning separately.
    Errors: none."""
    t0 = time.time()
    budget = [0] if learn_mode == "fast" else [12]
    lex = Lexicon()
    lex0 = lexicon_stats()["pairs"]
    # fix11 dialog gate FIRST: identity/greeting/naming/thanks...
    # answer DIRECTLY and honestly - never routed to external
    # sources, never memorized (the measured Trump/Olympics
    # failure: an identity question routed into tech-news RSS).
    # fix19b: PERSONA FIRST — the dignified identity/etiquette
    # layer outranks the dialog memory (measured: cheerful memory
    # replies were outranking it)
    try:
        from skills_core import try_persona
        _pp = try_persona(question, t0)
        if _pp is not None:
            return _pp
    except Exception:
        pass
    dlg = _dialog_try(question)
    if dlg is not None:
        dlg["latency_s"] = round(time.time() - t0, 2)
        return dlg
    # fix13 capability gate: math COMPUTED exactly, code/facts from
    # the learnable KB, creativity composed with seeds - all BEFORE
    # memory/retrieval so a capability request never touches RSS.
    skl = _skill_try(question)
    if skl is not None:
        skl["mode"] = "skill"
        skl["latency_s"] = round(time.time() - t0, 2)
        return skl
    mem, mstats = memory_hit(question)
    if mem is not None:
        return {"mode": "memory",
                "reply": " ".join(mem["sents"]),
                "url": mem["url"], "title": mem["title"],
                "learned_at": mem.get("learned_at", ""),
                "memory_stats": mstats,
                "http_calls": 0,
                "latency_s": round(time.time() - t0, 2)}
    # fix17: NEURAL UNDERSTANDING RESCUE - a question the gates,
    # skills and memory could not serve gets ONE confident
    # re-formulation (dialect/phrasing -> canonical, 20% neural
    # layer), then the SAME gates run again. The numeric core still
    # answers everything; the layer only routes. No confidence ->
    # the original question continues to retrieval untouched.
    _cand_q = question
    try:
        from understand_neural import canonicalize
        _und = canonicalize(question)
    except Exception:
        _und = None
    if _und is not None:
        _cq = _und["canonical"]
        # fix18: KB-kind rescue answers FROM THE ENTRY directly
        # (measured: rerunning the gates on the canonical text lost
        # dialect questions to sibling stem scores)
        if _und.get("kind") in ("code", "fact") \
                and _und.get("entry"):
            try:
                from skills_core import serve_entry
                _skl = serve_entry(_und["entry"],
                                   _und["kind"], t0,
                                   question=question)
                _skl["mode"] = "skill+neural"
                _skl["via"] = "neural_rescue_entry"
                _skl["understanding"] = _und
                _skl["latency_s"] = round(time.time() - t0, 2)
                return _skl
            except Exception:
                pass
        _dlg = _dialog_try(_cq)
        if _dlg is not None:
            _dlg["mode"] = "dialog+neural"
            _dlg["understanding"] = _und
            _dlg["latency_s"] = round(time.time() - t0, 2)
            return _dlg
        _skl = _skill_try(_cq)
        if _skl is not None:
            _skl["mode"] = "skill+neural"
            _skl["understanding"] = _und
            _skl["latency_s"] = round(time.time() - t0, 2)
            return _skl
        _mem2, _ms2 = memory_hit(_cq)
        if _mem2 is not None:
            return {"mode": "memory+neural",
                    "reply": " ".join(_mem2["sents"]),
                    "url": _mem2["url"], "title": _mem2["title"],
                    "understanding": _und,
                    "memory_stats": _ms2, "http_calls": 0,
                    "latency_s": round(time.time() - t0, 2)}
        _cand_q = _cq
    src, sc = route(_cand_q)
    srcl = REGISTRY[src]["lang"]
    qlang = "ar" if ar_ratio(question) >= 0.3 else "en"
    translated = srcl != qlang
    if translated:
        query = cross_query(_cand_q, srcl, lex, budget)
    else:
        query = (" ".join(content_tokens(_cand_q)[:6])
                 or question[:60])
    title, text, calls = fetch_source(src, query, _cand_q)
    if not text.strip():
        return {"mode": "no_answer",
                "reply": "لم أجد تغطية كافية في المصادر المسجلة الآن.",
                "source": src, "http_calls": calls,
                "latency_s": round(time.time() - t0, 2)}
    covs: list[float] = []
    facts_en: list[str] = []
    if src.startswith("news_"):
        items = _rss_titles(REGISTRY[src]["rss"])
        qt = set(content_tokens(question))
        chosen = []
        for it in items:
            ov = len(qt & set(content_tokens(it["title"])))
            chosen.append((ov, it))
        chosen.sort(key=lambda kv: -kv[0])
        # fix11 coherence gate: a news question with ZERO overlap
        # to every headline must not be answered with unrelated
        # headlines (the measured Trump/Olympics failure) - honest
        # no_answer instead, and nothing irrelevant is memorized.
        if not chosen or chosen[0][0] < 1:
            # fix11c: cross-language news (translate:true) carries
            # EN headlines, so zero title-overlap is EXPECTED for a
            # legit "اخبار التقنية؟" - accept when the question
            # genuinely matches the source's scope (routing
            # relevance), refuse only when NOTHING matches (the
            # measured Trump/Olympics failure).
            ov_scope = len(set(content_tokens(question, deep=True))
                           & set(content_tokens(
                               REGISTRY[src].get("scope_ar", "")
                               + " " + REGISTRY[src].get("scope", ""),
                               deep=True)))
            if ov_scope < 1:
                return {"mode": "no_answer",
                        "reply": ("لا علاقة لعناوين الأخبار الحالية "
                                  "بسؤالك، فلن أنقل لك عناوين "
                                  "بعيدة."),
                        "source": src, "http_calls": calls,
                        "latency_s": round(time.time() - t0, 2)}
        lines = []
        for ov, it in chosen[:4]:
            t = it["title"]
            if translated:
                t, c = gloss_translate(t, lex, budget)
                covs.append(c)
            lines.append(f"- {t} ({it['date'][:16]})")
        sents = lines
        facts_en = [it["title"] for _, it in chosen[:2]]
    else:
        if src == "pypi_en":
            # the official package summary IS the answer; the
            # sentence gate would reject such a short text
            sents = [text.strip().rstrip(".") + "."] \
                if len(text.strip()) > 20 else []
        else:
            sents = rank_sents(query if translated else _cand_q,
                               text, k=k)
            # fix11f: WHEN-questions must return the DATED sentence -
            # max-overlap ranking often picks a biography line
            # without the year (measured on ترامب); keep parentheses
            # so the date survives compression.
            if (_time_q(_cand_q) and not translated
                    and src.startswith("wiki")):
                qt_f = set(content_tokens(_cand_q, deep=True))
                best, best_n = None, -1
                for s in split_sents(text):
                    if not _MONTHYEAR_RE.search(s) \
                            and not _YEAR_RE.search(s):
                        continue
                    n = len(qt_f & set(content_tokens(s, deep=True)))
                    if n > best_n:
                        best, best_n = s, n
                if best is not None and best_n >= 1:
                    sents = [compress_keep(best)]
        if not sents:
            # fix11b: title-anchored fallback - if the found page
            # TITLE shares a token with the question, the page is
            # relevant even when sentence-overlap gating is strict
            # (morphology divergence like بنى vs بنيت); return its
            # opening as a definition instead of a bare no_answer.
            t_ov = len(set(content_tokens(_cand_q, deep=True))
                       & set(content_tokens(title, deep=True)))
            pool = split_sents(text)
            first = []
            if _time_q(question) and not translated:
                dated = [s for s in pool if _MONTHYEAR_RE.search(s)
                         or _YEAR_RE.search(s)]
                if dated:
                    first = [compress_keep(dated[0])]
            if not first:
                first = [compress(s) for s in pool[:2]]
            first = [s for s in first if s]
            if t_ov >= 1 and first:
                sents = first
            else:
                return {"mode": "no_answer",
                        "reply": "وجدت المصدر لكن لا جمل كافية تطابق السؤال.",
                        "source": src, "url": title,
                        "http_calls": calls,
                        "latency_s": round(time.time() - t0, 2)}
        facts_en = list(sents) if translated else []
        if translated:
            tr = []
            for s in sents:
                g, c = gloss_translate(s, lex, budget)
                tr.append(g)
                covs.append(c)
            sents = tr
    # defect-1: NUMERIC GENERATION - compose a novel Arabic reply
    # from the facts (anchor-hopping beam over the counting LM);
    # graceful fallback to the gloss reply when composition misses
    gen_meta = None
    if gen and facts_en:
        m = _lm()
        # fix11d: composition builds a FULL context index over the
        # LM; with the trained 5.17M-trigram checkpoint that index
        # OOM-killed the process (measured). Composition stays for
        # checkpoint sizes it can index; larger brains answer from
        # the extracted (gated) facts directly.
        if m and len(m.tri) <= 3_000_000:
            from numeric_gen import compose_answer
            try:
                ca = compose_answer(m, lex, facts_en, k=min(k, 2))
                if (ca["sentences"]
                        and ca["term_coverage"] >= 0.5):
                    gen_meta = {
                        "composed": True,
                        "n_sentences": len(ca["sentences"]),
                        "term_coverage": ca["term_coverage"],
                        "lm_score": round(sum(
                            s["lm_score"]
                            for s in ca["sentences"])
                            / len(ca["sentences"]), 2)}
                    sents = [s["text"] for s in ca["sentences"]]
            except Exception:
                gen_meta = {"composed": False,
                            "error": "fallback"}
    learn(_cand_q, title, title, sents,
          kind=REGISTRY.get(src, {}).get("kind", "general"))
    out = {"mode": "live",
           "reply": " ".join(sents),
           "source": src, "url": title,
           "http_calls": calls,
           "translated": bool(translated),
           "source_lang": srcl,
           "question_lang": qlang,
           "latency_s": round(time.time() - t0, 2),
           "lexicon_pairs": lexicon_stats()["pairs"],
           "lexicon_learned_this_call":
               lexicon_stats()["pairs"] - lex0}
    if covs:
        out["term_coverage"] = round(sum(covs) / len(covs), 2)
    if gen_meta:
        out["gen"] = gen_meta
    # defect-3 companion: surface the untranslated source terms so
    # the async teach_terms() can learn them AFTER the reply (the
    # model improves with use - lexicon pairs persist forever)
    if translated:
        left = []
        for s in sents:
            left += [t for t in re.findall(r"[A-Za-z]{4,}", s)]
        out["untranslated_terms"] = sorted(
            set(left), key=len, reverse=True)[:10]
    return out


def answer(question: str, k: int = 3,
           learn_mode: str = "fast", gen: bool = True) -> dict:
    """fix18 POLISHED ENTRY: every reply passes the deterministic
    output-polish layer (code fences + Arabic typography + structure
    with an auditable report) before it leaves the engine. The
    neural layer helps output upstream via candidate-coherence
    selection in skills_core (_neural_pick). The raw engine stays
    reachable as answer_raw for measured comparisons. Errors: none
    (polish failure -> raw reply)."""
    r = answer_raw(question, k=k, learn_mode=learn_mode, gen=gen)
    # phase-2 TWO-LAYER GATEWAY: when EVERY local layer declined,
    # the neural bridge answers - labeled honestly (mode="neural").
    # Identity is hard-protected upstream; offline keeps the
    # dignified local decline (nothing invented).
    if r.get("mode") == "no_answer":
        try:
            from neural_bridge import neural_fallback
            r = neural_fallback(question, r) or r
        except Exception:
            pass
    try:
        from polish_core import polish
        return polish(question, r)
    except Exception:
        return r


def teach_terms(terms: list[str]) -> dict:
    """SEPARATE learning pass part 2: learn AR labels for the
    UNTRANSLATED terms of a previous answer (Wikidata, containment
    + sitelinks gates). Out of the reply path; pairs persist, so
    the next question on the topic translates better - measured in
    the phase-1 demo as a before/after coverage delta.
    Errors: none."""
    t0 = time.time()
    lex = Lexicon()
    lex0 = lexicon_stats()["pairs"]
    budget = [10]
    for t in terms[:10]:
        lex.en2ar(t, budget)
    return {"learned_pairs": lexicon_stats()["pairs"] - lex0,
            "learn_seconds": round(time.time() - t0, 2),
            "remaining_budget": budget[0]}


def teach(question: str) -> dict:
    """SEPARATE learning pass (phase-1 defect-3): performs the
    wikidata AR->EN lookups and gloss learning for a question OUT
    of the reply path - call it after answering, or offline as a
    batch warmer. In production this runs as a background worker;
    here it is measured standalone so the reply path stays free of
    learning. Errors: none."""
    t0 = time.time()
    lex = Lexicon()
    lex0 = lexicon_stats()["pairs"]
    budget = [12]
    cross_query(question, "en", lex, budget)
    return {"learned_pairs": lexicon_stats()["pairs"] - lex0,
            "learn_seconds": round(time.time() - t0, 2),
            "remaining_budget": budget[0]}
