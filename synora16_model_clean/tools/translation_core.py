"""translation_core (rebuilt fix19 - the original source was lost in
a sandbox recycle; this rebuild matches the measured API webmind
consumes exactly):
  - Lexicon: persistent bilingual pair store (package-root
    lexicon_pairs.jsonl), attributes a2e/e2a dicts, .learn(en, ar,
    via=...), .en2ar(term, budget) with the TWO Wikidata gates
    (containment + sitelinks>=5), budget decremented in place.
  - gloss_translate(text, lex, budget) -> (glossed, coverage):
    direction auto-detected per token (Arabic -> en label, Latin ->
    ar label), unknown tokens kept verbatim.
  - lexicon_stats() -> {"pairs": N}.
Errors: none (every network call inside try)."""
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_LEX_PATH = os.path.join(os.path.dirname(_HERE),
                         "lexicon_pairs.jsonl")
_AR = re.compile(r"[\u0600-\u06FF]{3,}")
_EN = re.compile(r"[A-Za-z]{3,}")
_STATS = {"pairs": 0}

_AR_DIG = {"٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
           "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9"}


def _norm(w: str) -> str:
    w = w.translate(str.maketrans(_AR_DIG, "0123456789"))
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ة", "ه"),
                 ("ى", "ي"), ("ؤ", "و"), ("ئ", "ي")):
        w = w.replace(a, b)
    return re.sub(r"[\u064B-\u0652]", "", w).strip().lower()


class Lexicon:
    """Persistent learned bilingual lexicon (AR<->EN pairs)."""

    def __init__(self):
        self.a2e: dict = {}
        self.e2a: dict = {}
        self._via: dict = {}
        if os.path.isfile(_LEX_PATH):
            try:
                with open(_LEX_PATH, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = json.loads(line)
                        except Exception:
                            continue
                        ar, en = r.get("ar"), r.get("en")
                        if ar and en:
                            self.a2e[ar] = en
                            self.e2a[_norm(en)] = ar
                            self._via[ar] = r.get("via", "learned")
            except Exception:
                pass
        _STATS["pairs"] = len(self.a2e)

    def learn(self, en: str, ar: str, via: str = "learned") -> None:
        if not en or not ar:
            return
        self.a2e[ar] = en
        self.e2a[_norm(en)] = ar
        self._via[ar] = via
        _STATS["pairs"] = len(self.a2e)
        try:
            with open(_LEX_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ar": ar, "en": en, "via": via},
                                   ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _wikidata(self, query: str, lang: str,
                  budget: list) -> dict | None:
        if budget[0] <= 0:
            return None
        budget[0] -= 1
        try:
            url = ("https://www.wikidata.org/w/api.php?action="
                   "wbsearchentities&format=json&language=" + lang +
                   "&limit=5&search=" +
                   urllib.parse.quote(query))
            req = urllib.request.Request(
                url, headers={"User-Agent": "SynoraLex/1.0"})
            with urllib.request.urlopen(req, timeout=6) as r:
                data = json.loads(r.read().decode("utf-8"))
            for it in (data.get("search") or []):
                qid = it.get("id")
                if not qid:
                    continue
                ent = self._entity(qid)
                if not ent:
                    continue
                labels = ent.get("labels", {})
                aliases = ent.get("aliases", {})
                sitelinks = len(ent.get("sitelinks", {}) or {})
                ars = [labels.get("ar", {}).get("value", "")]
                ars += [a.get("value", "") for a in
                        aliases.get("ar", [])]
                ens = [labels.get("en", {}).get("value", "")]
                ens += [a.get("value", "") for a in
                        aliases.get("en", [])]
                ars = [a for a in ars if a]
                ens = [e for e in ens if e]
                if sitelinks < 5:
                    continue
                yield_q = (lang == "en" and
                           any(_norm(query) in _norm(a)
                               for a in ars)) or \
                          (lang == "ar" and
                           any(_norm(query) in _norm(e)
                               for e in ens))
                if not yield_q:
                    continue
                if lang == "en":
                    en = ens[0] if ens else None
                    ar = ars[0] if ars else None
                else:
                    en = ens[0] if ens else None
                    ar = ars[0] if ars else None
                if en and ar:
                    return {"en": en, "ar": ar}
        except Exception:
            return None
        return None

    def _entity(self, qid: str) -> dict | None:
        try:
            url = ("https://www.wikidata.org/w/api.php?action="
                   "wbgetentities&format=json&props=labels|"
                   "aliases|sitelinks&ids=" + qid)
            req = urllib.request.Request(
                url, headers={"User-Agent": "SynoraLex/1.0"})
            with urllib.request.urlopen(req, timeout=6) as r:
                data = json.loads(r.read().decode("utf-8"))
            return (data.get("entities") or {}).get(qid)
        except Exception:
            return None

    def en2ar(self, term: str, budget: list) -> str | None:
        """EN term -> AR label via Wikidata; learn + return."""
        if _norm(term) in self.e2a:
            return self.e2a[_norm(term)]
        hit = self._wikidata(term, "ar", budget)
        if hit:
            self.learn(hit["en"], hit["ar"], via="wikidata_en2ar")
            return hit["ar"]
        return None

    def ar2en(self, term: str, budget: list) -> str | None:
        """AR term -> EN label via Wikidata; learn + return."""
        if term in self.a2e:
            return self.a2e[term]
        hit = self._wikidata(term, "en", budget)
        if hit:
            self.learn(hit["en"], term, via="wikidata_ar2en")
            return hit["en"]
        return None


def gloss_translate(text: str, lex: Lexicon,
                    budget: list) -> tuple[str, float]:
    """Token-wise gloss in the detected direction per token.
    Returns (glossed_text, coverage over content tokens).
    Errors: none."""
    out, content, translated = [], 0, 0
    for tok in re.split(r"(\s+)", text):
        if not tok.strip():
            out.append(tok)
            continue
        m_ar = _AR.fullmatch(re.sub(r"[^\w\u0600-\u06FF]", "", tok))
        m_en = _EN.fullmatch(re.sub(r"[^\w\u0600-\u06FF]", "", tok))
        if m_ar:
            content += 1
            en = lex.a2e.get(tok)
            if en is None:
                en = lex.ar2en(tok.strip("،.؛:!?()«»"), budget)
            if en:
                translated += 1
                out.append(en)
            else:
                out.append(tok)
        elif m_en:
            content += 1
            ar = lex.e2a.get(_norm(tok))
            if ar is None:
                ar = lex.en2ar(tok.strip("،.؛:!?()«»"), budget)
            if ar:
                translated += 1
                out.append(ar)
            else:
                out.append(tok)
        else:
            out.append(tok)
    cov = translated / content if content else 0.0
    return ("".join(out), round(cov, 2))


def lexicon_stats() -> dict:
    return {"pairs": _STATS["pairs"]}
