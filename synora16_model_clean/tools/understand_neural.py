"""understand_neural (fix17): THE UNDERSTANDING LAYER of Synora.

Composition decision (owner-approved, documented): 20% neural /
60% numeric / 10% symbolic. This module is the neural part plus the
small symbolic dialect map:

  - SYMBOLIC (10%): a closed dialect->MSA function-word map applied
    before embedding + number masking + confidence gates. Cheap,
    exact, auditable - only words present in the closed map are
    replaced, so MSA questions are untouched.
  - NEURAL (20%): an open-source multilingual paraphrase encoder
    (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2,
    Apache-2.0; replaces the measured-marbert choice after raw
    MARBERT pooled space showed total anisotropy 0.99), packed int8
    at runtime, refined with BERT-whitening (shrinkage k=256) + a
    contrastive head trained on 6000+ synthetic dialect->canonical
    pairs built from the project's own KBs. Its ONLY job: map ANY
    formulation/dialect of a question to the canonical question or
    KB entry the numeric core serves. It NEVER generates text.
  - NUMERIC (60%, elsewhere): skills_core (exact math), the KBs,
    memory, retrieval and generation stay the answer engine.

Scoring (measured): HYBRID = 0.5 whitened cosine + 0.5 head cosine.
Integration (additive): canonical_entry() runs BEFORE the stem-IDF
KB fallback (kills the measured stem hijacks); canonicalize()
rescues questions that fell through dialog/skills/memory. Learned
memory entries become new anchors automatically (the layer grows
with use). Errors: none (missing torch/files -> layer disabled,
engine unchanged)."""
from __future__ import annotations

import json
import os
import re
import sys

_TOOLS = os.path.dirname(os.path.abspath(__file__))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

# ── symbolic dialect map (the 10%: closed, auditable) ─────────
_DIALECT_MAP = {
    "عايز": "أريد", "عاوز": "أريد", "عايزة": "أريد",
    "محتاج": "أريد", "محتاجة": "أريد",
    "بدي": "أريد", "بدّي": "أريد",
    "أبغى": "أريد", "ابغى": "أريد", "أبي": "أريد",
    "ودي": "أريد", "نبي": "أريد",
    "أش": "أريد", "اش": "أريد",
    "شو": "ماذا", "وشو": "ماذا", "وش": "ماذا", "شنو": "ماذا",
    "إيه": "ماذا", "ايه": "ماذا", "إيش": "ماذا",
    "وين": "أين", "فين": "أين", "قوين": "أين",
    "إزاي": "كيف", "ازاي": "كيف", "شلون": "كيف", "لش": "كيف",
    "ليش": "لماذا", "ليه": "لماذا", "امتى": "متى",
    "بيطلع": "يساوي", "يطلع": "يساوي", "بيصير": "يساوي",
    "يصير": "يساوي", "يبقى": "يساوي",
    "زايد": "زائد", "كام": "كم",
    "بالمية": "بالمئة", "بالميه": "بالمئة",
    "فولدر": "مجلد", "باكدج": "مكتبة",
    "جيت": "git",
}
_MAP_RE = re.compile(
    "|".join(re.escape(k) for k in _DIALECT_MAP))


def map_dialect(question: str) -> str:
    """Symbolic fast pass: closed-map dialect function words to MSA.
    Errors: none."""
    return _MAP_RE.sub(lambda m: _DIALECT_MAP[m.group(0)], question)


_AR_DIG = {"٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
           "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9"}
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_FUNCT = set("""ما ماذا من هو هي في على عن كيف هل اين متى لماذا
يساوي كم يبقى اللي الذي التي ذا ياد تا و او ام ولا اكبر اصغر واحد
بس ياه لها فيها فيه بها بعد الا اذا كذلك شلون ازاي إزاي وين فين
ليش ليه كام امتى شنو شو وش عايز أبغى أبغي بدي أريد""".split())


def extract_numbers(question: str) -> list[str]:
    """Ordered western-digit numbers of the ORIGINAL question.
    Errors: none."""
    q = "".join(_AR_DIG.get(c, c) for c in question)
    return _NUM_RE.findall(q)


def mask_numbers(text: str) -> str:
    """Embedding-side normalization: every number -> "1" so the
    neural match is about WORDS not values. Errors: none."""
    q = "".join(_AR_DIG.get(c, c) for c in text)
    return _NUM_RE.sub("1", q)


def content_stems(text: str) -> set:
    try:
        from paraphrase_core import stem as _stem
    except Exception:
        _stem = lambda w: w  # noqa: E731
    out = set()
    for t in re.findall(r"[\u0600-\u06FFa-zA-Z0-9]+",
                        "".join(_AR_DIG.get(c, c) for c in text)):
        t = t.lower().replace("أ", "ا").replace("إ", "ا") \
            .replace("آ", "ا").replace("ة", "ه").replace("ى", "ي")
        st = _stem(t)
        if len(st) >= 3 and st not in _FUNCT:
            out.add(st)
    return out


def stems_shared(qs: set, as_: set) -> bool:
    """Engine-consistent stem matching (tok_match: prefix/suffix/
    equivalence), not raw set intersection - 'شعار' vs 'شعارا'
    must count as shared. Errors: none."""
    try:
        from paraphrase_core import tok_match as _tm
    except Exception:
        _tm = lambda a, b: a == b  # noqa: E731
    return any(_tm(q, a_) for q in qs for a_ in as_)


def _extract_subject(question: str) -> str:
    """Subject for creative canonicalization: after عن/لل/لـ from
    the ORIGINAL wording, tails stripped. Errors: none."""
    qd = re.sub(r"[\u064B-\u0652]", "", question)
    m = re.search(r"عن\s+(.+)", qd)
    rest = m.group(1) if m else ""
    if not rest:
        m2 = re.search(r"(?:شعار|وصف|قصة|قصه|دعوة|دعوه|شعر|"
                       r"افكار|أفكار)[ا]?\s+(.+)", qd)
        rest = m2.group(1) if m2 else ""
        rest = re.sub(r"^\s*(?:لـ|لل|ل)\s*", "", rest)
    rest = re.sub(r"[؟.!,؛:\s]+$", "", rest).strip()
    rest = re.sub(r"\s*(?:واكتبه|واكتبها|من فضلك|لو سمحت)+$",
                  "", rest).strip()
    return rest or "مشروعك"


_LAYER = None


def model_dir() -> str:
    """Neural assets dir. WEBMIND_NEURAL is an EXPLICIT override:
    a valid dir enables the layer, anything else set DISABLES it
    (measured: a set-but-invalid env silently re-enabled the layer
    and polluted the gate-mode measurement). No env: package
    neural/ > dev build dirs. Errors: none."""
    env = os.environ.get("WEBMIND_NEURAL")
    if env is not None:
        if env and os.path.isfile(env + "/anchors.json"):
            return env
        return "/nonexistent_neural_layer_disabled"
    p1 = os.path.join(os.path.dirname(_TOOLS), "neural")
    if os.path.isfile(os.path.join(p1, "anchors.json")):
        return p1
    for p in ("/home/z/my-project/build/neural_layer/MiniLM",
              "/home/z/my-project/build/neural_layer"):
        if os.path.isfile(os.path.join(p, "anchors.json")):
            return p
    return "/nonexistent_neural_layer_disabled"


def layer():
    """Lazy singleton; None -> layer disabled (engine unchanged).
    Errors: none."""
    global _LAYER
    if _LAYER is not None:
        return _LAYER or None
    try:
        import torch  # noqa: F401
        from marbert_encoder import BertExtractor
        import numpy as np
        md = model_dir()
        enc = BertExtractor(md)
        # phase-4: prefer the 4-bit pack, fall back to int8
        npz = md + "/marbert_int4a.npz"
        if not os.path.isfile(npz):
            npz = md + "/marbert_int4.npz"
        if not os.path.isfile(npz):
            npz = md + "/marbert_int8.npz"
        if os.path.isfile(npz):
            enc._load_int8(npz)
        else:
            enc._load_bin(md + "/pytorch_model.bin")
        L = {"enc": enc, "np": np}
        A = np.load(md + "/anchors_raw.npy")
        L["meta"] = json.load(open(md + "/anchors.json",
                                   encoding="utf-8"))
        w = np.load(md + "/whitening.npz")
        L["mu"], L["W"] = w["mu"].astype(np.float32), \
            w["W"].astype(np.float32)
        p = np.load(md + "/projection.npz")
        L["W1"], L["b1"] = p["W1"].astype(np.float32), \
            p["b1"].astype(np.float32)
        L["W2"], L["b2"] = p["W2"].astype(np.float32), \
            p["b2"].astype(np.float32)

        def _project(Amat):
            """whiten -> head -> L2 normalize both channels."""
            vw = (Amat.astype(np.float32) - L["mu"]) @ L["W"].T
            hw = np.tanh(vw @ L["W1"].T + L["b1"])
            ho = hw @ L["W2"].T + L["b2"]
            vn = vw / np.clip(np.linalg.norm(vw, axis=1,
                                             keepdims=True),
                              1e-9, None)
            hn = ho / np.clip(np.linalg.norm(ho, axis=1,
                                             keepdims=True),
                              1e-9, None)
            return vn, hn

        L["project"] = _project
        L["An"], L["HAn"] = _project(A)
        # memory entries become anchors (the layer grows with use)
        try:
            root = os.environ.get("WEBMIND_OUTD") or \
                os.path.dirname(_TOOLS)
            mem = os.path.join(root, "webmind_memory.jsonl")
            if os.path.isfile(mem):
                extra = []
                for line in open(mem, encoding="utf-8"):
                    try:
                        e = json.loads(line)
                    except Exception:
                        continue
                    if e.get("superseded_by"):
                        continue
                    extra.append(e["q"])
                if extra:
                    V = enc.embed(
                        [mask_numbers(map_dialect(q))
                         for q in extra], batch=32)
                    vn, hn = _project(V)
                    L["An"] = np.concatenate([L["An"], vn],
                                             axis=0)
                    L["HAn"] = np.concatenate([L["HAn"], hn],
                                              axis=0)
                    L["meta"] = L["meta"] + [
                        {"text": q, "kind": "memory",
                         "canonical": q, "entry": "",
                         "n_nums": 0, "seq": False}
                        for q in extra]
        except Exception:
            pass
        _LAYER = L
    except Exception:
        _LAYER = False
    return _LAYER or None


def _match(L: dict, question: str) -> tuple[int, float, float]:
    """HYBRID ranking with GROUP-AWARE margin: 0.5 whitened-cosine
    + 0.5 head-cosine; anchors sharing one canonical target form
    one competitor group (two بالمئة anchors of the same op must
    not eat each other's margin - measured sim 0.917 margin 0.023
    rejecting a correct match). An IDF lexical re-rank was MEASURED
    (fix17 calibration round 2): 137/150 but it traded one exact-
    math hit for one code hit - reverted; exact math must stay 40/40.
    Errors: none."""
    np = L["np"]
    mapped = map_dialect(question)
    v = L["enc"].embed(
        [mask_numbers(mapped)])[0].astype(np.float32)
    vn, hn = L["project"](v[None])
    sims = 0.5 * (L["An"] @ vn[0]) + 0.5 * (L["HAn"] @ hn[0])
    order = np.argsort(-sims)
    top = int(order[0])
    # fix18: kb2 python siblings are verb-discriminated - rerank the
    # code/fact candidates with tight lexical agreement (measured
    # 19 dialect programming questions lost to 0.03 margins)
    if L["meta"][top]["kind"] in ("code", "fact"):
        kf = [int(j) for j in order[:8]
              if L["meta"][int(j)]["kind"] in ("code", "fact")]
        if len(kf) >= 2:
            best_i, _ = _rerank_code(L, question, kf, sims)
            best_c = L["meta"][best_i].get("canonical", "")
            s1 = float(sims[best_i])
            s2 = 0.0
            for j in order:
                j = int(j)
                if j == best_i:
                    continue
                if L["meta"][j].get("canonical", "") != best_c:
                    s2 = float(sims[j])
                    break
            return best_i, s1, s2
    best_i = int(order[0])
    best_c = L["meta"][best_i].get("canonical", "")
    s2 = 0.0
    for j in order[1:]:
        j = int(j)
        if L["meta"][j].get("canonical", "") != best_c:
            s2 = float(sims[j])
            break
    return best_i, float(sims[best_i]), s2


TAU_RESCUE = 0.55   # fall-through rescue (gates + memory missed)
MARGIN = 0.03       # runner-up margin (group-aware)
TAU_KB = 0.60       # KB-first (0.52 measured -1 bank code answer;
                    # 0.60 keeps the bank at 137/150 and the
                    # lexical arbiter still fixes the measured v2
                    # while/for regression at this level)


def _fill_math(anchor: dict, nums: list[str]) -> str | None:
    if anchor.get("seq"):
        if len(nums) < 3:
            return None
        return anchor["canonical"].replace(
            "{s}", "، ".join(nums))
    n = anchor.get("n_nums", 0)
    if n and len(nums) < n:
        return None
    c = anchor["canonical"]
    if n >= 2:
        c = c.replace("{a}", nums[0]).replace("{b}", nums[1])
    return c


def best_candidate(question: str, candidates: list) -> str | None:
    """OUTPUT-side neural role (fix18): among composed candidate
    texts (slogans/ads/descriptions), pick the one most coherent
    with the question - RAW encoder cosine (the whitening/head are
    fitted to the anchor-matching distribution; the raw paraphrase
    space is the right tool for open candidates). Deterministic.
    Errors: none (None -> caller falls back to the hash pick)."""
    L = layer()
    if L is None or not candidates:
        return None
    try:
        np = L["np"]
        mapped = mask_numbers(map_dialect(question))
        qv = L["enc"].embed([mapped])[0].astype(np.float32)
        cv = L["enc"].embed(
            [mask_numbers(c) for c in candidates],
            batch=32).astype(np.float32)
        sims = cv @ qv
        return candidates[int(np.argmax(sims))]
    except Exception:
        return None


def _rerank_code(L: dict, question: str, order,
                 sims) -> tuple[int, float]:
    """fix18 (measured): kb2 python siblings (read/write/append/
    delete/exists - all 'ملف+بايثون') sit within 0.03 margin of each
    other in embedding space; the VERB is the discriminator.
    Among the top-5 neural candidates, re-score with tight lexical
    agreement (equality / >=4 overlap / 1-char prefix) on the anchor
    surfaces. Returns (best_index, best_combined). Errors: none."""
    np = L["np"]
    try:
        from paraphrase_core import tok_match as _tm
    except Exception:
        _tm = lambda a, b: a == b  # noqa: E731

    def tight(t: str, q: str) -> bool:
        if t == q:
            return True
        if not _tm(t, q):
            return False
        lo, hi = sorted((len(t), len(q)))
        if lo >= 4:
            return True
        return (hi - lo) <= 1 and lo >= 3 and (
            t.startswith(q) or q.startswith(t))

    qs = content_stems(question)
    best_i, best_s = None, -1.0
    for j in order[:5]:
        j = int(j)
        a_stems = content_stems(L["meta"][j]["text"])
        hit = sum(1.0 for a_s in a_stems
                  if any(tight(q, a_s) for q in qs))
        lex = hit / max(len(a_stems), 1)
        combined = 0.7 * float(sims[j]) + 0.3 * lex
        if combined > best_s:
            best_s, best_i = combined, j
    return (best_i or int(order[0])), best_s


def canonicalize(question: str,
                 tau: float | None = None) -> dict | None:
    """Rescue path: any dialect/formulation -> canonical question
    for the numeric core. Gates: sim floor, runner-up margin, and a
    shared content stem OR very-high sim. Errors: none."""
    L = layer()
    if L is None:
        return None
    tau = tau if tau is not None else TAU_RESCUE
    i, s1, s2 = _match(L, question)
    a = L["meta"][i]
    if s1 < tau or s1 < s2 + MARGIN:
        return None
    if a["kind"] == "math":
        nums = extract_numbers(question)
        c = _fill_math(a, nums)
    elif a["kind"] == "creative":
        c = a["canonical"].replace(
            "{S}", _extract_subject(question))
    else:
        c = a["canonical"]
    if not c:
        return None
    if not stems_shared(content_stems(question),
                        content_stems(a["text"])):
        if s1 < 0.62:
            return None
    out = {"canonical": c, "kind": a["kind"],
           "sim": round(s1, 3), "margin": round(s1 - s2, 3),
           "mode": "neural"}
    # fix18: resolve the KB entry NOW - the rescue answers from the
    # entry directly (measured: rerunning the gates on the canonical
    # text lost dialect questions to sibling stem scores)
    if a["kind"] in ("code", "fact"):
        try:
            from skills_core import CODE_KB, FACTS_KB
            pool = CODE_KB if a["kind"] == "code" else FACTS_KB
            for e in pool:
                if e.get("q") == c:
                    out["entry"] = e
                    break
        except Exception:
            pass
    return out


def _idf_doc_score(q_stems: list, index, doc_i: int,
                   matcher=None) -> float:
    """Σ idf over question stems matched in ONE KB doc (same
    semantics as paraphrase_core.KBIndex.score, restricted to one
    candidate). matcher overrides the default tightened matcher.
    Errors: none."""
    try:
        if matcher is None:
            from paraphrase_core import tok_match

            def matcher(t: str, q: str) -> bool:
                if t == q:
                    return True
                if min(len(t), len(q)) < 4:
                    return False
                return tok_match(t, q)

        doc = index.docs[doc_i]
        s = 0.0
        for q in q_stems:
            cand = [index.idf(t) for t in doc if matcher(t, q)]
            if cand:
                s += max(cand)
        return s
    except Exception:
        return 0.0


def _q_stems_of(question: str) -> list:
    """Question stems exactly as kb_rank builds them. Errors: none."""
    from paraphrase_core import _TOK, _CLEAN, _STOP, stem
    q_text = _CLEAN.sub(" ", question)
    return [stem(t) for t in _TOK.findall(q_text)
            if t not in _STOP and len(t) >= 2]


def canonical_entry(question: str) -> dict | None:
    """KB-first path (BEFORE the stem-IDF fallback): the neural
    layer PROPOSES a code/fact entry; the lexical count ARBITRATES
    against the stem-IDF best when they disagree (measured v2
    regression: 'حلقة while' answered from the for-loop sibling -
    the distinctive 'while' token must win; measured hijack:
    'أعمل دالة' answered print - the neural pick must win). High
    precision gates + IDF-weighted arbitration. Errors: none."""
    L = layer()
    if L is None:
        return None
    i, s1, s2 = _match(L, question)
    a = L["meta"][i]
    if a["kind"] not in ("code", "fact"):
        return None
    if s1 < TAU_KB or s1 < s2 + MARGIN:
        return None
    if not stems_shared(content_stems(question),
                        content_stems(a["text"])):
        return None
    try:
        from skills_core import CODE_KB, FACTS_KB, \
            CODE_INDEX, FACTS_INDEX
        from paraphrase_core import kb_rank
        pool = CODE_KB if a["kind"] == "code" else FACTS_KB
        index = CODE_INDEX if a["kind"] == "code" \
            else FACTS_INDEX
        n_i = next((k for k, e in enumerate(pool)
                    if e.get("q") == a["canonical"]), None)
        if n_i is None:
            return None
        # EXACT-SURFACE OVERRIDE (measured v2-bench regression): a
        # near-exact anchor match (sim>=0.98) IS the strongest
        # evidence - the lexical arbiter must never override it
        # ('إخراج أمر إلى ملف؟' matched its tee entry 1.000 and was
        # wrongly arbitrated to the mv sibling).
        if s1 >= 0.98:
            suffix = (" — من معرفتي المثبتة."
                      if a["kind"] == "fact" else "")
            return {"entry": pool[n_i], "kind": a["kind"],
                    "sim": round(s1, 3),
                    "mode": "neural_kb_exact"}
        # fix18: arbitrate ONLY same-KB (kb1 neural pick vs the
        # kb1-frozen stem index). kb2 entries have NO stem index by
        # design (measured hijack) - the neural pick stands for them.
        from skills_core import CODE_KB1_LEN
        if a["kind"] == "code" and n_i >= CODE_KB1_LEN:
            return {"entry": pool[n_i], "kind": a["kind"],
                    "sim": round(s1, 3),
                    "mode": "neural_kb_first"}
        q_stems = _q_stems_of(question)
        neural_sc = _idf_doc_score(q_stems, index, n_i)
        # the stem-IDF challengers from BOTH indexes
        best_lex, best_lex_entry, best_lex_name = -1.0, None, ""
        for name, idx in (("code", CODE_INDEX),
                          ("fact", FACTS_INDEX)):
            if idx is None:
                continue
            r = kb_rank(question, idx)
            if r is not None:
                entry, sc, _2 = r
                if sc > best_lex:
                    best_lex, best_lex_entry, best_lex_name = \
                        sc, entry, name
        if best_lex_entry is not None \
                and best_lex_entry is not pool[n_i] \
                and best_lex > neural_sc * 1.15:
            kbname = best_lex_name
            suffix = (" — من معرفتي المثبتة."
                      if kbname == "fact" else "")
            return {"entry": best_lex_entry, "kind": kbname,
                    "sim": round(s1, 3),
                    "mode": "kb_lexical_arbiter"}
        suffix = (" — من معرفتي المثبتة."
                  if a["kind"] == "fact" else "")
        return {"entry": pool[n_i], "kind": a["kind"],
                "sim": round(s1, 3), "mode": "neural_kb_first"}
    except Exception:
        pass
    return None
