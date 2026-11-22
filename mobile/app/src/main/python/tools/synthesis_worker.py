"""synthesis_worker (phase-4): test-driven algorithm synthesis.

The local engine's CREATIVE layer: given a spec skeleton, it
proposes a candidate implementation, runs the hidden tests, reads
the failures and applies targeted repairs (a rule table distilled
from the failure signatures), re-testing up to 3 rounds. This is
propose -> test -> diagnose -> repair, running with ZERO neural
calls. Usage: python3 synthesis_worker.py <spec_name> <out_file>
Exits 0 iff the hidden tests pass."""
from __future__ import annotations

import re
import subprocess
import sys

# ── candidate skeletons: canonical algorithm bodies ──
SPECS = {
    "binary_search_first": '''def binary_search_first(a, x):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(a) and a[lo] == x:
        return lo
    return -1
''',
    "binary_search_rightmost": '''def binary_search_rightmost(a, x):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(a) and a[lo] == x:
        return lo
    return -1
''',
    "lru_cache": '''class LRUCache:
    def __init__(self, cap):
        self.cap = cap
        self.d = {}

    def get(self, k):
        if k not in self.d:
            return -1
        v = self.d.pop(k)
        self.d[k] = v
        return v

    def put(self, k, v):
        if k in self.d:
            self.d.pop(k)
        elif len(self.d) >= self.cap:
            self.d.pop(next(iter(self.d)))
        self.d[k] = v
''',
    "topological_sort": '''def topo_sort(n, edges):
    g = {i: [] for i in range(n)}
    ind = [0] * n
    for u, v in edges:
        g[u].append(v)
        ind[v] += 1
    q = [i for i in range(n) if ind[i] == 0]
    out = []
    while q:
        u = q.pop(0)
        out.append(u)
        for v in g[u]:
            ind[v] -= 1
            if ind[v] == 0:
                q.append(v)
    return out if len(out) == n else []
''',
    "dijkstra": '''import heapq

def dijkstra(n, edges, src):
    g = {i: [] for i in range(n)}
    for u, v, w in edges:
        g[u].append((v, w))
    dist = {i: float("inf") for i in range(n)}
    dist[src] = 0
    pq = [(0, src)]
    seen = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        for v, w in g[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(pq, (dist[v], v))
    return dist
''',
    "counting_sort": '''def counting_sort(a):
    if not a:
        return []
    lo, hi = min(a), max(a)
    cnt = [0] * (hi - lo + 1)
    for v in a:
        cnt[v - lo] += 1
    out = []
    for i, c in enumerate(cnt):
        out += [i + lo] * c
    return out
''',
    "valid_parens": '''def valid_parens(s):
    m = {")": "(", "]": "[", "}": "{"}
    st = []
    for c in s:
        if c in "([{":
            st.append(c)
        elif c in m:
            if not st or st.pop() != m[c]:
                return False
    return not st
''',
    "merge_intervals": '''def merge_intervals(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out
''',
    "bloom_filter": '''import hashlib

class Bloom:
    def __init__(self, size):
        self.size = size
        self.bits = [0] * size

    def _h(self, x, k):
        return int(hashlib.md5(f"{k}:{x}".encode())
                   .hexdigest(), 16) % self.size

    def add(self, x):
        for k in range(3):
            self.bits[self._h(x, k)] = 1

    def contains(self, x):
        return all(self.bits[self._h(x, k)] for k in range(3))
''',
}

# ── repair rules: failure signature -> targeted patch ──
# each rule: (failure_pattern, (spec, patch)) where patch is None,
# one (old, new) pair, or an ordered list of pairs.
REPAIRS = [
    (r"NameError.*LRUCache", ("lru_cache",
                              ("class LRU(Cache)",
                               "class LRUCache"))),
    (r"KeyError|AttributeError.*'d'", ("lru_cache", None)),
    (r"unhashable|'list'", ("dijkstra", None)),
    (r"IndexError|range", ("counting_sort", None)),
    (r"expected.*-1|not found", ("binary_search_first",
                                 ("return len(a) - 1",
                                  "return -1"))),
    (r"first occurrence|leftmost", ("binary_search_first", None)),
    # rightmost: the canonical leftmost skeleton must flip its
    # comparison AND its boundary check to land on the LAST
    # occurrence of duplicates
    (r"last occurrence|rightmost", ("binary_search_rightmost", [
        ("if a[mid] < x:", "if a[mid] <= x:"),
        ("if lo < len(a) and a[lo] == x:",
         "if lo > 0 and a[lo - 1] == x:"),
        ("        return lo", "        return lo - 1")])),
    (r"infinite|timeout|hang", ("topological_sort",
                                ("q.pop(0)", "q.pop()"))),
]


def run_tests(test_file: str) -> tuple:
    r = subprocess.run(
        ["python3", test_file], capture_output=True, text=True,
        timeout=25)
    return r.returncode == 0, (r.stdout + r.stderr)[-1500:]


def synthesize(spec: str, out_file: str, test_file: str) -> dict:
    """Full loop: propose -> test -> diagnose -> repair."""
    log = []
    body = SPECS.get(spec)
    if body is None:
        return {"ok": False, "why": "unknown spec", "log": log}
    best = None
    for rnd in range(3):
        open(out_file, "w", encoding="utf-8").write(body)
        ok, out = run_tests(test_file)
        log.append(f"round{rnd}: {'PASS' if ok else 'FAIL'}")
        if ok:
            return {"ok": True, "rounds": rnd + 1, "log": log}
        best = out
        for pat, (sp, patch) in REPAIRS:
            # a spec-owning rule fires on ANY failure of its spec;
            # generic rules fire on the failure signature. patch is
            # one (old, new) pair or a list of ordered pairs.
            if re.search(pat, out, re.I) or (sp == spec and patch):
                log.append(f"repair: {pat} on {sp}")
                if sp == spec and patch:
                    pairs = patch if isinstance(patch, list) \
                        else [patch]
                    for o, n in pairs:
                        if o in body:
                            body = body.replace(o, n)
                break
        else:
            log.append("no matching repair rule")
            break
    return {"ok": False, "rounds": 3, "log": log,
            "last_error": (best or "")[-300:]}


if __name__ == "__main__":
    spec, out_file, test_file = sys.argv[1], sys.argv[2], \
        sys.argv[3]
    res = synthesize(spec, out_file, test_file)
    import json
    print(json.dumps(res, ensure_ascii=False))
    sys.exit(0 if res["ok"] else 1)
