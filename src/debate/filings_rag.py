"""
Filings RAG — fetch a company's latest 10-K / 10-Q from SEC EDGAR, index them, and serve
"""

from __future__ import annotations

import html
import json
import math
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field

SEC_UA = os.environ.get("SEC_USER_AGENT", "analyst-debate-prototype tech.admin@credilinq.ai")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

_CHUNK_CHARS = 1200
_STOP = set(
    "the a an of to in for and or is are was were be been on with by as at from that this it its "
    "which we our us their they he she his her not no but if then than into over under".split()
)


# --------------------------------------------------------------------------------------
# HTTP (polite, cached)
# --------------------------------------------------------------------------------------
def _fetch(url: str, cache_name: str | None = None) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    if cache_name:
        path = os.path.join(CACHE_DIR, cache_name)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    req = urllib.request.Request(url, headers={"User-Agent": SEC_UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read().decode("utf-8", errors="replace")
    time.sleep(0.2)  # respect SEC's ~10 req/s ceiling
    if cache_name:
        with open(os.path.join(CACHE_DIR, cache_name), "w", encoding="utf-8") as f:
            f.write(body)
    return body


def _ticker_to_cik(ticker: str) -> str:
    data = json.loads(_fetch(TICKERS_URL, "company_tickers.json"))
    for row in data.values():
        if row["ticker"].upper() == ticker.upper():
            return str(row["cik_str"]).zfill(10)
    raise ValueError(f"ticker {ticker!r} not found in SEC company list")



def _strip_html(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|head).*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    raw = raw.replace("\xa0", " ")
    return re.sub(r"\s+", " ", raw).strip()


def _item_map(text: str) -> list[tuple[int, str]]:
    """Offsets of '... Item 1A.' / 'Item 7.' headers, for labelling chunks by section."""
    marks: list[tuple[int, str]] = []
    for m in re.finditer(r"\bItem\s+(\d{1,2}[AB]?)\b[.:\s]", text):
        marks.append((m.start(), f"Item {m.group(1).upper()}"))
    return marks


def _item_for(offset: int, marks: list[tuple[int, str]]) -> str:
    label = "—"
    for pos, name in marks:
        if pos <= offset:
            label = name
        else:
            break
    return label


def _chunk(text: str) -> list[tuple[int, int, str]]:
    """Split into ~_CHUNK_CHARS windows on whitespace boundaries; return (start, end, text)."""
    out, n, i = [], len(text), 0
    while i < n:
        end = min(i + _CHUNK_CHARS, n)
        if end < n:
            sp = text.rfind(" ", i + _CHUNK_CHARS // 2, end)
            if sp != -1:
                end = sp
        out.append((i, end, text[i:end].strip()))
        i = end
    return [c for c in out if len(c[2]) > 80]



def _tok(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in _STOP and len(t) > 1]


class _BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.docs = k1, b, docs
        self.dl = [len(d) for d in docs]
        self.avgdl = (sum(self.dl) / len(docs)) if docs else 0.0
        self.tf = [_count(d) for d in docs]
        df: dict[str, int] = {}
        for counts in self.tf:
            for term in counts:
                df[term] = df.get(term, 0) + 1
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, k: int) -> list[int]:
        q = _tok(query)
        scores = []
        for i, counts in enumerate(self.tf):
            s = 0.0
            for term in q:
                if term not in counts:
                    continue
                f = counts[term]
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / (self.avgdl or 1))
                s += self.idf.get(term, 0.0) * f * (self.k1 + 1) / denom
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [i for _, i in scores[:k]]


def _count(tokens: list[str]) -> dict[str, int]:
    d: dict[str, int] = {}
    for t in tokens:
        d[t] = d.get(t, 0) + 1
    return d



@dataclass
class Chunk:
    chunk_id: str
    ticker: str
    form: str
    fiscal: str
    item: str
    url: str
    start: int
    end: int
    text: str


@dataclass
class FilingStore:
    ticker: str
    chunks: list[Chunk] = field(default_factory=list)
    _by_id: dict[str, Chunk] = field(default_factory=dict)
    _bm25: _BM25 | None = None
    _filings: list[str] = field(default_factory=list)  # human labels for summary()

    # ---- build ----
    @classmethod
    def from_ticker(cls, ticker: str, forms=("10-K", "10-Q"), per_form: int = 1) -> "FilingStore":
        ticker = ticker.upper()
        cik = _ticker_to_cik(ticker)
        subs = json.loads(_fetch(f"https://data.sec.gov/submissions/CIK{cik}.json", f"sub-{cik}.json"))
        recent = subs["filings"]["recent"]
        store = cls(ticker=ticker)
        idx = 0
        wanted = {f: per_form for f in forms}
        for form, accn, doc, rdate in zip(
            recent["form"], recent["accessionNumber"], recent["primaryDocument"], recent["reportDate"]
        ):
            if form not in wanted or wanted[form] <= 0 or not doc:
                continue
            wanted[form] -= 1
            fiscal = (rdate or "")[:4]
            accn_nodash = accn.replace("-", "")
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn_nodash}/{doc}"
            raw = _fetch(url, f"{cik}-{accn_nodash}-{doc}.html")
            text = _strip_html(raw)
            marks = _item_map(text)
            code = form.replace("-", "")  # 10-K -> 10K
            store._filings.append(f"{form} ({fiscal})")
            for s, e, body in _chunk(text):
                cid = f"{ticker}-{code}-{idx:04d}"
                ch = Chunk(cid, ticker, form, fiscal, _item_for(s, marks), url, s, e, body)
                store.chunks.append(ch)
                store._by_id[cid] = ch
                idx += 1
            if all(v <= 0 for v in wanted.values()):
                break
        store._bm25 = _BM25([_tok(c.text) for c in store.chunks])
        return store

    # ---- query ----
    def search(self, query: str, k: int = 5) -> list[Chunk]:
        if not self._bm25 or not self.chunks:
            return []
        return [self.chunks[i] for i in self._bm25.search(query, k)]

    def get(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id.strip())

    def summary(self) -> str:
        return ", ".join(self._filings) if self._filings else "no filings"

    def __len__(self) -> int:
        return len(self.chunks)


if __name__ == "__main__":  # quick manual smoke test:  python filings_rag.py NVDA "data center revenue"
    import sys

    store = FilingStore.from_ticker(sys.argv[1] if len(sys.argv) > 1 else "NVDA")
    print(f"indexed {len(store)} chunks from {store.summary()}")
    for h in store.search(sys.argv[2] if len(sys.argv) > 2 else "revenue growth", k=3):
        print(f"\n[{h.chunk_id}] {h.form} {h.fiscal} - {h.item}\n{h.text[:300]}...")
