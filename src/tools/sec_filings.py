"""SEC EDGAR filing retrieval and lightweight passage search.

This is intentionally small and dependency-free: fetch the latest 10-K/10-Q,
strip HTML, chunk text, and rank chunks with a BM25-style scorer. Network or SEC
failures are allowed to bubble to the snapshot layer, which already records
partial data and lets the filings analyst abstain.
"""

from __future__ import annotations

import html
import json
import math
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from src.data.models import FilingExcerpt
from src.utils.progress import progress, current_fetch_owner


TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "nova-trader/0.1 research@example.com")
CACHE_DIR = Path(os.getenv("NOVA_SEC_CACHE_DIR", Path.home() / ".nova-trader" / "sec-cache"))

DEFAULT_FILINGS_QUERIES = [
    "business revenue segments growth margins",
    "risk factors competition customer concentration demand",
    "liquidity cash debt capital allocation",
    "management discussion operating results outlook",
    "supply chain regulation litigation cybersecurity",
]

_CHUNK_CHARS = 1400
_STOP = set(
    "the a an of to in for and or is are was were be been on with by as at from that this it its "
    "which we our us their they he she his her not no but if then than into over under company".split()
)


@dataclass(frozen=True)
class _Chunk:
    chunk_id: str
    ticker: str
    form: str
    fiscal_year: str
    item: str
    url: str
    start: int
    end: int
    text: str


def _fetch(url: str, cache_name: str | None = None) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / cache_name if cache_name else None
    if path is not None and path.exists():
        return path.read_text(encoding="utf-8")
    req = urllib.request.Request(url, headers={"User-Agent": SEC_USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as response:
        body = response.read().decode("utf-8", errors="replace")
    time.sleep(0.12)
    if path is not None:
        path.write_text(body, encoding="utf-8")
    return body


def _ticker_to_cik(ticker: str) -> str:
    data = json.loads(_fetch(TICKERS_URL, "company_tickers.json"))
    for row in data.values():
        if row.get("ticker", "").upper() == ticker.upper():
            return str(row["cik_str"]).zfill(10)
    raise ValueError(f"ticker {ticker!r} not found in SEC company list")


def _strip_html(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|head).*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    raw = html.unescape(raw).replace("\xa0", " ")
    return re.sub(r"\s+", " ", raw).strip()


def _item_marks(text: str) -> list[tuple[int, str]]:
    marks: list[tuple[int, str]] = []
    for match in re.finditer(r"\bItem\s+(\d{1,2}[AB]?)\b[.:\s]", text, flags=re.IGNORECASE):
        marks.append((match.start(), f"Item {match.group(1).upper()}"))
    return marks


def _item_for(offset: int, marks: list[tuple[int, str]]) -> str:
    label = "unknown"
    for pos, item in marks:
        if pos <= offset:
            label = item
        else:
            break
    return label


def _chunks(text: str) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    n = len(text)
    i = 0
    while i < n:
        end = min(i + _CHUNK_CHARS, n)
        if end < n:
            split = text.rfind(" ", i + _CHUNK_CHARS // 2, end)
            if split != -1:
                end = split
        body = text[i:end].strip()
        if len(body) > 120:
            out.append((i, end, body))
        i = max(end, i + 1)
    return out


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in _STOP and len(token) > 1]


def _counts(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def _rank_chunks(chunks: list[_Chunk], query: str, limit: int) -> list[_Chunk]:
    if not chunks:
        return []
    docs = [_tokens(chunk.text) for chunk in chunks]
    avg_len = sum(len(doc) for doc in docs) / max(len(docs), 1)
    tf = [_counts(doc) for doc in docs]
    df: dict[str, int] = {}
    for counts in tf:
        for term in counts:
            df[term] = df.get(term, 0) + 1
    q = _tokens(query)
    scores: list[tuple[float, int]] = []
    for i, counts in enumerate(tf):
        score = 0.0
        for term in q:
            freq = counts.get(term)
            if not freq:
                continue
            idf = math.log(1 + (len(docs) - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
            denom = freq + 1.5 * (1 - 0.75 + 0.75 * len(docs[i]) / (avg_len or 1))
            score += idf * freq * 2.5 / denom
        if score > 0:
            scores.append((score, i))
    scores.sort(reverse=True)
    return [chunks[i] for _, i in scores[:limit]]


def _latest_filing_chunks(ticker: str, forms: tuple[str, ...], per_form: int) -> list[_Chunk]:
    cik = _ticker_to_cik(ticker)
    subs = json.loads(_fetch(f"https://data.sec.gov/submissions/CIK{cik}.json", f"sub-{cik}.json"))
    recent = subs["filings"]["recent"]
    wanted = {form: per_form for form in forms}
    chunks: list[_Chunk] = []
    idx = 0
    for form, accession, document, report_date in zip(
        recent["form"],
        recent["accessionNumber"],
        recent["primaryDocument"],
        recent["reportDate"],
    ):
        if form not in wanted or wanted[form] <= 0 or not document:
            continue
        wanted[form] -= 1
        accession_no_dash = accession.replace("-", "")
        fiscal_year = (report_date or "")[:4]
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dash}/{document}"
        raw = _fetch(url, f"{cik}-{accession_no_dash}-{document}.html")
        text = _strip_html(raw)
        marks = _item_marks(text)
        code = form.replace("-", "")
        for start, end, body in _chunks(text):
            chunks.append(
                _Chunk(
                    chunk_id=f"{ticker.upper()}-{code}-{idx:04d}",
                    ticker=ticker.upper(),
                    form=form,
                    fiscal_year=fiscal_year,
                    item=_item_for(start, marks),
                    url=url,
                    start=start,
                    end=end,
                    text=body,
                )
            )
            idx += 1
        if all(remaining <= 0 for remaining in wanted.values()):
            break
    return chunks


def get_sec_filing_excerpts(
    ticker: str,
    *,
    queries: list[str] | None = None,
    forms: tuple[str, ...] = ("10-K", "10-Q"),
    per_form: int = 1,
    per_query: int = 3,
    max_excerpts: int = 14,
) -> list[FilingExcerpt]:
    """Return relevant excerpts from the latest SEC filings for a ticker."""

    progress.record_fetch(current_fetch_owner.get(), "sec_filings")
    ticker = ticker.upper()
    chunks = _latest_filing_chunks(ticker, forms=forms, per_form=per_form)
    selected: dict[str, _Chunk] = {}
    for query in queries or DEFAULT_FILINGS_QUERIES:
        for chunk in _rank_chunks(chunks, query, per_query):
            selected.setdefault(chunk.chunk_id, chunk)
            if len(selected) >= max_excerpts:
                break
        if len(selected) >= max_excerpts:
            break
    if not selected:
        selected = {chunk.chunk_id: chunk for chunk in chunks[:max_excerpts]}
    return [
        FilingExcerpt(
            ticker=chunk.ticker,
            chunk_id=chunk.chunk_id,
            form=chunk.form,
            fiscal_year=chunk.fiscal_year,
            item=chunk.item,
            url=chunk.url,
            text=chunk.text[:1200],
        )
        for chunk in selected.values()
    ]
