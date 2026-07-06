"""
Citation-accuracy evaluation for research-desk debate findings.

Specialist drafts cite filing chunks by id (e.g. "NVDA-10K-0007"); nothing stops
the model from inventing an id or attaching real ids to fabricated evidence. This
harness scores dumped findings against the FilingStore that produced the debate:
a finding is *grounded* if some token of its ``source`` resolves to a real chunk,
and *supported* if the cited chunk's text actually backs the ``evidence`` string —
by default via token-overlap (threshold 0.30), optionally via a Haiku LLM judge.
Ported from analyst_debate/eval_harness.py in the upstream hedge-fund repo.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterator

_STOP = set("the a an of to in for and or is are was were be on with by as at from that this it its".split())
_JUDGE_MODEL = "claude-haiku-4-5"
DEFAULT_THRESHOLD = 0.30


def _toks(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in _STOP and len(t) > 1}


def _lexical_support(evidence: str, chunk_text: str) -> float:
    """Fraction of evidence tokens found in the chunk — cheap attribution proxy."""
    e = _toks(evidence)
    return (len(e & _toks(chunk_text)) / len(e)) if e else 0.0


def _candidate_ids(source: str, store: Any) -> Iterator[str]:
    """Yield tokens of the source string that resolve to a real chunk in the store.

    Models sometimes wrap the chunk id in prose or punctuation ("see NVDA-10K-0003,
    Item 7"), so every whitespace/punctuation-split token is tried against the store.
    """
    for tok in re.split(r"[\s,;|()\[\]]+", source or ""):
        tok = tok.strip()
        if tok and store.get(tok):
            yield tok


async def _judge_supported(client: Any, evidence: str, chunk_text: str) -> bool:
    """LLM attribution check (claude-haiku-4-5). Only called when a client is passed."""
    schema = {
        "type": "object",
        "properties": {"supported": {"type": "boolean"}},
        "required": ["supported"],
        "additionalProperties": False,
    }
    resp = await client.messages.create(
        model=_JUDGE_MODEL,
        max_tokens=200,
        system="You verify attribution. Answer only whether the passage substantiates the claimed evidence.",
        messages=[{
            "role": "user",
            "content": f"PASSAGE:\n{chunk_text}\n\nCLAIMED EVIDENCE:\n{evidence}\n\n"
                       "Does the passage substantiate the claimed evidence?",
        }],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    txt = next((b.text for b in resp.content if b.type == "text"), '{"supported": false}')
    try:
        return bool(json.loads(txt).get("supported"))
    except (ValueError, AttributeError):
        return False


async def evaluate_findings(
    findings: list[dict],
    store: Any,
    threshold: float = DEFAULT_THRESHOLD,
    judge_client: Any = None,
) -> dict:
    """Score finding dicts (SpecialistDraft.key_findings dumps: claim/evidence/source)
    against a FilingStore. Returns the metrics dict; never raises on bad findings —
    missing fields count as hallucinated/unsupported rather than erroring the run.
    """
    total = len(findings)
    grounded = supported = 0
    hallucinated: list[str] = []

    for f in findings:
        source = f.get("source", "") or ""
        chunk = next((store.get(cid) for cid in _candidate_ids(source, store)), None)
        if chunk is None:
            hallucinated.append(source or "<empty>")
            continue
        grounded += 1
        evidence = f.get("evidence", "") or ""
        if judge_client is not None:
            ok = await _judge_supported(judge_client, evidence, chunk.text)
        else:
            ok = _lexical_support(evidence, chunk.text) >= threshold
        supported += int(ok)

    return {
        "total_findings": total,
        "grounded": grounded,
        "supported": supported,
        "grounded_rate": grounded / total if total else 0.0,
        "supported_of_grounded": supported / grounded if grounded else 0.0,
        "citation_accuracy": supported / total if total else 0.0,
        "hallucinated_sources": hallucinated,
        "method": "llm-judge" if judge_client is not None else f"lexical>={threshold}",
    }
