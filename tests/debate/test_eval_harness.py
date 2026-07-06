"""Tests for the citation-accuracy eval harness — fake store, no network, no LLM."""

import asyncio

from src.debate.eval_harness import _lexical_support, evaluate_findings


class _StubChunk:
    def __init__(self, text: str):
        self.text = text


class _StubStore:
    """Minimal FilingStore stand-in: .get(chunk_id) -> chunk with .text."""

    def __init__(self, chunks: dict[str, str]):
        self._chunks = {k: _StubChunk(v) for k, v in chunks.items()}

    def get(self, chunk_id: str):
        return self._chunks.get(chunk_id.strip())


_STORE = _StubStore({
    "NVDA-10K-0001": "Data center revenue grew 4o9% year over year driven by strong "
                     "demand for AI accelerator products across hyperscale customers.",
    "NVDA-10K-0002": "The company repurchased shares under its existing buyback program.",
})

_FINDINGS = [
    {   # grounded + supported: id resolves, evidence tokens overlap chunk text
        "claim": "Data center is growing fast",
        "evidence": "data center revenue grew driven by strong AI accelerator demand",
        "source": "NVDA-10K-0001",
    },
    {   # grounded + unsupported: id resolves, evidence unrelated to chunk text
        "claim": "Margins expanded",
        "evidence": "gross margin expanded materially due pricing power inventory normalization",
        "source": "see NVDA-10K-0002, Item 7",
    },
    {   # hallucinated: no token resolves to a chunk
        "claim": "Guidance raised",
        "evidence": "management raised full year guidance",
        "source": "NVDA-10K-9999",
    },
]


def test_metrics_all_cases():
    m = asyncio.run(evaluate_findings(_FINDINGS, _STORE))
    assert m["total_findings"] == 3
    assert m["grounded"] == 2
    assert m["supported"] == 1
    assert m["grounded_rate"] == 2 / 3
    assert m["supported_of_grounded"] == 0.5
    assert m["citation_accuracy"] == 1 / 3
    assert m["hallucinated_sources"] == ["NVDA-10K-9999"]
    assert m["method"] == "lexical>=0.3"


def test_source_id_extracted_from_prose():
    # the second finding's source wraps the id in prose/punctuation yet grounds
    m = asyncio.run(evaluate_findings([_FINDINGS[1]], _STORE))
    assert m["grounded"] == 1


def test_empty_findings_and_empty_source():
    m = asyncio.run(evaluate_findings([], _STORE))
    assert m["total_findings"] == 0
    assert m["grounded_rate"] == 0.0
    assert m["supported_of_grounded"] == 0.0
    assert m["citation_accuracy"] == 0.0

    m = asyncio.run(evaluate_findings([{"claim": "x", "evidence": "y", "source": ""}], _STORE))
    assert m["hallucinated_sources"] == ["<empty>"]


def test_lexical_support_threshold():
    assert _lexical_support("data center revenue grew", _STORE.get("NVDA-10K-0001").text) >= 0.30
    assert _lexical_support("unrelated words entirely", _STORE.get("NVDA-10K-0001").text) < 0.30
    assert _lexical_support("", "anything") == 0.0


def test_judge_path_uses_stub_client_not_network():
    class _Block:
        type = "text"
        text = '{"supported": true}'

    class _Resp:
        content = [_Block()]

    class _Messages:
        async def create(self, **kwargs):
            assert kwargs["model"] == "claude-haiku-4-5"
            return _Resp()

    class _StubClient:
        messages = _Messages()

    m = asyncio.run(evaluate_findings([_FINDINGS[1]], _STORE, judge_client=_StubClient()))
    assert m["supported"] == 1  # judge overrides the lexical miss
    assert m["method"] == "llm-judge"
