from src.chat.models import DEFAULT_AGENTS
from src.registry import AGENT_REGISTRY, default_agent_ids


def test_default_agents_have_one_registered_source_of_truth():
    assert DEFAULT_AGENTS == default_agent_ids()
    assert all(agent_id in AGENT_REGISTRY for agent_id in DEFAULT_AGENTS)
    assert "reddit_sentiment" not in DEFAULT_AGENTS
    assert "adaptive_research" not in DEFAULT_AGENTS


def test_default_agent_ids_returns_a_fresh_list():
    first = default_agent_ids()
    first.append("not-real")
    assert "not-real" not in default_agent_ids()
