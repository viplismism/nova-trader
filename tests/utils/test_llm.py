from src.utils.llm import (
    _extract_json_object,
    _openai_compatible_config,
    _prompt_to_openai_messages,
)


def test_extract_json_object_from_fenced_response():
    assert _extract_json_object('```json\n{"decision":"hold"}\n```') == {"decision": "hold"}


def test_openai_messages_add_json_instruction_when_missing():
    messages = _prompt_to_openai_messages("Give me a decision")

    assert messages[0]["role"] == "system"
    assert "JSON" in messages[0]["content"]


def test_minimax_uses_openai_compatible_config(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "minimax-test-key")
    monkeypatch.delenv("MINIMAX_API_BASE", raising=False)

    api_key, base_url, headers = _openai_compatible_config("MiniMax", None)

    assert api_key == "minimax-test-key"
    assert base_url == "https://api.minimax.io/v1"
    assert headers == {}
