from src.utils import llm
from src.utils.llm import (
    _extract_json_object,
    _fallback_model_config,
    _format_exception,
    _openai_compatible_config,
    _prompt_to_openai_messages,
    _should_try_fallback,
    _split_anthropic_messages,
    provider_has_credentials,
    required_api_key_name,
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


def test_provider_credential_status_uses_expected_env_key(monkeypatch):
    monkeypatch.setattr(llm, "_ENV_LOADED", True)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert required_api_key_name("OpenAI") == "OPENAI_API_KEY"
    assert not provider_has_credentials("OpenAI")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    assert provider_has_credentials("OpenAI")


def test_quota_errors_can_fallback_to_minimax(monkeypatch):
    monkeypatch.setattr(llm, "_ENV_LOADED", True)
    monkeypatch.setenv("MINIMAX_API_KEY", "minimax-test-key")
    monkeypatch.delenv("NOVA_LLM_FALLBACK_PROVIDER", raising=False)
    monkeypatch.delenv("NOVA_LLM_FALLBACK_MODEL", raising=False)

    assert _should_try_fallback(RuntimeError("Error code: 429 insufficient_quota"))
    assert _fallback_model_config("OpenAI", "gpt-4.1-mini") == ("MiniMax-M2.7", "MiniMax")


def test_anthropic_provider_uses_expected_env_key(monkeypatch):
    monkeypatch.setattr(llm, "_ENV_LOADED", True)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert required_api_key_name("Anthropic") == "ANTHROPIC_API_KEY"
    assert not provider_has_credentials("Anthropic")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    assert provider_has_credentials("Anthropic")


def test_split_anthropic_messages_extracts_system_and_maps_roles():
    system, convo = _split_anthropic_messages(
        [
            {"role": "system", "content": "You are Nova."},
            {"role": "user", "content": "What changed?"},
            {"role": "assistant", "content": "Earnings beat."},
        ]
    )

    assert system == "You are Nova."
    assert convo == [
        {"role": "user", "content": "What changed?"},
        {"role": "assistant", "content": "Earnings beat."},
    ]


def test_split_anthropic_messages_synthesizes_user_turn_when_only_system():
    system, convo = _split_anthropic_messages([{"role": "system", "content": "Context only."}])

    assert system == "Context only."
    assert convo == [{"role": "user", "content": "Context only."}]


def test_format_exception_includes_cause_chain():
    try:
        try:
            raise OSError("DNS lookup failed")
        except OSError as cause:
            raise RuntimeError("Connection error.") from cause
    except RuntimeError as exc:
        text = _format_exception(exc)

    assert "RuntimeError: Connection error." in text
    assert "OSError: DNS lookup failed" in text
