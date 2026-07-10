"""Small direct LLM helper used by agents."""

import json
import os
from typing import Any

import requests
from dotenv import load_dotenv
from pydantic import BaseModel
from src.utils.progress import progress
# Lightweight state shim — call_llm accepts a dict-like state with a "metadata" key.
# Kept loose because both the new v2 engine and the backtest adapter pass plain dicts.
StateLike = dict[str, Any]

_ENV_LOADED = False

_PROVIDER_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "groq": "GROQ_API_KEY",
    "minimax": "MINIMAX_API_KEY",
    "xai": "XAI_API_KEY",
}

# Default Anthropic model used when a caller selects the Anthropic provider
# without naming a model. Opus 4.8 is the most capable Claude model.
_ANTHROPIC_DEFAULT_MODEL = "claude-opus-4-8"


def _ensure_env_loaded() -> None:
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv(override=False)
        _ENV_LOADED = True


def _provider_name(model_provider: str | object) -> str:
    if hasattr(model_provider, "value"):
        return str(model_provider.value).lower()
    return str(model_provider).lower()


def _message_role(message: object) -> str:
    message_type = getattr(message, "type", "")
    if message_type == "system":
        return "system"
    if message_type == "ai":
        return "assistant"
    return "user"


def _prompt_to_openai_messages(prompt: object) -> list[dict[str, str]]:
    """Convert supported prompt shapes into OpenAI chat messages."""
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    elif hasattr(prompt, "to_messages"):
        messages = [
            {"role": _message_role(message), "content": str(getattr(message, "content", ""))}
            for message in prompt.to_messages()
        ]
    elif isinstance(prompt, list):
        messages = []
        for item in prompt:
            if isinstance(item, dict):
                messages.append({
                    "role": str(item.get("role", "user")),
                    "content": str(item.get("content", "")),
                })
            else:
                messages.append({
                    "role": _message_role(item),
                    "content": str(getattr(item, "content", item)),
                })
    else:
        messages = [{"role": "user", "content": str(prompt)}]

    joined = " ".join(message["content"] for message in messages).lower()
    if "json" not in joined:
        messages.insert(0, {
            "role": "system",
            "content": "Return only valid JSON that matches the requested schema.",
        })
    return messages


def _split_anthropic_messages(messages: list[dict]) -> tuple[str, list[dict[str, str]]]:
    """Split OpenAI-style chat messages into Anthropic's (system, messages) shape.

    Anthropic takes system prompts as a top-level ``system`` argument rather than
    a message with ``role: "system"``, and only accepts ``user`` / ``assistant``
    turns. System messages are concatenated; everything else maps to a user turn
    unless it is explicitly an assistant turn.
    """
    system_parts: list[str] = []
    convo: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        if role == "system":
            if content:
                system_parts.append(content)
            continue
        convo.append({"role": "assistant" if role == "assistant" else "user", "content": content})
    if not convo:
        convo = [{"role": "user", "content": "\n\n".join(system_parts) or "Continue."}]
    return "\n\n".join(system_parts), convo


def _extract_json_object(content: str) -> dict | None:
    """Parse a JSON object, including common fenced-code responses."""
    if not content:
        return None

    stripped = content.strip()
    if stripped.startswith("```"):
        parsed = extract_json_from_response(stripped)
        if parsed is not None:
            return parsed

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(stripped[start:end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _format_exception(exc: Exception) -> str:
    """Include SDK wrapper errors plus their lower-level transport cause."""
    parts: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).strip() or repr(current)
        label = current.__class__.__name__
        parts.append(f"{label}: {message}" if label not in message else message)
        current = current.__cause__ or current.__context__
    return " | caused by: ".join(parts)


def _api_key(api_keys: dict | None, name: str) -> str:
    _ensure_env_loaded()
    value = (api_keys or {}).get(name) or os.getenv(name)
    if not value:
        raise ValueError(f"{name} not found. Set {name} in .env.")
    return value


def required_api_key_name(provider: str | object) -> str | None:
    """Return the environment key required for a provider, without reading it."""

    prov = _provider_name(provider)
    if prov == "azure openai":
        return "AZURE_OPENAI_API_KEY"
    return _PROVIDER_KEYS.get(prov)


def provider_has_credentials(provider: str | object, api_keys: dict | None = None) -> bool:
    """Credential preflight used by the browser UI. Never exposes secret values."""

    _ensure_env_loaded()
    prov = _provider_name(provider)
    if prov == "ollama":
        return True
    if prov == "azure openai":
        return bool(
            (api_keys or {}).get("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
        ) and bool(os.getenv("AZURE_OPENAI_ENDPOINT")) and bool(
            os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or os.getenv("MODEL_NAME")
        )
    key_name = required_api_key_name(prov)
    return bool(key_name and ((api_keys or {}).get(key_name) or os.getenv(key_name)))


def _fallback_model_config(model_provider: str | object, model_name: str) -> tuple[str, str] | None:
    """Fallback target for quota/auth failures.

    Defaults to MiniMax because this workspace is configured for it and it is
    OpenAI-compatible. Env overrides keep the behavior deployment-friendly.
    """

    _ensure_env_loaded()
    fallback_provider = os.getenv("NOVA_LLM_FALLBACK_PROVIDER", "MiniMax")
    fallback_model = os.getenv("NOVA_LLM_FALLBACK_MODEL", "MiniMax-M2.7")
    if _provider_name(fallback_provider) == _provider_name(model_provider):
        return None
    if not provider_has_credentials(fallback_provider):
        return None
    return fallback_model, fallback_provider


def _should_try_fallback(exc: Exception) -> bool:
    text = _format_exception(exc).lower()
    return any(
        marker in text
        for marker in (
            "insufficient_quota",
            "exceeded your current quota",
            "quota",
            "billing",
            "429",
            "rate limit",
            "authentication",
            "api key",
        )
    )


def _openai_compatible_config(provider: str, api_keys: dict | None) -> tuple[str, str | None, dict[str, str]]:
    provider = provider.lower()
    if provider == "openai":
        return _api_key(api_keys, "OPENAI_API_KEY"), os.getenv("OPENAI_API_BASE") or None, {}
    if provider == "openrouter":
        headers = {}
        if os.getenv("YOUR_SITE_URL"):
            headers["HTTP-Referer"] = os.getenv("YOUR_SITE_URL", "")
        if os.getenv("YOUR_SITE_NAME"):
            headers["X-Title"] = os.getenv("YOUR_SITE_NAME", "Nova Trader")
        return _api_key(api_keys, "OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1", headers
    if provider == "deepseek":
        return _api_key(api_keys, "DEEPSEEK_API_KEY"), "https://api.deepseek.com", {}
    if provider == "groq":
        return _api_key(api_keys, "GROQ_API_KEY"), "https://api.groq.com/openai/v1", {}
    if provider == "minimax":
        return (
            _api_key(api_keys, "MINIMAX_API_KEY"),
            os.getenv("MINIMAX_API_BASE", "https://api.minimax.io/v1"),
            {},
        )
    if provider == "xai":
        return _api_key(api_keys, "XAI_API_KEY"), "https://api.x.ai/v1", {}
    raise ValueError(f"Provider {provider!r} is not OpenAI-compatible in this adapter.")


def _call_openai_compatible_json(
    *,
    prompt: object,
    pydantic_model: type[BaseModel],
    model_name: str,
    model_provider: str,
    api_keys: dict | None,
    seed: int | None,
) -> tuple[BaseModel, dict]:
    """Call OpenAI or an OpenAI-compatible endpoint. Returns (parsed, telemetry)."""
    from openai import OpenAI

    api_key, base_url, default_headers = _openai_compatible_config(model_provider, api_keys)

    client = OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers or None)
    kwargs = {
        "model": model_name,
        "messages": _prompt_to_openai_messages(prompt),
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    if seed is not None:
        kwargs["seed"] = seed
    response = client.chat.completions.create(**kwargs)
    msg = response.choices[0].message
    content = msg.content or ""
    telemetry = {
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "prompt_tokens": getattr(response.usage, "prompt_tokens", None) if response.usage else None,
        "completion_tokens": getattr(response.usage, "completion_tokens", None) if response.usage else None,
        "response_content": content,
        # Present on reasoning models (o1/o3/DeepSeek-R1); ignored otherwise.
        "reasoning_content": getattr(msg, "reasoning_content", None),
    }
    parsed = _extract_json_object(content)
    if parsed is None:
        raise ValueError(f"Model returned non-JSON content: {content!r}")
    return pydantic_model.model_validate(parsed), telemetry


def _call_anthropic_json(
    *,
    prompt: object,
    pydantic_model: type[BaseModel],
    model_name: str,
    api_keys: dict | None,
) -> tuple[BaseModel, dict]:
    """Call the Anthropic Messages API with structured output. Returns (parsed, telemetry).

    Uses ``messages.parse`` so the response is validated against ``pydantic_model``
    server-side; the SDK strips JSON-schema constraints the structured-output API
    does not support and re-validates them client-side. No sampling params or
    ``thinking`` config are sent — those 400 on Opus 4.8 and aren't needed for
    short structured extractions.
    """
    import anthropic

    api_key = _api_key(api_keys, "ANTHROPIC_API_KEY")
    system, messages = _split_anthropic_messages(_prompt_to_openai_messages(prompt))

    client = anthropic.Anthropic(api_key=api_key)
    kwargs: dict[str, Any] = {
        "model": model_name or _ANTHROPIC_DEFAULT_MODEL,
        "max_tokens": 8192,
        "messages": messages,
        "output_format": pydantic_model,
    }
    if system:
        kwargs["system"] = system

    response = client.messages.parse(**kwargs)
    if getattr(response, "stop_reason", None) == "refusal":
        raise ValueError("Anthropic declined the request (stop_reason=refusal).")

    parsed = getattr(response, "parsed_output", None)
    if parsed is None:
        raise ValueError("Anthropic returned no parseable structured output.")

    text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), "")
    usage = getattr(response, "usage", None)
    telemetry = {
        "system_fingerprint": None,
        "prompt_tokens": getattr(usage, "input_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "output_tokens", None) if usage else None,
        "response_content": text,
        "reasoning_content": None,
    }
    return parsed, telemetry


def _call_azure_openai_json(
    *,
    prompt: object,
    pydantic_model: type[BaseModel],
    model_name: str,
    api_keys: dict | None,
    seed: int | None,
) -> tuple[BaseModel, dict]:
    """Call Azure OpenAI with the official OpenAI SDK. Returns (parsed, telemetry)."""
    from openai import AzureOpenAI

    api_key = (api_keys or {}).get("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = model_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    if not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY not found. Set it in .env.")
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT not found. Set it in .env.")
    if not deployment:
        raise ValueError("Azure deployment missing. Set MODEL_NAME or AZURE_OPENAI_DEPLOYMENT_NAME.")

    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )
    kwargs = {
        "model": deployment,
        "messages": _prompt_to_openai_messages(prompt),
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    if seed is not None:
        kwargs["seed"] = seed
    response = client.chat.completions.create(**kwargs)
    msg = response.choices[0].message
    content = msg.content or ""
    telemetry = {
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "prompt_tokens": getattr(response.usage, "prompt_tokens", None) if response.usage else None,
        "completion_tokens": getattr(response.usage, "completion_tokens", None) if response.usage else None,
        "response_content": content,
        "reasoning_content": getattr(msg, "reasoning_content", None),
    }
    parsed = _extract_json_object(content)
    if parsed is None:
        raise ValueError(f"Azure OpenAI returned non-JSON content: {content!r}")
    return pydantic_model.model_validate(parsed), telemetry


def _call_ollama_json(
    *,
    prompt: object,
    pydantic_model: type[BaseModel],
    model_name: str,
    seed: int | None,
) -> tuple[BaseModel, dict]:
    """Call Ollama directly over HTTP. Returns (parsed, telemetry)."""
    ollama_host = os.getenv("OLLAMA_HOST", "localhost")
    base_url = os.getenv("OLLAMA_BASE_URL", f"http://{ollama_host}:11434").rstrip("/")
    options = {"temperature": 0}
    if seed is not None:
        options["seed"] = seed
    response = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model_name,
            "messages": _prompt_to_openai_messages(prompt),
            "format": "json",
            "stream": False,
            "options": options,
        },
        timeout=120,
    )
    response.raise_for_status()
    body = response.json()
    content = body.get("message", {}).get("content", "")
    telemetry = {
        "system_fingerprint": None,  # Ollama doesn't return one.
        "prompt_tokens": body.get("prompt_eval_count"),
        "completion_tokens": body.get("eval_count"),
        "response_content": content,
    }
    parsed = _extract_json_object(content)
    if parsed is None:
        raise ValueError(f"Ollama returned non-JSON content: {content!r}")
    return pydantic_model.model_validate(parsed), telemetry


def _call_json_model(
    *,
    prompt: object,
    pydantic_model: type[BaseModel],
    model_name: str,
    model_provider: str,
    api_keys: dict | None,
    seed: int | None,
) -> tuple[BaseModel, dict]:
    """Returns (parsed_model, telemetry_dict)."""
    provider = _provider_name(model_provider)
    if provider == "anthropic":
        return _call_anthropic_json(
            prompt=prompt,
            pydantic_model=pydantic_model,
            model_name=model_name,
            api_keys=api_keys,
        )
    if provider in {"openai", "openrouter", "deepseek", "groq", "minimax", "xai"}:
        return _call_openai_compatible_json(
            prompt=prompt,
            pydantic_model=pydantic_model,
            model_name=model_name,
            model_provider=provider,
            api_keys=api_keys,
            seed=seed,
        )
    if provider == "azure openai":
        return _call_azure_openai_json(
            prompt=prompt,
            pydantic_model=pydantic_model,
            model_name=model_name,
            api_keys=api_keys,
            seed=seed,
        )
    if provider == "ollama":
        return _call_ollama_json(
            prompt=prompt,
            pydantic_model=pydantic_model,
            model_name=model_name,
            seed=seed,
        )
    raise ValueError(
        f"Provider {model_provider!r} has no direct adapter yet. "
        "Use Anthropic, MiniMax, OpenAI, Azure OpenAI, OpenRouter, DeepSeek, Groq, xAI, or Ollama."
    )


def call_llm(
    prompt: Any,
    pydantic_model: type[BaseModel],
    agent_name: str | None = None,
    state: StateLike | None = None,
    max_retries: int = 3,
    default_factory=None,
    *,
    seed: int | None = None,
    recorder=None,
    ticker: str | None = None,
) -> BaseModel:
    """Make an LLM call with retry + structured output validation + optional recording.

    Args:
        prompt: The prompt to send.
        pydantic_model: Pydantic model class to validate the output against.
        agent_name: Used for progress updates and per-agent model config lookup.
        state: Dict-like state carrying model config under metadata.
        max_retries: Retry budget on transient failures.
        default_factory: Builds a fallback object if all attempts fail.
        seed: Passed to the provider for reproducibility.
        recorder: If provided, every attempt is recorded (prompt, response,
            fingerprint, tokens, latency, error).
        ticker: Logged into the recorder for trace context.

    Returns:
        An instance of `pydantic_model`. On total failure returns a default.
    """
    import time

    if state and agent_name:
        model_name, model_provider = get_agent_model_config(state, agent_name)
    else:
        model_name = "MiniMax-M2.7"
        model_provider = "MiniMax"
    fallback_used = False

    api_keys = None
    if state:
        request = state.get("metadata", {}).get("request")
        if request and hasattr(request, 'api_keys'):
            api_keys = request.api_keys

    for attempt in range(max_retries):
        t0 = time.monotonic()
        telemetry: dict = {}
        error: str | None = None
        try:
            result, telemetry = _call_json_model(
                prompt=prompt,
                pydantic_model=pydantic_model,
                model_name=model_name,
                model_provider=model_provider,
                api_keys=api_keys,
                seed=seed,
            )
            latency_ms = (time.monotonic() - t0) * 1000.0
            if recorder is not None:
                recorder.append_llm_call(
                    agent_id=agent_name or "unknown",
                    ticker=ticker,
                    model=model_name,
                    provider=str(model_provider),
                    prompt=prompt,
                    response=telemetry.get("response_content", ""),
                    seed=seed,
                    system_fingerprint=telemetry.get("system_fingerprint"),
                    latency_ms=latency_ms,
                    prompt_tokens=telemetry.get("prompt_tokens"),
                    completion_tokens=telemetry.get("completion_tokens"),
                    attempt=attempt + 1,
                    error=None,
                )
            progress.add_tokens(
                agent_name or "unknown",
                telemetry.get("prompt_tokens") or 0,
                telemetry.get("completion_tokens") or 0,
            )
            progress.capture_reasoning(
                agent_name or "unknown",
                ticker,
                telemetry.get("reasoning_content"),
                telemetry.get("response_content"),
            )
            return result

        except Exception as e:
            error = _format_exception(e)
            latency_ms = (time.monotonic() - t0) * 1000.0
            if recorder is not None:
                recorder.append_llm_call(
                    agent_id=agent_name or "unknown",
                    ticker=ticker,
                    model=model_name,
                    provider=str(model_provider),
                    prompt=prompt,
                    response=telemetry.get("response_content", ""),
                    seed=seed,
                    system_fingerprint=telemetry.get("system_fingerprint"),
                    latency_ms=latency_ms,
                    prompt_tokens=telemetry.get("prompt_tokens"),
                    completion_tokens=telemetry.get("completion_tokens"),
                    attempt=attempt + 1,
                    error=error,
                )
            progress.add_tokens(
                agent_name or "unknown",
                telemetry.get("prompt_tokens") or 0,
                telemetry.get("completion_tokens") or 0,
            )
            fallback = _fallback_model_config(model_provider, model_name)
            if not fallback_used and fallback and _should_try_fallback(e):
                model_name, model_provider = fallback
                fallback_used = True
                if agent_name:
                    progress.update_status(agent_name, ticker, f"Switching to {model_provider}")
                continue
            if agent_name:
                progress.update_status(agent_name, ticker, f"Error - retry {attempt + 1}/{max_retries}")
            if attempt == max_retries - 1:
                print(f"Error in LLM call after {max_retries} attempts: {e}")
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)

    return create_default_response(pydantic_model)


def create_default_response(model_class: type[BaseModel]) -> BaseModel:
    """Creates a safe default response based on the model's fields."""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        else:
            # For other types (like Literal), try to use the first allowed value
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None

    return model_class(**default_values)


class _ThinkTagSplitter:
    """Splits streamed content into ('reasoning', t) inside <think>/<thinking> tags and
    ('answer', t) outside. Buffers across chunk boundaries so a tag split between two
    deltas is still recognised."""

    OPEN = ("<think>", "<thinking>")
    CLOSE = ("</think>", "</thinking>")
    _MAX_TAG = 11  # len("<thinking>")

    def __init__(self):
        self._buf = ""
        self._in_think = False

    @staticmethod
    def _first_tag(text: str, tags: tuple[str, ...]) -> tuple[int, str | None]:
        best, found = -1, None
        for tag in tags:
            idx = text.find(tag)
            if idx != -1 and (best == -1 or idx < best):
                best, found = idx, tag
        return best, found

    @classmethod
    def _safe_len(cls, text: str, tags: tuple[str, ...]) -> int:
        """How much of `text` is safe to emit without splitting a not-yet-complete tag."""
        hold = 0
        for k in range(1, min(cls._MAX_TAG, len(text) + 1)):
            suffix = text[-k:]
            if any(tag.startswith(suffix) for tag in tags):
                hold = k
        return len(text) - hold

    def feed(self, text: str) -> list[tuple[str, str]]:
        self._buf += text
        out: list[tuple[str, str]] = []
        while self._buf:
            if not self._in_think:
                idx, tag = self._first_tag(self._buf, self.OPEN)
                if idx == -1:
                    safe = self._safe_len(self._buf, self.OPEN)
                    if safe > 0:
                        out.append(("answer", self._buf[:safe]))
                        self._buf = self._buf[safe:]
                    break
                if idx > 0:
                    out.append(("answer", self._buf[:idx]))
                self._buf = self._buf[idx + len(tag):]
                self._in_think = True
            else:
                idx, tag = self._first_tag(self._buf, self.CLOSE)
                if idx == -1:
                    safe = self._safe_len(self._buf, self.CLOSE)
                    if safe > 0:
                        out.append(("reasoning", self._buf[:safe]))
                        self._buf = self._buf[safe:]
                    break
                if idx > 0:
                    out.append(("reasoning", self._buf[:idx]))
                self._buf = self._buf[idx + len(tag):]
                self._in_think = False
        return out

    def flush(self) -> list[tuple[str, str]]:
        if not self._buf:
            return []
        kind = "reasoning" if self._in_think else "answer"
        out = [(kind, self._buf)]
        self._buf = ""
        return out


def stream_chat(
    messages: list[dict[str, str]],
    *,
    provider: str | object,
    model: str,
    api_keys: dict | None = None,
    temperature: float = 0.3,
):
    """Yield ``(channel, text)`` chunks from a free-form (non-JSON) chat completion.

    ``channel`` is ``"reasoning"`` for a model's thinking (``reasoning_content`` field or
    ``<think>...</think>`` tags) and ``"answer"`` for the user-facing reply. This lets the
    chat surface stream thinking into a small box and keep the final answer clean.
    Supports the OpenAI-compatible providers, Azure OpenAI, and Ollama.
    """
    try:
        yield from _stream_chat_once(
            messages,
            provider=provider,
            model=model,
            api_keys=api_keys,
            temperature=temperature,
        )
        return
    except Exception as exc:
        fallback = _fallback_model_config(provider, model)
        if not fallback or not _should_try_fallback(exc):
            raise
        fallback_model, fallback_provider = fallback
        yield from _stream_chat_once(
            messages,
            provider=fallback_provider,
            model=fallback_model,
            api_keys=api_keys,
            temperature=temperature,
        )


def _stream_chat_once(
    messages: list[dict[str, str]],
    *,
    provider: str | object,
    model: str,
    api_keys: dict | None = None,
    temperature: float = 0.3,
):
    prov = _provider_name(provider)

    def _emit_openai_stream(stream):
        splitter = _ThinkTagSplitter()
        for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
            if reasoning:
                yield ("reasoning", reasoning)
            content = getattr(delta, "content", None)
            if content:
                yield from splitter.feed(content)
        yield from splitter.flush()

    if prov == "anthropic":
        import anthropic

        api_key = _api_key(api_keys, "ANTHROPIC_API_KEY")
        system, convo = _split_anthropic_messages(messages)
        client = anthropic.Anthropic(api_key=api_key)
        chat_model = model or _ANTHROPIC_DEFAULT_MODEL
        stream_kwargs: dict[str, Any] = {
            "model": chat_model,
            "max_tokens": 16000,
            "messages": convo,
        }
        # Adaptive thinking with summarized display so the chat surface can
        # stream Claude's reasoning into its own channel. Haiku rejects the
        # thinking param (and grounded Q&A doesn't need it). No temperature —
        # sampling params 400 on Opus 4.8.
        if "haiku" not in chat_model:
            stream_kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        if system:
            stream_kwargs["system"] = system
        with client.messages.stream(**stream_kwargs) as stream:
            for event in stream:
                if event.type != "content_block_delta":
                    continue
                delta = event.delta
                delta_type = getattr(delta, "type", None)
                if delta_type == "thinking_delta" and getattr(delta, "thinking", None):
                    yield ("reasoning", delta.thinking)
                elif delta_type == "text_delta" and getattr(delta, "text", None):
                    yield ("answer", delta.text)
        return

    if prov in {"openai", "openrouter", "deepseek", "groq", "minimax", "xai"}:
        from openai import OpenAI

        api_key, base_url, headers = _openai_compatible_config(prov, api_keys)
        client = OpenAI(api_key=api_key, base_url=base_url, default_headers=headers or None)
        stream = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, stream=True,
        )
        yield from _emit_openai_stream(stream)
        return

    if prov == "azure openai":
        from openai import AzureOpenAI

        api_key = (api_keys or {}).get("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = model or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        if not (api_key and endpoint and deployment):
            raise ValueError("Azure OpenAI needs AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and a deployment.")
        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )
        stream = client.chat.completions.create(
            model=deployment, messages=messages, temperature=temperature, stream=True,
        )
        yield from _emit_openai_stream(stream)
        return

    if prov == "ollama":
        ollama_host = os.getenv("OLLAMA_HOST", "localhost")
        base_url = os.getenv("OLLAMA_BASE_URL", f"http://{ollama_host}:11434").rstrip("/")
        response = requests.post(
            f"{base_url}/api/chat",
            json={"model": model, "messages": messages, "stream": True, "options": {"temperature": temperature}},
            stream=True,
            timeout=120,
        )
        response.raise_for_status()
        splitter = _ThinkTagSplitter()
        for line in response.iter_lines():
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = payload.get("message", {})
            reasoning = message.get("reasoning_content") or message.get("thinking")
            if reasoning:
                yield ("reasoning", reasoning)
            content = message.get("content")
            if content:
                yield from splitter.feed(content)
        yield from splitter.flush()
        return

    raise ValueError(f"Provider {provider!r} does not support streaming chat in this adapter.")


def extract_json_from_response(content: str) -> dict | None:
    """Extracts JSON from markdown-formatted response."""
    try:
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7 :]  # Skip past ```json
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                return json.loads(json_text)
    except Exception as e:
        print(f"Error extracting JSON from response: {e}")
    return None


def get_agent_model_config(state, agent_name):
    """
    Get model configuration for a specific agent from the state.
    Falls back to global model configuration if agent-specific config is not available.
    Always returns valid model_name and model_provider values.
    """
    request = state.get("metadata", {}).get("request")
    
    if request and hasattr(request, 'get_agent_model_config'):
        # Get agent-specific model configuration
        model_name, model_provider = request.get_agent_model_config(agent_name)
        # Ensure we have valid values
        if model_name and model_provider:
            return model_name, model_provider.value if hasattr(model_provider, 'value') else str(model_provider)
    
    # Fall back to global configuration (system defaults)
    model_name = state.get("metadata", {}).get("model_name") or "MiniMax-M2.7"
    model_provider = state.get("metadata", {}).get("model_provider") or "MiniMax"
    
    # Convert enum to string if necessary
    if hasattr(model_provider, 'value'):
        model_provider = model_provider.value
    
    return model_name, model_provider
