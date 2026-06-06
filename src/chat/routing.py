"""Intent routing and ticker parsing helpers for Nova chat."""

from __future__ import annotations

import os
import re

from src.chat.models import IntentRoute, PROVIDERS


def normalize_provider(value: str) -> str:
    compact = value.strip().lower().replace("_", " ").replace("-", " ")
    aliases = {
        "openai": "OpenAI",
        "mini max": "MiniMax",
        "minimax": "MiniMax",
        "deepseek": "DeepSeek",
        "deep seek": "DeepSeek",
        "groq": "Groq",
        "xai": "xAI",
        "x ai": "xAI",
        "openrouter": "OpenRouter",
        "open router": "OpenRouter",
        "azure openai": "Azure OpenAI",
        "azure": "Azure OpenAI",
        "ollama": "Ollama",
    }
    return aliases.get(compact, value.strip())


def extract_tickers(text: str) -> list[str]:
    aliases = {
        "AAPL": "AAPL",
        "APPLE": "AAPL",
        "AMZN": "AMZN",
        "AMAZON": "AMZN",
        "AMD": "AMD",
        "GOOG": "GOOG",
        "GOOGL": "GOOGL",
        "GOOGLE": "GOOGL",
        "ALPHABET": "GOOGL",
        "META": "META",
        "FACEBOOK": "META",
        "MSFT": "MSFT",
        "MICROSOFT": "MSFT",
        "NFLX": "NFLX",
        "NETFLIX": "NFLX",
        "NVDA": "NVDA",
        "NVIDIA": "NVDA",
        "NVDIA": "NVDA",
        "TSLA": "TSLA",
        "TESLA": "TSLA",
    }
    ignored = {
        "A", "AN", "ANALYZE", "ANALYSE", "RUN", "SHOW", "RERUN", "MODEL",
        "PROVIDER", "AGENTS", "DETAIL", "DETAILS", "WHY", "HELP", "EXIT",
        "QUIT", "SET", "SETUP", "USE", "WITH", "AND", "ARE", "AS", "AT",
        "BACK", "BE", "BUY", "CAN", "CALL", "CALLS", "DO", "DOES", "GOING",
        "HAPPENING", "HAS", "HAVE", "HOW", "I", "IF", "IN", "IS", "IT",
        "ITS", "KNOW", "ME", "MY", "OF", "ON", "OPTION", "OPTIONS", "OR",
        "OUT", "PUT", "PUTS", "SELL", "SHORT", "SHOULD", "S", "STOCK",
        "THAT", "THIS", "TO", "TRADE", "VIEW", "WANT", "WE", "WHAT",
        "WHATS", "WHEN", "WHERE", "WHICH", "WHO", "WOULD", "FOR", "THE",
        "LAST", "YOU", "YOUR", "YOURS",
    }
    found = re.findall(r"\b[A-Za-z][A-Za-z0-9.]{0,10}\b", text)
    tickers: list[str] = []
    for raw in found:
        token = raw.upper()
        if token in ignored:
            continue
        mapped = aliases.get(token)
        if not mapped and raw == token and len(token) <= 5:
            mapped = token
        if mapped and mapped not in tickers:
            tickers.append(mapped)
    return tickers


def is_analysis_prompt(text: str) -> bool:
    lower = text.lower()
    if any(word in lower for word in ("analyze", "analyse", "recommend", "evaluate", "check", "run ")):
        return True
    if extract_tickers(text) and "," in text and not re.search(r"\b(what|why|how|who|where|when)\b", lower):
        return True
    compact = re.sub(r"[\s,]+", "", text)
    return bool(compact) and bool(re.fullmatch(r"[A-Za-z0-9.,\s]+", text)) and any(
        token.isupper() for token in re.findall(r"\b[A-Z][A-Z0-9.]{0,5}\b", text)
    )


def fallback_ticker_route(text: str, tickers: list[str]) -> IntentRoute:
    """Conservative fallback when the tiny router model is unavailable."""
    lower = text.lower()
    if tickers and any(word in lower for word in ("analyze", "analyse", "recommend", "evaluate", "run ")):
        return IntentRoute(route="analyze", confidence=0.95, reason="explicit analysis command")
    if tickers and lower.startswith(("details", "detail", "why", "explain")):
        return IntentRoute(route="details", confidence=0.95, reason="explicit details command")
    if is_analysis_prompt(text):
        return IntentRoute(route="analyze", confidence=0.8, reason="ticker-list shorthand")
    if tickers and any(word in lower for word in ("short", "put", "call", "option", "buy", "sell", "hedge", "risk")):
        return IntentRoute(route="analyze", confidence=0.65, reason="ticker trade question")
    return IntentRoute(route="chat", confidence=0.3, reason="router unavailable")


def default_router_model(provider: str, model: str) -> tuple[str, str]:
    env_provider = os.getenv("NOVA_ROUTER_PROVIDER")
    env_model = os.getenv("NOVA_ROUTER_MODEL")
    if env_provider and env_model:
        return normalize_provider(env_provider), env_model

    provider = normalize_provider(provider)
    if provider == "OpenAI":
        return "OpenAI", "gpt-4.1-mini"
    if provider == "MiniMax":
        return "MiniMax", "MiniMax-M2.7-highspeed"
    if provider == "DeepSeek":
        return "DeepSeek", "deepseek-chat"
    return provider, model


def provider_choices() -> str:
    return ", ".join(PROVIDERS)


def model_choices_for(provider: str) -> list[str]:
    try:
        from src.llm.models import AVAILABLE_MODELS, OLLAMA_MODELS
    except Exception:
        return []

    return [
        model.model_name
        for model in AVAILABLE_MODELS + OLLAMA_MODELS
        if model.provider.value.lower() == provider.lower() and model.model_name
    ]
