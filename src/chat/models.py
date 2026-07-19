"""Small data contracts used by the chat CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from src.registry import default_agent_ids

DEFAULT_AGENTS = default_agent_ids()

PROVIDERS = [
    "Anthropic",
    "OpenAI",
    "MiniMax",
    "DeepSeek",
    "Groq",
    "xAI",
    "OpenRouter",
    "Azure OpenAI",
    "Ollama",
]


@dataclass
class ChatSettings:
    provider: str
    model: str
    portfolio_mode: Literal["research", "long_only", "long_short"] = "research"
    agents: list[str] = field(default_factory=lambda: DEFAULT_AGENTS.copy())
    initial_cash: float = 100_000.0
    margin_requirement: float = 0.5
    router_provider: str | None = None
    router_model: str | None = None
    show_reasoning: bool = True  # per-analyst LLM "why" narration; toggled by /reasoning


@dataclass
class ChatEvent:
    kind: str
    title: str
    body: str = ""


class IntentRoute(BaseModel):
    route: Literal["analyze", "chat", "details"] = "chat"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
