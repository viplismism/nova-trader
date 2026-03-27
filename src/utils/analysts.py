"""Constants and utilities related to analysts configuration.

Template-based agents (investor personas) are loaded from YAML configs.
Pure-calculation agents (technicals, fundamentals, etc.) remain as Python modules.
"""

# Pure-calculation agents — kept as Python (no LLM persona, just math)
from src.agents.fundamentals import fundamentals_analyst_agent
from src.agents.sentiment import sentiment_analyst_agent
from src.agents.technicals import technical_analyst_agent
from src.agents.valuation import valuation_analyst_agent
from src.agents.news_sentiment import news_sentiment_agent
from src.agents.growth_agent import growth_analyst_agent

# Template-based agent loader
from src.agents.loader import load_all_template_agents


def _build_analyst_config():
    """Build the unified ANALYST_CONFIG from templates + hardcoded Python agents."""
    config = {}

    # 1. Load all template-based agents (investor personas from YAML)
    for key, info in load_all_template_agents().items():
        config[key] = {
            "display_name": info["display_name"],
            "description": info["description"],
            "investing_style": info["config"].get("persona", {}).get("philosophy", ""),
            "agent_func": info["agent_fn"],
            "type": info["type"],
            "order": info["order"],
        }

    # 2. Pure-calculation agents (stay as Python — no LLM persona)
    python_agents = {
        "technical_analyst": {
            "display_name": "Technical Analyst",
            "description": "Chart Pattern Specialist",
            "investing_style": "Focuses on chart patterns and market trends using technical indicators and price action analysis.",
            "agent_func": technical_analyst_agent,
            "type": "analyst",
            "order": 50,
        },
        "fundamentals_analyst": {
            "display_name": "Fundamentals Analyst",
            "description": "Financial Statement Specialist",
            "investing_style": "Delves into financial statements and economic indicators to assess intrinsic value.",
            "agent_func": fundamentals_analyst_agent,
            "type": "analyst",
            "order": 51,
        },
        "growth_analyst": {
            "display_name": "Growth Analyst",
            "description": "Growth Specialist",
            "investing_style": "Analyzes growth trends and valuation to identify growth opportunities.",
            "agent_func": growth_analyst_agent,
            "type": "analyst",
            "order": 52,
        },
        "news_sentiment_analyst": {
            "display_name": "News Sentiment Analyst",
            "description": "News Sentiment Specialist",
            "investing_style": "Uses LLM to classify news headlines and aggregate sentiment signals.",
            "agent_func": news_sentiment_agent,
            "type": "analyst",
            "order": 53,
        },
        "sentiment_analyst": {
            "display_name": "Sentiment Analyst",
            "description": "Market Sentiment Specialist",
            "investing_style": "Gauges market sentiment via insider trades and news to identify behavioral signals.",
            "agent_func": sentiment_analyst_agent,
            "type": "analyst",
            "order": 54,
        },
        "valuation_analyst": {
            "display_name": "Valuation Analyst",
            "description": "Company Valuation Specialist",
            "investing_style": "Determines fair value using DCF, comparable analysis, and financial metrics.",
            "agent_func": valuation_analyst_agent,
            "type": "analyst",
            "order": 55,
        },
    }
    config.update(python_agents)

    return config


# Build config once at import time
ANALYST_CONFIG = _build_analyst_config()

# Derive ANALYST_ORDER from ANALYST_CONFIG for backwards compatibility
ANALYST_ORDER = [(config["display_name"], key) for key, config in sorted(ANALYST_CONFIG.items(), key=lambda x: x[1]["order"])]


def get_analyst_nodes():
    """Get the mapping of analyst keys to their (node_name, agent_func) tuples."""
    return {key: (f"{key}_agent", config["agent_func"]) for key, config in ANALYST_CONFIG.items()}


def get_agents_list():
    """Get the list of agents for API responses."""
    return [
        {
            "key": key,
            "display_name": config["display_name"],
            "description": config["description"],
            "investing_style": config["investing_style"],
            "order": config["order"]
        }
        for key, config in sorted(ANALYST_CONFIG.items(), key=lambda x: x[1]["order"])
    ]
