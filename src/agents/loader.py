"""Agent loader: reads YAML configs and creates agent functions via the harness.

This module:
  1. Scans templates/configs/ for .yaml files
  2. Parses each into a config dict
  3. Calls make_agent_from_config() to produce a callable agent function
  4. Provides a registry mapping agent_key -> (node_name, agent_fn)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import yaml

from src.agents.harness import make_agent_from_config

CONFIGS_DIR = Path(__file__).parent / "templates" / "configs"


def _load_yaml(path: Path) -> dict:
    """Load and parse a YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_all_template_agents() -> dict[str, dict]:
    """Load all YAML agent configs from the configs directory.

    Returns: {agent_key: {"config": dict, "agent_fn": callable, ...}}
    """
    agents = {}

    if not CONFIGS_DIR.exists():
        return agents

    for yaml_path in sorted(CONFIGS_DIR.glob("*.yaml")):
        try:
            config = _load_yaml(yaml_path)
            agent_key = config["id"]
            agent_fn = make_agent_from_config(config)

            agents[agent_key] = {
                "config": config,
                "agent_fn": agent_fn,
                "display_name": config.get("display_name", agent_key),
                "description": config.get("description", ""),
                "order": config.get("order", 99),
                "type": config.get("type", "analyst"),
            }
        except Exception as e:
            print(f"[Loader] Failed to load agent config {yaml_path.name}: {e}")

    return agents


def get_template_agent_nodes() -> dict[str, tuple[str, Callable]]:
    """Get template agents in the format expected by the pipeline.

    Returns: {agent_key: (node_name, agent_fn)}
    """
    agents = load_all_template_agents()
    return {
        key: (f"{key}_agent", info["agent_fn"])
        for key, info in agents.items()
    }


def get_template_agent_config_list() -> list[dict]:
    """Get template agent configs for display/API purposes."""
    agents = load_all_template_agents()
    return [
        {
            "key": key,
            "display_name": info["display_name"],
            "description": info["description"],
            "investing_style": info["config"].get("persona", {}).get("philosophy", ""),
            "order": info["order"],
        }
        for key, info in sorted(agents.items(), key=lambda x: x[1]["order"])
    ]
