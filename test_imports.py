"""Quick import verification for the rebuilt core."""

import sys

sys.path.insert(0, "/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader")

from src.core import CoreConfig, PortfolioState, TradingPipeline, TradingService


config = CoreConfig()
portfolio = PortfolioState(cash=100_000)
pipeline = TradingPipeline(config=config)
service = TradingService(config=config)

assert config.initial_cash == 100_000.0
assert portfolio.cash == 100_000
assert pipeline.config.signal_models == ("factor", "fundamentals", "sentiment")
assert service.config.execution_mode == "dry_run"

print("ALL_TESTS_PASSED")
