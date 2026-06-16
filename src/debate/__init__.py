"""Research-desk debate mode (additive).

A supervisor decomposes a question into per-specialist mandates; four specialists
research filings/web in parallel; a bear challenges the thesis; a synthesizer
produces a cited, conviction-scored memo. Runs alongside Nova's deterministic
analyst engine — it does not replace it.
"""

from src.debate.engine import USAGE, run_debate

__all__ = ["run_debate", "USAGE"]
