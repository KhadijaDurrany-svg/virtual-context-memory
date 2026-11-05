"""knapmem: a benchmarked, deterministic retention policy for LLM working memory.

Given a stream of context items and a fixed token budget, decide what to keep,
summarize, or evict, and measure how well needed facts survive when later
queried, against naive baselines, with honest confidence intervals and a
significance test. Fully offline and reproducible.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
