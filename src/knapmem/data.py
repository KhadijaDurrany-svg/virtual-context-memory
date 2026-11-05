"""Seeded synthetic workload generation.

A workload is a merged timeline of *item* events (new content arriving into the
context window) and *query* events (a later request that needs a specific fact
to still be present). Everything here is deterministic given a seed: the same
seed always yields byte-identical workloads, which is what makes the benchmark
reproducible and lets us checksum the dataset.

The generative story, kept deliberately simple and auditable:

* Each item carries exactly one *fact* (identified by the item's own id).
* An item has a full token cost and a cheaper summary token cost. Summaries are
  lossy: a summary preserves the item's fact only when ``summarizable`` is True.
* Whether an item is ever queried is correlated with a *salience* feature the
  policy can observe, but the correlation is intentionally noisy. A policy that
  trusts salience blindly will therefore make mistakes, so no policy can be
  perfect. This is what keeps measured recall realistic rather than suspicious.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np

# Event kinds on the timeline.
ITEM = "item"
QUERY = "query"


@dataclass(frozen=True)
class Item:
    """A single piece of content arriving into the context window."""

    item_id: int
    cost_full: int
    cost_summary: int
    summarizable: bool
    # Observable (noisy) feature in [0, 1] that loosely predicts future queries.
    salience: float


@dataclass(frozen=True)
class Event:
    """A timeline entry: either an item arrival or a query."""

    kind: str
    turn: int
    # For ITEM events this is the arriving item; for QUERY events it is None.
    item: Item | None = None
    # For QUERY events this is the queried item_id; for ITEM events it is -1.
    query_item_id: int = -1


@dataclass(frozen=True)
class Scenario:
    """One independent workload: a sequence of events plus book-keeping."""

    scenario_id: int
    events: list[Event] = field(default_factory=list)
    n_items: int = 0
    n_queries: int = 0


@dataclass(frozen=True)
class DatasetConfig:
    """All knobs for workload generation. Defaults give a tight-budget regime."""

    seed: int = 7
    n_scenarios: int = 200
    items_per_scenario: int = 60
    queries_per_scenario: int = 40
    cost_min: int = 20
    cost_max: int = 120
    summary_ratio: float = 0.35
    summarizable_prob: float = 0.6
    # Probability an item is "truly important" (more likely to be queried).
    important_prob: float = 0.35
    # Strength of correlation between salience feature and true importance.
    salience_noise: float = 0.35
    # Fraction of queries that target recent items (vs. salience-weighted).
    recency_query_frac: float = 0.4

    def validate(self) -> None:
        if self.n_scenarios <= 0:
            raise ValueError("n_scenarios must be positive")
        if self.items_per_scenario <= 1:
            raise ValueError("items_per_scenario must be > 1")
        if self.queries_per_scenario <= 0:
            raise ValueError("queries_per_scenario must be positive")
        if not 0 < self.cost_min <= self.cost_max:
            raise ValueError("require 0 < cost_min <= cost_max")
        if not 0.0 < self.summary_ratio < 1.0:
            raise ValueError("summary_ratio must be in (0, 1)")
        for name in ("summarizable_prob", "important_prob", "recency_query_frac"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.salience_noise < 0.0:
            raise ValueError("salience_noise must be non-negative")


def _make_scenario(scenario_id: int, cfg: DatasetConfig, rng: np.random.Generator) -> Scenario:
    n = cfg.items_per_scenario
    costs_full = rng.integers(cfg.cost_min, cfg.cost_max + 1, size=n)
    summarizable = rng.random(n) < cfg.summarizable_prob
    important = rng.random(n) < cfg.important_prob

    # Salience is the (clipped) true-importance signal plus gaussian noise, so it
    # is informative but not a perfect oracle.
    noise = rng.normal(0.0, cfg.salience_noise, size=n)
    salience = np.clip(important.astype(float) * 0.7 + 0.15 + noise, 0.0, 1.0)

    items: list[Item] = []
    for i in range(n):
        cost_full = int(costs_full[i])
        cost_summary = max(1, int(round(cost_full * cfg.summary_ratio)))
        items.append(
            Item(
                item_id=i,
                cost_full=cost_full,
                cost_summary=cost_summary,
                summarizable=bool(summarizable[i]),
                salience=float(salience[i]),
            )
        )

    # Query targeting weights: important items are far more likely to be asked
    # about; a configurable share of queries instead favours recent items.
    importance_weight = np.where(important, 5.0, 1.0)
    importance_weight = importance_weight / importance_weight.sum()
    recency_weight = np.arange(1, n + 1, dtype=float)
    recency_weight = recency_weight / recency_weight.sum()

    events: list[Event] = [Event(kind=ITEM, turn=i, item=items[i]) for i in range(n)]

    # Queries occur after their target has arrived. We schedule each query at a
    # turn strictly greater than the target's arrival turn.
    q = cfg.queries_per_scenario
    use_recency = rng.random(q) < cfg.recency_query_frac
    query_events: list[Event] = []
    for k in range(q):
        weights = recency_weight if use_recency[k] else importance_weight
        target = int(rng.choice(n, p=weights))
        # Place the query somewhere strictly after arrival (inclusive of the end).
        earliest = target + 1
        turn = int(rng.integers(earliest, n + 1)) if earliest <= n else n
        query_events.append(Event(kind=QUERY, turn=turn, query_item_id=target))

    # Merge item and query events into a stable timeline. Items use their arrival
    # turn; queries fire *after* the item at the same turn index, so queries sort
    # after items at equal turn via a secondary key.
    def sort_key(e: Event) -> tuple[int, int, int]:
        secondary = 0 if e.kind == ITEM else 1
        tertiary = e.item.item_id if e.item is not None else e.query_item_id
        return (e.turn, secondary, tertiary)

    merged = sorted(events + query_events, key=sort_key)
    return Scenario(scenario_id=scenario_id, events=merged, n_items=n, n_queries=q)


def generate(cfg: DatasetConfig) -> list[Scenario]:
    """Generate the full list of scenarios for a config (deterministic)."""

    cfg.validate()
    root = np.random.default_rng(cfg.seed)
    # Derive an independent child seed per scenario so that scenario k is stable
    # regardless of how many scenarios precede it.
    child_seeds = root.integers(0, 2**32 - 1, size=cfg.n_scenarios)
    return [
        _make_scenario(i, cfg, np.random.default_rng(int(child_seeds[i])))
        for i in range(cfg.n_scenarios)
    ]


def _event_to_jsonable(e: Event) -> dict:
    row: dict = {"kind": e.kind, "turn": e.turn}
    if e.item is not None:
        row["item"] = asdict(e.item)
    if e.kind == QUERY:
        row["query_item_id"] = e.query_item_id
    return row


def checksum(scenarios: list[Scenario]) -> str:
    """Stable SHA-256 over the canonical JSON of all scenarios.

    Used in the benchmark manifest so a result can be tied to the exact dataset
    that produced it.
    """

    payload = [
        {
            "scenario_id": s.scenario_id,
            "n_items": s.n_items,
            "n_queries": s.n_queries,
            "events": [_event_to_jsonable(e) for e in s.events],
        }
        for s in scenarios
    ]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
