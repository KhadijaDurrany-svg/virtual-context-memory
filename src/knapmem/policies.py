"""Retention policies: given a working set and a token budget, decide the state
(FULL / SUMMARY / EVICTED) of every item so total cost fits the budget.

All policies are deterministic. The interesting one is :class:`KnapsackPolicy`,
which treats retention as a multiple-choice knapsack over observable features
and may *summarize* an item as a middle ground between keeping and evicting.
The baselines (recency/FIFO, random, LRU) never summarize, matching how naive
systems behave.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .data import Item


class State(str, Enum):
    FULL = "full"
    SUMMARY = "summary"
    EVICTED = "evicted"


@dataclass
class Slot:
    """Mutable per-item working-set entry."""

    item: Item
    state: State
    arrival_turn: int
    last_touch_turn: int
    access_count: int = 0

    def cost(self) -> int:
        if self.state is State.FULL:
            return self.item.cost_full
        if self.state is State.SUMMARY:
            return self.item.cost_summary
        return 0

    def retains_fact(self) -> bool:
        """Whether this slot currently preserves the item's fact."""

        if self.state is State.FULL:
            return True
        if self.state is State.SUMMARY:
            return self.item.summarizable
        return False


class Policy:
    """Base class. Subclasses implement :meth:`reduce`."""

    name: str = "base"

    def reduce(self, slots: list[Slot], budget: int, now: int) -> None:
        """Mutate ``slots`` in place so total cost <= ``budget``.

        Called whenever a freshly arrived item pushes total cost over budget.
        """

        raise NotImplementedError


def total_cost(slots: list[Slot]) -> int:
    return sum(s.cost() for s in slots)


class RecencyPolicy(Policy):
    """Keep the most recent items at full detail; evict the oldest first.

    This is the FIFO / recency-window / keep-all-then-truncate family: with a
    pure recency timeline these all reduce to the same behaviour, so we provide
    one canonical implementation and call it the recency baseline.
    """

    name = "recency"

    def reduce(self, slots: list[Slot], budget: int, now: int) -> None:
        # Oldest arrival evicted first.
        order = sorted(
            (s for s in slots if s.state is not State.EVICTED),
            key=lambda s: s.arrival_turn,
        )
        i = 0
        while total_cost(slots) > budget and i < len(order):
            order[i].state = State.EVICTED
            i += 1


class RandomPolicy(Policy):
    """Evict a uniformly random live item until the budget is met.

    Seeded per policy instance so runs are reproducible.
    """

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def reduce(self, slots: list[Slot], budget: int, now: int) -> None:
        while total_cost(slots) > budget:
            live = [s for s in slots if s.state is not State.EVICTED]
            if not live:
                return
            victim = live[int(self._rng.integers(0, len(live)))]
            victim.state = State.EVICTED


class LRUPolicy(Policy):
    """Evict the least-recently-touched item first (a query 'touches' its item).

    Ties broken by oldest arrival so the policy is fully deterministic.
    """

    name = "lru"

    def reduce(self, slots: list[Slot], budget: int, now: int) -> None:
        order = sorted(
            (s for s in slots if s.state is not State.EVICTED),
            key=lambda s: (s.last_touch_turn, s.arrival_turn),
        )
        i = 0
        while total_cost(slots) > budget and i < len(order):
            order[i].state = State.EVICTED
            i += 1


@dataclass
class _Upgrade:
    """A candidate state change with its marginal value and cost."""

    slot: Slot
    to_state: State
    delta_value: float
    delta_cost: int


class KnapsackPolicy(Policy):
    """Value-density knapsack over observable features, with summarization.

    Each item gets a *utility* estimate from features the policy can actually
    see at decision time: salience (noisy), recency decay, and how often it has
    been queried so far. We then solve the multiple-choice knapsack greedily:
    start from everything evicted and repeatedly apply the upgrade (evict->
    summary, summary->full, or evict->full) with the best value-per-token, while
    the budget allows. Greedy is intentionally sub-optimal, which keeps results
    honest rather than suspiciously perfect.
    """

    name = "knapsack"

    def __init__(
        self,
        w_salience: float = 1.0,
        w_recency: float = 0.6,
        w_access: float = 0.9,
        recency_halflife: float = 12.0,
        summary_value_factor: float = 0.85,
    ) -> None:
        self.w_salience = w_salience
        self.w_recency = w_recency
        self.w_access = w_access
        self.recency_halflife = recency_halflife
        self.summary_value_factor = summary_value_factor

    def utility(self, slot: Slot, now: int) -> float:
        age = max(0, now - slot.arrival_turn)
        recency = 0.5 ** (age / self.recency_halflife)
        access = 1.0 - 0.5 ** slot.access_count  # saturating in [0, 1)
        return (
            self.w_salience * slot.item.salience
            + self.w_recency * recency
            + self.w_access * access
        )

    def reduce(self, slots: list[Slot], budget: int, now: int) -> None:
        live = [s for s in slots if s.state is not State.EVICTED]
        if not live:
            return

        util = {id(s): self.utility(s, now) for s in live}

        # Reset to the empty (all-evicted) allocation, then greedily fill.
        for s in live:
            s.state = State.EVICTED

        spent = 0

        def candidate_upgrades() -> list[_Upgrade]:
            ups: list[_Upgrade] = []
            for s in live:
                u = util[id(s)]
                if s.state is State.EVICTED:
                    if s.item.summarizable:
                        ups.append(
                            _Upgrade(
                                s,
                                State.SUMMARY,
                                u * self.summary_value_factor,
                                s.item.cost_summary,
                            )
                        )
                    # Direct jump to full (used when not summarizable, or when a
                    # single full upgrade beats a summary on density).
                    ups.append(_Upgrade(s, State.FULL, u, s.item.cost_full))
                elif s.state is State.SUMMARY:
                    extra_cost = s.item.cost_full - s.item.cost_summary
                    extra_value = u * (1.0 - self.summary_value_factor)
                    if extra_cost > 0:
                        ups.append(_Upgrade(s, State.FULL, extra_value, extra_cost))
            return ups

        while True:
            ups = [u for u in candidate_upgrades() if spent + u.delta_cost <= budget and u.delta_cost > 0]
            if not ups:
                break
            # Pick the highest value-density upgrade; deterministic tie-break on
            # arrival turn then item id.
            best = max(
                ups,
                key=lambda u: (
                    u.delta_value / u.delta_cost,
                    -u.slot.arrival_turn,
                    -u.slot.item.item_id,
                ),
            )
            best.slot.state = best.to_state
            spent += best.delta_cost


# Registry for the CLI / benchmark. ``random`` and ``knapsack`` are built fresh
# per run so their internal seeds/params are explicit.
def default_policies(random_seed: int = 0) -> list[Policy]:
    return [
        RecencyPolicy(),
        RandomPolicy(seed=random_seed),
        LRUPolicy(),
        KnapsackPolicy(),
    ]


BASELINE_NAMES: tuple[str, ...] = ("recency", "random", "lru")
TREATMENT_NAME = "knapsack"
