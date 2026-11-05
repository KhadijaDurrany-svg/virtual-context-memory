"""Replay a scenario under a policy and measure recall under budget.

The simulator walks the merged timeline. On an ITEM event the item enters the
working set at FULL detail; if that pushes total cost over the budget, the
policy reduces the working set. On a QUERY event we record whether the queried
item's fact is still present (FULL, or SUMMARY when the item is summarizable),
and we 'touch' the item so access-aware policies can react.

Recall for a scenario is the fraction of its queries whose fact was retained.
"""

from __future__ import annotations

from dataclasses import dataclass

from .data import ITEM, QUERY, Scenario
from .policies import Policy, Slot, State, total_cost


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: int
    recall: float
    n_queries: int
    hits: int
    # Average fraction of the budget actually used across all query moments.
    mean_utilization: float


def run_scenario(scenario: Scenario, policy: Policy, budget: int) -> ScenarioResult:
    if budget <= 0:
        raise ValueError("budget must be positive")

    slots: dict[int, Slot] = {}
    order: list[Slot] = []
    hits = 0
    seen_queries = 0
    util_samples: list[float] = []

    for event in scenario.events:
        if event.kind == ITEM:
            item = event.item
            assert item is not None
            slot = Slot(
                item=item,
                state=State.FULL,
                arrival_turn=event.turn,
                last_touch_turn=event.turn,
            )
            slots[item.item_id] = slot
            order.append(slot)
            if total_cost(order) > budget:
                policy.reduce(order, budget, now=event.turn)
        elif event.kind == QUERY:
            seen_queries += 1
            target = slots.get(event.query_item_id)
            if target is not None and target.retains_fact():
                hits += 1
                # A successful recall counts as an access and refreshes recency.
                target.access_count += 1
                target.last_touch_turn = event.turn
            util_samples.append(total_cost(order) / budget)
        else:  # pragma: no cover - guarded by data model
            raise ValueError(f"unknown event kind: {event.kind}")

    recall = hits / seen_queries if seen_queries else 0.0
    mean_util = sum(util_samples) / len(util_samples) if util_samples else 0.0
    return ScenarioResult(
        scenario_id=scenario.scenario_id,
        recall=recall,
        n_queries=seen_queries,
        hits=hits,
        mean_utilization=mean_util,
    )


def run_policy(scenarios: list[Scenario], policy: Policy, budget: int) -> list[ScenarioResult]:
    return [run_scenario(s, policy, budget) for s in scenarios]
