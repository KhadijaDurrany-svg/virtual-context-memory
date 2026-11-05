from __future__ import annotations

from knapmem.data import Item
from knapmem.policies import (
    KnapsackPolicy,
    LRUPolicy,
    RandomPolicy,
    RecencyPolicy,
    Slot,
    State,
    default_policies,
    total_cost,
)


def make_slot(item_id: int, cost_full: int, turn: int, summarizable: bool = True, salience: float = 0.5) -> Slot:
    item = Item(
        item_id=item_id,
        cost_full=cost_full,
        cost_summary=max(1, cost_full // 3),
        summarizable=summarizable,
        salience=salience,
    )
    return Slot(item=item, state=State.FULL, arrival_turn=turn, last_touch_turn=turn)


def test_default_policies_names():
    names = [p.name for p in default_policies()]
    assert names == ["recency", "random", "lru", "knapsack"]


def test_recency_evicts_oldest_first():
    slots = [make_slot(i, 50, turn=i) for i in range(5)]  # total 250
    RecencyPolicy().reduce(slots, budget=120, now=4)
    assert total_cost(slots) <= 120
    # Oldest items evicted, newest kept.
    assert slots[0].state is State.EVICTED
    assert slots[4].state is State.FULL


def test_lru_evicts_least_recently_touched():
    slots = [make_slot(i, 50, turn=i) for i in range(4)]  # total 200
    # Touch item 0 most recently.
    slots[0].last_touch_turn = 100
    LRUPolicy().reduce(slots, budget=120, now=100)
    assert total_cost(slots) <= 120
    assert slots[0].state is State.FULL  # protected by recent touch
    assert slots[1].state is State.EVICTED  # least recently touched


def test_random_is_seeded_and_reproducible():
    a = [make_slot(i, 50, turn=i) for i in range(6)]
    b = [make_slot(i, 50, turn=i) for i in range(6)]
    RandomPolicy(seed=3).reduce(a, budget=150, now=5)
    RandomPolicy(seed=3).reduce(b, budget=150, now=5)
    assert [s.state for s in a] == [s.state for s in b]
    assert total_cost(a) <= 150


def test_knapsack_respects_budget():
    slots = [make_slot(i, 60, turn=i, salience=0.1 * i) for i in range(8)]
    KnapsackPolicy().reduce(slots, budget=200, now=7)
    assert total_cost(slots) <= 200


def test_knapsack_prefers_high_salience():
    slots = [
        make_slot(0, 60, turn=0, salience=0.95),
        make_slot(1, 60, turn=0, salience=0.02),
    ]
    # Budget fits only one full item.
    KnapsackPolicy(w_recency=0.0, w_access=0.0).reduce(slots, budget=60, now=0)
    assert slots[0].state is State.FULL
    assert slots[1].state is State.EVICTED


def test_knapsack_uses_summary_tier_when_it_helps():
    # Two items, budget fits both only if at least one is summarized.
    slots = [
        make_slot(0, 60, turn=1, salience=0.8),
        make_slot(1, 60, turn=0, salience=0.8),
    ]
    KnapsackPolicy().reduce(slots, budget=80, now=1)
    assert total_cost(slots) <= 80
    states = {s.state for s in slots}
    assert State.SUMMARY in states  # summarization was exercised


def test_non_summarizable_item_never_summarized():
    slots = [make_slot(0, 90, turn=0, summarizable=False, salience=0.9)]
    KnapsackPolicy().reduce(slots, budget=40, now=0)
    # Cannot summarize; with budget below full cost it must be evicted.
    assert slots[0].state is State.EVICTED


def test_summary_retains_fact_only_when_summarizable():
    s_ok = make_slot(0, 60, turn=0, summarizable=True)
    s_ok.state = State.SUMMARY
    assert s_ok.retains_fact() is True

    s_no = make_slot(1, 60, turn=0, summarizable=False)
    s_no.state = State.SUMMARY
    assert s_no.retains_fact() is False


def test_evicted_slot_has_zero_cost_and_no_fact():
    s = make_slot(0, 70, turn=0)
    s.state = State.EVICTED
    assert s.cost() == 0
    assert s.retains_fact() is False


def test_reduce_noop_when_under_budget():
    slots = [make_slot(i, 10, turn=i) for i in range(3)]
    before = [s.state for s in slots]
    RecencyPolicy().reduce(slots, budget=1000, now=2)
    assert [s.state for s in slots] == before


def test_knapsack_utility_monotone_in_access():
    pol = KnapsackPolicy()
    s_lo = make_slot(0, 50, turn=0)
    s_hi = make_slot(0, 50, turn=0)
    s_hi.access_count = 5
    assert pol.utility(s_hi, now=0) > pol.utility(s_lo, now=0)


def test_all_evicted_reduce_is_safe():
    slots = [make_slot(i, 50, turn=i) for i in range(3)]
    for s in slots:
        s.state = State.EVICTED
    # No live slots; should not raise and stays within (zero) cost.
    LRUPolicy().reduce(slots, budget=10, now=5)
    assert total_cost(slots) == 0
