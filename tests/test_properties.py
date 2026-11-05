from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from knapmem.data import Item
from knapmem.policies import (
    KnapsackPolicy,
    LRUPolicy,
    RandomPolicy,
    RecencyPolicy,
    Slot,
    State,
    total_cost,
)
from knapmem.stats import bootstrap_mean_ci, paired_permutation_test

SLOW = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _slots(costs: list[int], summarizable: list[bool], saliences: list[float]) -> list[Slot]:
    out = []
    for i, (c, can_sum, sal) in enumerate(zip(costs, summarizable, saliences, strict=True)):
        item = Item(
            item_id=i,
            cost_full=c,
            cost_summary=max(1, c // 3),
            summarizable=can_sum,
            salience=sal,
        )
        out.append(Slot(item=item, state=State.FULL, arrival_turn=i, last_touch_turn=i))
    return out


item_lists = st.lists(
    st.tuples(
        st.integers(min_value=1, max_value=200),
        st.booleans(),
        st.floats(min_value=0.0, max_value=1.0),
    ),
    min_size=1,
    max_size=25,
)


@SLOW
@given(items=item_lists, budget=st.integers(min_value=1, max_value=400))
def test_every_policy_respects_budget(items, budget):
    costs = [c for c, _, _ in items]
    summ = [s for _, s, _ in items]
    sals = [a for _, _, a in items]
    for policy in (RecencyPolicy(), RandomPolicy(seed=0), LRUPolicy(), KnapsackPolicy()):
        slots = _slots(costs, summ, sals)
        if total_cost(slots) > budget:
            policy.reduce(slots, budget, now=len(costs))
        assert total_cost(slots) <= budget


@SLOW
@given(items=item_lists, budget=st.integers(min_value=1, max_value=400))
def test_states_are_valid_and_evicted_costs_zero(items, budget):
    costs = [c for c, _, _ in items]
    summ = [s for _, s, _ in items]
    sals = [a for _, _, a in items]
    slots = _slots(costs, summ, sals)
    KnapsackPolicy().reduce(slots, budget, now=len(costs))
    for s in slots:
        assert s.state in (State.FULL, State.SUMMARY, State.EVICTED)
        if s.state is State.EVICTED:
            assert s.cost() == 0
        if s.state is State.SUMMARY:
            # Summary only chosen for summarizable items (others gain nothing).
            assert s.item.summarizable


@SLOW
@given(items=item_lists)
def test_knapsack_fills_budget_when_everything_fits(items):
    costs = [c for c, _, _ in items]
    summ = [s for _, s, _ in items]
    sals = [a for _, _, a in items]
    slots = _slots(costs, summ, sals)
    budget = sum(costs)  # exactly enough for all-full
    KnapsackPolicy().reduce(slots, budget, now=len(costs))
    # With room for everything at full detail, nothing should be dropped.
    assert all(s.state is State.FULL for s in slots)


@SLOW
@given(
    vals=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=2, max_size=100),
)
def test_bootstrap_ci_brackets_sample_mean(vals):
    arr = np.array(vals)
    ci = bootstrap_mean_ci(arr, n_boot=500, seed=0)
    assert abs(ci.mean - arr.mean()) < 1e-9
    # Percentile CI lies within the data range.
    assert arr.min() - 1e-9 <= ci.lo <= ci.hi <= arr.max() + 1e-9


@SLOW
@given(
    diffs=st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=2, max_size=80),
)
def test_permutation_pvalue_in_unit_interval(diffs):
    base = np.zeros(len(diffs))
    treat = np.array(diffs)
    res = paired_permutation_test(treat, base, n_perm=400, seed=0)
    assert 0.0 < res.p_value <= 1.0


@SLOW
@given(
    a=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=2, max_size=60),
)
def test_identical_inputs_have_zero_diff(a):
    arr = np.array(a)
    res = paired_permutation_test(arr, arr, n_perm=200, seed=1)
    assert res.observed_diff == 0.0
