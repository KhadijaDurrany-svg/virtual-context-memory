from __future__ import annotations

import pytest

from knapmem.data import DatasetConfig, generate
from knapmem.policies import KnapsackPolicy, RecencyPolicy, default_policies
from knapmem.simulate import run_policy, run_scenario


def test_recall_in_unit_interval():
    scns = generate(DatasetConfig(n_scenarios=8, items_per_scenario=30, queries_per_scenario=20))
    for policy in default_policies():
        for r in run_policy(scns, policy, budget=400):
            assert 0.0 <= r.recall <= 1.0
            assert r.hits <= r.n_queries


def test_generous_budget_gives_perfect_recall():
    # A budget large enough to hold every item at full detail retains all facts.
    cfg = DatasetConfig(n_scenarios=5, items_per_scenario=20, queries_per_scenario=15, cost_max=100)
    scns = generate(cfg)
    huge = 20 * 100 + 10
    for r in run_policy(scns, RecencyPolicy(), budget=huge):
        assert r.recall == 1.0


def test_tiny_budget_hurts_recall():
    cfg = DatasetConfig(n_scenarios=10, items_per_scenario=40, queries_per_scenario=30)
    scns = generate(cfg)
    tight = [r.recall for r in run_policy(scns, RecencyPolicy(), budget=150)]
    loose = [r.recall for r in run_policy(scns, RecencyPolicy(), budget=2000)]
    assert sum(tight) < sum(loose)


def test_knapsack_beats_recency_on_average_tight_budget():
    cfg = DatasetConfig(n_scenarios=60)
    scns = generate(cfg)
    knap = sum(r.recall for r in run_policy(scns, KnapsackPolicy(), budget=512))
    rec = sum(r.recall for r in run_policy(scns, RecencyPolicy(), budget=512))
    assert knap > rec


def test_utilization_never_exceeds_one():
    scns = generate(DatasetConfig(n_scenarios=5, items_per_scenario=30, queries_per_scenario=20))
    for policy in default_policies():
        for r in run_policy(scns, policy, budget=300):
            assert 0.0 <= r.mean_utilization <= 1.0 + 1e-9


def test_zero_budget_raises():
    scn = generate(DatasetConfig(n_scenarios=1))[0]
    with pytest.raises(ValueError):
        run_scenario(scn, RecencyPolicy(), budget=0)


def test_result_fields_consistent():
    scn = generate(DatasetConfig(n_scenarios=1, items_per_scenario=25, queries_per_scenario=18))[0]
    r = run_scenario(scn, KnapsackPolicy(), budget=400)
    assert r.n_queries == scn.n_queries
    assert r.scenario_id == scn.scenario_id
    if r.n_queries:
        assert abs(r.recall - r.hits / r.n_queries) < 1e-12
