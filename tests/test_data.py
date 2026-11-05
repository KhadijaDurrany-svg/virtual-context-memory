from __future__ import annotations

import pytest

from knapmem.data import (
    ITEM,
    QUERY,
    DatasetConfig,
    checksum,
    generate,
)


def test_generate_count_and_shape():
    cfg = DatasetConfig(n_scenarios=5, items_per_scenario=10, queries_per_scenario=7)
    scns = generate(cfg)
    assert len(scns) == 5
    for s in scns:
        assert s.n_items == 10
        assert s.n_queries == 7
        assert len(s.events) == 10 + 7


def test_generate_is_deterministic():
    cfg = DatasetConfig(n_scenarios=4, items_per_scenario=12, queries_per_scenario=8)
    assert checksum(generate(cfg)) == checksum(generate(cfg))


def test_different_seed_changes_dataset():
    a = generate(DatasetConfig(seed=1, n_scenarios=4))
    b = generate(DatasetConfig(seed=2, n_scenarios=4))
    assert checksum(a) != checksum(b)


def test_queries_fire_after_their_target_arrives():
    scns = generate(DatasetConfig(n_scenarios=10, items_per_scenario=20, queries_per_scenario=15))
    for s in scns:
        arrival = {}
        for e in s.events:
            if e.kind == ITEM:
                assert e.item is not None
                arrival[e.item.item_id] = e.turn
            else:
                assert e.kind == QUERY
                assert e.query_item_id in arrival
                assert e.turn > arrival[e.query_item_id]


def test_item_costs_within_bounds_and_summary_cheaper():
    cfg = DatasetConfig(n_scenarios=6, cost_min=20, cost_max=120)
    for s in generate(cfg):
        for e in s.events:
            if e.kind == ITEM:
                it = e.item
                assert 20 <= it.cost_full <= 120
                assert 1 <= it.cost_summary < it.cost_full or it.cost_full == 1
                assert 0.0 <= it.salience <= 1.0


def test_events_sorted_items_before_queries_same_turn():
    scns = generate(DatasetConfig(n_scenarios=5, items_per_scenario=15, queries_per_scenario=10))
    for s in scns:
        keys = [(e.turn, 0 if e.kind == ITEM else 1) for e in s.events]
        assert keys == sorted(keys)


def test_child_seed_stability_of_first_scenario():
    a = generate(DatasetConfig(seed=11, n_scenarios=1))
    b = generate(DatasetConfig(seed=11, n_scenarios=50))
    assert checksum([a[0]]) == checksum([b[0]])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_scenarios": 0},
        {"items_per_scenario": 1},
        {"queries_per_scenario": 0},
        {"cost_min": 0},
        {"cost_min": 50, "cost_max": 10},
        {"summary_ratio": 0.0},
        {"summary_ratio": 1.0},
        {"summarizable_prob": 1.5},
        {"important_prob": -0.1},
        {"salience_noise": -1.0},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        generate(DatasetConfig(**kwargs))


def test_checksum_is_hex_sha256():
    cs = checksum(generate(DatasetConfig(n_scenarios=2)))
    assert len(cs) == 64
    int(cs, 16)  # parses as hex
