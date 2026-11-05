from __future__ import annotations

import numpy as np
import pytest

from knapmem.stats import bootstrap_mean_ci, paired_permutation_test


def test_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(0)
    vals = rng.normal(0.5, 0.1, size=200)
    ci = bootstrap_mean_ci(vals, n_boot=2000, seed=1)
    assert ci.lo <= ci.mean <= ci.hi
    assert abs(ci.mean - vals.mean()) < 1e-9


def test_bootstrap_ci_reproducible():
    vals = np.linspace(0, 1, 50)
    a = bootstrap_mean_ci(vals, n_boot=1000, seed=4)
    b = bootstrap_mean_ci(vals, n_boot=1000, seed=4)
    assert (a.lo, a.hi, a.mean) == (b.lo, b.hi, b.mean)


def test_bootstrap_ci_narrows_with_more_data():
    rng = np.random.default_rng(2)
    small = bootstrap_mean_ci(rng.normal(0, 1, 30), n_boot=2000, seed=0)
    large = bootstrap_mean_ci(rng.normal(0, 1, 3000), n_boot=2000, seed=0)
    assert (large.hi - large.lo) < (small.hi - small.lo)


def test_bootstrap_empty_raises():
    with pytest.raises(ValueError):
        bootstrap_mean_ci(np.array([]), n_boot=10)


def test_permutation_detects_real_difference():
    rng = np.random.default_rng(5)
    base = rng.normal(0.5, 0.05, size=300)
    treat = base + 0.1  # clearly larger, paired
    res = paired_permutation_test(treat, base, n_perm=2000, seed=0)
    assert res.observed_diff > 0.09
    assert res.p_value < 0.01


def test_permutation_no_difference_is_not_significant():
    rng = np.random.default_rng(6)
    base = rng.normal(0.5, 0.1, size=300)
    treat = base.copy()  # identical
    res = paired_permutation_test(treat, base, n_perm=2000, seed=0)
    assert abs(res.observed_diff) < 1e-12
    assert res.p_value > 0.05


def test_permutation_p_never_zero():
    treat = np.ones(50)
    base = np.zeros(50)
    res = paired_permutation_test(treat, base, n_perm=100, seed=0)
    assert res.p_value >= 1 / (100 + 1)


def test_permutation_shape_mismatch_raises():
    with pytest.raises(ValueError):
        paired_permutation_test(np.ones(5), np.ones(4), n_perm=10)


def test_permutation_reproducible():
    rng = np.random.default_rng(7)
    base = rng.normal(0, 1, 100)
    treat = base + 0.05
    a = paired_permutation_test(treat, base, n_perm=1000, seed=3)
    b = paired_permutation_test(treat, base, n_perm=1000, seed=3)
    assert a.p_value == b.p_value
