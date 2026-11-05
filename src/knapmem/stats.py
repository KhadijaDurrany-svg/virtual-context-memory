"""Honest uncertainty: bootstrap confidence intervals and a paired permutation
test. Both are non-parametric and seeded, so quoted numbers are reproducible.

The resampling unit is the *scenario* (each scenario contributes one recall
value), which respects the fact that queries within a scenario are correlated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeanCI:
    mean: float
    lo: float
    hi: float


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> MeanCI:
    """Percentile bootstrap CI for the mean of ``values``."""

    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("values must be non-empty")
    rng = np.random.default_rng(seed)
    n = values.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return MeanCI(mean=float(values.mean()), lo=lo, hi=hi)


@dataclass(frozen=True)
class PermTestResult:
    observed_diff: float
    p_value: float
    n_perm: int


def paired_permutation_test(
    treatment: np.ndarray,
    baseline: np.ndarray,
    n_perm: int = 10_000,
    seed: int = 0,
) -> PermTestResult:
    """Two-sided paired permutation test on the mean difference.

    Pairs are matched by position (same scenario). Under the null, the sign of
    each per-scenario difference is exchangeable, so we flip signs at random and
    build the null distribution of the mean difference.
    """

    treatment = np.asarray(treatment, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if treatment.shape != baseline.shape:
        raise ValueError("treatment and baseline must have equal shape")
    if treatment.size == 0:
        raise ValueError("inputs must be non-empty")

    diff = treatment - baseline
    observed = float(diff.mean())

    rng = np.random.default_rng(seed)
    n = diff.size
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    null_means = (signs * diff).mean(axis=1)
    # +1 correction so p is never exactly zero (conservative, standard).
    count = int(np.sum(np.abs(null_means) >= abs(observed) - 1e-12))
    p_value = (count + 1) / (n_perm + 1)
    return PermTestResult(observed_diff=observed, p_value=p_value, n_perm=n_perm)
