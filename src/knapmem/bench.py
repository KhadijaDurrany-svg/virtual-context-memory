"""Benchmark orchestration: run every policy at every budget over the generated
dataset, attach bootstrap CIs and a paired permutation test versus the best
baseline, and build a reproducibility manifest.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass, field

import numpy as np

from . import __version__
from .data import DatasetConfig, Scenario, checksum, generate
from .policies import (
    BASELINE_NAMES,
    TREATMENT_NAME,
    default_policies,
)
from .simulate import run_policy
from .stats import bootstrap_mean_ci, paired_permutation_test


@dataclass
class PolicyBudgetResult:
    policy: str
    budget: int
    mean_recall: float
    ci_lo: float
    ci_hi: float
    mean_utilization: float


@dataclass
class ComparisonResult:
    budget: int
    best_baseline: str
    treatment_mean: float
    best_baseline_mean: float
    observed_diff: float
    p_value: float


@dataclass
class BenchResult:
    config: dict
    budgets: list[int]
    dataset_checksum: str
    per_policy: list[PolicyBudgetResult] = field(default_factory=list)
    comparisons: list[ComparisonResult] = field(default_factory=list)


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        sha = out.stdout.strip()
        return sha if out.returncode == 0 and sha else "n/a"
    except (OSError, subprocess.SubprocessError):
        return "n/a"


def build_manifest(result: BenchResult) -> dict:
    return {
        "tool": "knapmem",
        "version": __version__,
        "git_sha": git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "dataset_checksum": result.dataset_checksum,
        "budgets": result.budgets,
        "config": result.config,
    }


def run_benchmark(
    cfg: DatasetConfig,
    budgets: list[int],
    n_boot: int = 10_000,
    n_perm: int = 10_000,
    stat_seed: int = 0,
    scenarios: list[Scenario] | None = None,
) -> BenchResult:
    if not budgets:
        raise ValueError("at least one budget is required")

    scenarios = scenarios if scenarios is not None else generate(cfg)
    result = BenchResult(
        config=_config_to_dict(cfg),
        budgets=list(budgets),
        dataset_checksum=checksum(scenarios),
    )

    for budget in budgets:
        recall_by_policy: dict[str, np.ndarray] = {}
        util_by_policy: dict[str, float] = {}
        for policy in default_policies(random_seed=cfg.seed):
            scenario_results = run_policy(scenarios, policy, budget)
            recalls = np.array([r.recall for r in scenario_results], dtype=float)
            recall_by_policy[policy.name] = recalls
            util_by_policy[policy.name] = float(
                np.mean([r.mean_utilization for r in scenario_results])
            )
            ci = bootstrap_mean_ci(recalls, n_boot=n_boot, seed=stat_seed)
            result.per_policy.append(
                PolicyBudgetResult(
                    policy=policy.name,
                    budget=budget,
                    mean_recall=ci.mean,
                    ci_lo=ci.lo,
                    ci_hi=ci.hi,
                    mean_utilization=util_by_policy[policy.name],
                )
            )

        # Compare the treatment against the strongest baseline at this budget.
        best_baseline = max(BASELINE_NAMES, key=lambda nm: recall_by_policy[nm].mean())
        perm = paired_permutation_test(
            recall_by_policy[TREATMENT_NAME],
            recall_by_policy[best_baseline],
            n_perm=n_perm,
            seed=stat_seed,
        )
        result.comparisons.append(
            ComparisonResult(
                budget=budget,
                best_baseline=best_baseline,
                treatment_mean=float(recall_by_policy[TREATMENT_NAME].mean()),
                best_baseline_mean=float(recall_by_policy[best_baseline].mean()),
                observed_diff=perm.observed_diff,
                p_value=perm.p_value,
            )
        )

    return result


def _config_to_dict(cfg: DatasetConfig) -> dict:
    return {
        "seed": cfg.seed,
        "n_scenarios": cfg.n_scenarios,
        "items_per_scenario": cfg.items_per_scenario,
        "queries_per_scenario": cfg.queries_per_scenario,
        "cost_min": cfg.cost_min,
        "cost_max": cfg.cost_max,
        "summary_ratio": cfg.summary_ratio,
        "summarizable_prob": cfg.summarizable_prob,
        "important_prob": cfg.important_prob,
        "salience_noise": cfg.salience_noise,
        "recency_query_frac": cfg.recency_query_frac,
    }


def result_to_dict(result: BenchResult) -> dict:
    return {
        "config": result.config,
        "budgets": result.budgets,
        "dataset_checksum": result.dataset_checksum,
        "per_policy": [vars(r) for r in result.per_policy],
        "comparisons": [vars(c) for c in result.comparisons],
    }
