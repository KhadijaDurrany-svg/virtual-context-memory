"""Command-line interface.

Subcommands:
  bench   Run the full benchmark; writes results.json + manifest.json unless
          ``--no-write`` is given.
  demo    Run a single small scenario under every policy and print recall.
  report  Pretty-print tables from an existing results.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .bench import build_manifest, result_to_dict, run_benchmark
from .data import DatasetConfig, generate
from .policies import default_policies
from .simulate import run_scenario

DEFAULT_BUDGETS = [256, 512, 1024]


def _add_dataset_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed", type=int, default=7, help="dataset seed")
    p.add_argument("--scenarios", type=int, default=200, help="number of scenarios")
    p.add_argument("--items", type=int, default=60, help="items per scenario")
    p.add_argument("--queries", type=int, default=40, help="queries per scenario")


def _cfg_from_args(args: argparse.Namespace) -> DatasetConfig:
    return DatasetConfig(
        seed=args.seed,
        n_scenarios=args.scenarios,
        items_per_scenario=args.items,
        queries_per_scenario=args.queries,
    )


def _cmd_bench(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    budgets = args.budgets or DEFAULT_BUDGETS
    result = run_benchmark(
        cfg,
        budgets,
        n_boot=args.n_boot,
        n_perm=args.n_perm,
        stat_seed=args.stat_seed,
    )

    _print_bench_tables(result)

    if args.no_write:
        print("\n[--no-write] results not persisted.")
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    manifest_path = out_dir / "manifest.json"
    results_path.write_text(json.dumps(result_to_dict(result), indent=2) + "\n")
    manifest_path.write_text(json.dumps(build_manifest(result), indent=2) + "\n")
    print(f"\nwrote {results_path}")
    print(f"wrote {manifest_path}")
    return 0


def _print_bench_tables(result) -> None:  # noqa: ANN001 - internal pretty-printer
    by_budget: dict[int, list] = {}
    for row in result.per_policy:
        by_budget.setdefault(row.budget, []).append(row)

    print("Recall under budget (mean [95% bootstrap CI], % of needed facts retained)")
    print(f"dataset checksum: {result.dataset_checksum[:16]}...")
    for budget in result.budgets:
        print(f"\n  budget = {budget} tokens")
        print(f"    {'policy':<10} {'recall':>8} {'95% CI':>20} {'budget used':>12}")
        for row in sorted(by_budget[budget], key=lambda r: r.mean_recall, reverse=True):
            ci = f"[{row.ci_lo:.3f}, {row.ci_hi:.3f}]"
            print(
                f"    {row.policy:<10} {row.mean_recall:>8.3f} {ci:>20} "
                f"{row.mean_utilization:>11.1%}"
            )

    print("\nTreatment (knapsack) vs best baseline, paired permutation test")
    print(f"    {'budget':>7} {'baseline':>10} {'knap':>7} {'base':>7} {'diff':>7} {'p':>9}")
    for cmp in result.comparisons:
        print(
            f"    {cmp.budget:>7} {cmp.best_baseline:>10} "
            f"{cmp.treatment_mean:>7.3f} {cmp.best_baseline_mean:>7.3f} "
            f"{cmp.observed_diff:>+7.3f} {cmp.p_value:>9.4f}"
        )


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = DatasetConfig(seed=args.seed, n_scenarios=1, items_per_scenario=args.items, queries_per_scenario=args.queries)
    scenario = generate(cfg)[0]
    budget = args.budget
    print(f"demo scenario: {scenario.n_items} items, {scenario.n_queries} queries, budget {budget}")
    for policy in default_policies(random_seed=cfg.seed):
        res = run_scenario(scenario, policy, budget)
        print(f"  {policy.name:<10} recall={res.recall:.3f} ({res.hits}/{res.n_queries})")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    path = Path(args.results)
    if not path.exists():
        print(f"no such file: {path}", file=sys.stderr)
        return 2
    data = json.loads(path.read_text())
    print(f"dataset checksum: {data['dataset_checksum']}")
    print(f"budgets: {data['budgets']}")
    print(f"{'policy':<10} {'budget':>7} {'recall':>8} {'ci_lo':>8} {'ci_hi':>8}")
    for row in data["per_policy"]:
        print(
            f"{row['policy']:<10} {row['budget']:>7} {row['mean_recall']:>8.3f} "
            f"{row['ci_lo']:>8.3f} {row['ci_hi']:>8.3f}"
        )
    print("\ncomparisons (knapsack vs best baseline):")
    for cmp in data["comparisons"]:
        print(
            f"  budget={cmp['budget']:>5} baseline={cmp['best_baseline']:<8} "
            f"diff={cmp['observed_diff']:+.3f} p={cmp['p_value']:.4f}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="knapmem",
        description="Benchmarked deterministic retention policy for LLM working memory.",
    )
    parser.add_argument("--version", action="version", version=f"knapmem {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("bench", help="run the full benchmark")
    _add_dataset_args(b)
    b.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=None,
        help=f"token budgets to evaluate (default {DEFAULT_BUDGETS})",
    )
    b.add_argument("--n-boot", type=int, default=10_000, help="bootstrap resamples")
    b.add_argument("--n-perm", type=int, default=10_000, help="permutations")
    b.add_argument("--stat-seed", type=int, default=0, help="seed for stats")
    b.add_argument("--out-dir", default="benchmarks", help="output directory")
    b.add_argument("--no-write", action="store_true", help="do not persist outputs")
    b.set_defaults(func=_cmd_bench)

    d = sub.add_parser("demo", help="run one scenario under every policy")
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--items", type=int, default=60)
    d.add_argument("--queries", type=int, default=40)
    d.add_argument("--budget", type=int, default=512)
    d.set_defaults(func=_cmd_demo)

    r = sub.add_parser("report", help="print tables from a results.json")
    r.add_argument("results", help="path to results.json")
    r.set_defaults(func=_cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
