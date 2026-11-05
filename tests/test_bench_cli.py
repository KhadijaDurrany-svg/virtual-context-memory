from __future__ import annotations

import json

import pytest

from knapmem.bench import build_manifest, git_sha, result_to_dict, run_benchmark
from knapmem.cli import main
from knapmem.data import DatasetConfig


def small_cfg() -> DatasetConfig:
    return DatasetConfig(n_scenarios=25, items_per_scenario=30, queries_per_scenario=20)


def test_run_benchmark_structure():
    res = run_benchmark(small_cfg(), budgets=[256, 512], n_boot=300, n_perm=300)
    assert res.budgets == [256, 512]
    assert len(res.per_policy) == 2 * 4  # 2 budgets x 4 policies
    assert len(res.comparisons) == 2
    for c in res.comparisons:
        assert c.best_baseline in {"recency", "random", "lru"}
        assert 0.0 < c.p_value <= 1.0


def test_benchmark_requires_a_budget():
    with pytest.raises(ValueError):
        run_benchmark(small_cfg(), budgets=[])


def test_ci_bounds_bracket_mean():
    res = run_benchmark(small_cfg(), budgets=[400], n_boot=500, n_perm=500)
    for row in res.per_policy:
        assert row.ci_lo <= row.mean_recall <= row.ci_hi


def test_manifest_fields_present():
    res = run_benchmark(small_cfg(), budgets=[256], n_boot=200, n_perm=200)
    man = build_manifest(res)
    for key in ("tool", "version", "git_sha", "python", "dataset_checksum", "budgets", "config"):
        assert key in man
    assert man["tool"] == "knapmem"


def test_git_sha_returns_string():
    sha = git_sha()
    assert isinstance(sha, str)
    assert sha  # non-empty ("n/a" when unavailable)


def test_result_to_dict_roundtrips_json():
    res = run_benchmark(small_cfg(), budgets=[256], n_boot=200, n_perm=200)
    blob = json.dumps(result_to_dict(res))
    back = json.loads(blob)
    assert back["budgets"] == [256]
    assert "per_policy" in back


def test_benchmark_is_deterministic():
    a = run_benchmark(small_cfg(), budgets=[300], n_boot=400, n_perm=400)
    b = run_benchmark(small_cfg(), budgets=[300], n_boot=400, n_perm=400)
    assert result_to_dict(a) == result_to_dict(b)


def test_cli_bench_no_write(capsys):
    rc = main(
        ["bench", "--scenarios", "20", "--budgets", "256", "--n-boot", "200", "--n-perm", "200", "--no-write"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Recall under budget" in out
    assert "knapsack" in out


def test_cli_bench_writes_files(tmp_path, capsys):
    out_dir = tmp_path / "bench_out"
    rc = main(
        [
            "bench",
            "--scenarios",
            "20",
            "--budgets",
            "256",
            "--n-boot",
            "200",
            "--n-perm",
            "200",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    results = json.loads((out_dir / "results.json").read_text())
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert results["budgets"] == [256]
    assert manifest["dataset_checksum"] == results["dataset_checksum"]


def test_cli_demo(capsys):
    rc = main(["demo", "--items", "25", "--queries", "15", "--budget", "300"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "recall=" in out


def test_cli_report_roundtrip(tmp_path, capsys):
    out_dir = tmp_path / "bench_out"
    main(
        [
            "bench",
            "--scenarios",
            "15",
            "--budgets",
            "256",
            "--n-boot",
            "150",
            "--n-perm",
            "150",
            "--out-dir",
            str(out_dir),
        ]
    )
    rc = main(["report", str(out_dir / "results.json")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dataset checksum" in out


def test_cli_report_missing_file():
    rc = main(["report", "/nonexistent/results.json"])
    assert rc == 2
