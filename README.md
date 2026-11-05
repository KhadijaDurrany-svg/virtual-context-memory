# knapmem

**A benchmarked, deterministic retention policy for LLM working memory under a fixed token budget.**

When an LLM agent runs long, content arrives faster than the context window can
hold it. Something has to go. `knapmem` treats that choice as a *multiple-choice
knapsack*: for every item, pick exactly one of **keep full**, **summarize**
(cheaper, lossy), or **evict**, so the retained set fits a fixed token budget
while keeping the items most likely to be needed later. It then **measures** how
well each policy does, by replaying streams of items and later queries and
checking whether the fact a query needs was still present.

Everything is **offline, deterministic, and reproducible**: seeded synthetic
workloads, no network, no API keys, numpy as the only runtime dependency.

## Results

Measured by running `knapmem bench` on this machine (200 scenarios, 60 items and
40 queries each, seed 7). **Recall** = fraction of later queries whose needed
fact was still retained (at full detail, or in a summary that preserves it).
CIs are percentile bootstrap (10,000 resamples) over scenarios.

| budget | knapsack | recency (FIFO) | lru | random |
|-------:|---------:|---------------:|----:|-------:|
| 256  | **0.399** `[0.387, 0.411]` | 0.183 `[0.174, 0.192]` | 0.182 `[0.174, 0.191]` | 0.161 `[0.152, 0.170]` |
| 512  | **0.584** `[0.571, 0.596]` | 0.370 `[0.358, 0.381]` | 0.368 `[0.357, 0.380]` | 0.309 `[0.297, 0.321]` |
| 1024 | **0.794** `[0.784, 0.805]` | 0.603 `[0.591, 0.615]` | 0.601 `[0.589, 0.613]` | 0.489 `[0.477, 0.501]` |

Knapsack vs the **strongest** baseline at each budget (paired permutation test,
10,000 permutations):

| budget | best baseline | knapsack | baseline | difference | p-value |
|-------:|:--------------|---------:|---------:|-----------:|--------:|
| 256  | recency | 0.399 | 0.183 | **+0.216** | 0.0001 |
| 512  | recency | 0.584 | 0.370 | **+0.214** | 0.0001 |
| 1024 | recency | 0.794 | 0.603 | **+0.191** | 0.0001 |

The gain is large and significant, but recall is far from perfect (0.40–0.79),
which is expected: the budgets are tight and the policy scores items from a
*noisy* salience signal, so it cannot behave like an oracle. The p-value floor
is `1 / (n_perm + 1)` ≈ 0.0001 by construction; it means "no permutation
matched the observed gap," not a literally-zero probability.

Reproduce exactly:

```bash
knapmem bench          # writes benchmarks/results.json + manifest.json
knapmem report benchmarks/results.json
```

The manifest records the git SHA, Python/numpy versions, the full config, and a
SHA-256 checksum of the generated dataset, so any result can be tied to the
exact inputs that produced it.

## How this differs from a naive approach

A naive context manager keeps the most recent turns and truncates the rest
(FIFO / recency window). That is the `recency` baseline above, and it throws
away an old-but-important fact the moment a few newer turns arrive. `knapmem`
differs in two structural ways:

1. **It scores value, not just age.** Each item's keep-value combines a
   salience signal, a recency decay, and how often it has already been queried;
   the knapsack then spends the budget on the highest value-per-token items.
2. **It has a middle gear: summarize.** An item can be compressed to a cheaper
   form that still carries its key fact, so the policy can retain *more* facts
   per token instead of making a hard keep-or-drop choice. The baselines only
   keep or evict.

Those two changes are the whole source of the measured gap — nothing else about
the workload differs between policies.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"      # dev extras: pytest, hypothesis, ruff
```

Runtime needs only `numpy` (Python >= 3.10).

## Use

```bash
knapmem bench                        # full benchmark, writes benchmarks/*.json
knapmem bench --no-write             # print tables only
knapmem bench --budgets 128 256 --scenarios 500
knapmem demo --budget 400            # one scenario under every policy
knapmem report benchmarks/results.json
```

As a library:

```python
from knapmem.data import DatasetConfig, generate
from knapmem.policies import KnapsackPolicy
from knapmem.simulate import run_policy

scenarios = generate(DatasetConfig(n_scenarios=50))
results = run_policy(scenarios, KnapsackPolicy(), budget=512)
print(sum(r.recall for r in results) / len(results))
```

## Project layout

```
src/knapmem/
  data.py       seeded synthetic workloads (items + queries) and dataset checksum
  policies.py   retention policies: recency/FIFO, random, LRU, and the knapsack
  simulate.py   replay a scenario under a policy, measure recall under budget
  stats.py      bootstrap CIs and a paired permutation test
  bench.py      orchestrate the benchmark and build the reproducibility manifest
  cli.py        argparse CLI: bench / demo / report
tests/          pytest + Hypothesis property tests
docs/adr/       architecture decision records
```

## How it works

Each scenario is a merged timeline of **item arrivals** and later **queries**.
An item carries one fact and has a full token cost and a cheaper summary cost;
a summary preserves the fact only when the item is `summarizable`. Whether an
item is ever queried is correlated with its observable salience, but with
deliberate noise, so no policy can be perfect. On each arrival the item enters
the working set; if the budget is exceeded, the policy reduces the set. On each
query we record whether the needed fact is still retained. See
`docs/adr/` for the three key design decisions (recall-under-budget as the
metric, knapsack-with-summarize as the policy, scenario-level resampling for
statistics).

## Known gaps

- **Synthetic data is cleaner than reality.** Real workloads have correlated
  turns, topic drift, duplicated facts, and far messier salience signals than
  our Gaussian-noised feature. Absolute recall numbers would differ on real
  logs; treat the *ordering* of policies as the finding, not the magnitudes.
- **Recall is necessary, not sufficient.** We measure whether a needed fact
  stayed in the budget, not whether a model would then answer correctly. No LLM
  is run (by design — that keeps the benchmark offline and deterministic).
- **Token costs are integers, not a specific tokenizer.** Budgets and item
  costs are abstract token units, not the output of any particular model's
  tokenizer.
- **Summarization is modeled, not performed.** A "summary" is a cheaper cost
  plus a fact-preservation flag; we do not actually run a summarizer, so we do
  not capture summary-quality degradation beyond the single fact.
- **The knapsack solver is greedy, not optimal.** It maximizes value-density
  locally; an exact DP would do slightly better. This is intentional: it keeps
  the policy cheap per turn and the results honest.
- **Once evicted, an item stays evicted.** There is no retrieval-back / page-in
  mechanism; this is a pure in-budget retention study, not a paging system.

## License

MIT, Copyright (c) 2026 Khadija. See [LICENSE](LICENSE).
