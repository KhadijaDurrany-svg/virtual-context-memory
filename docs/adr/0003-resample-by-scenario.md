# ADR 0003: Quantify uncertainty by resampling scenarios

## Status
Accepted

## Context
A single mean recall number is not enough to claim a policy is better. Queries
within one scenario are correlated (they share the same item stream and budget
pressure), so treating individual queries as independent samples would
understate uncertainty and inflate significance.

## Decision
The unit of statistical resampling is the **scenario**. Each scenario yields one
recall value (its own query hit rate). We then:

- report a **percentile bootstrap 95% CI** for each policy's mean recall, and
- compare the knapsack policy to the *strongest* baseline at each budget with a
  **paired permutation test** (sign-flipping the per-scenario differences),
  which needs no distributional assumptions.

All resampling is seeded, so reported intervals and p-values are reproducible.

## Consequences
- Confidence intervals widen appropriately because n is the number of scenarios,
  not the number of queries.
- The permutation test is paired, matching scenarios across policies, which is
  the correct design since both policies see identical workloads.
- The p-value uses the standard +1 correction so it is never exactly zero.
