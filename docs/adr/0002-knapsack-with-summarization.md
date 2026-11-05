# ADR 0002: Retention as a multiple-choice knapsack with a summarize tier

## Status
Accepted

## Context
Naive context managers make a binary keep/evict choice, usually by recency
(FIFO). But items are not equally valuable, and an item can often be compressed
to a cheaper summary that still carries its key fact. That turns retention into
a *multiple-choice knapsack*: for each item pick exactly one of {full, summary,
evicted}, each with its own token cost and value, maximizing total value under
the budget.

Optimal multiple-choice knapsack is pseudo-polynomial (DP over the budget). We
decided not to use the exact DP.

## Decision
Use a **greedy value-density** solver: start from everything evicted and
repeatedly apply the highest value-per-token upgrade (evict to summary, summary
to full, or evict straight to full) that still fits. Item value is estimated
from features observable at decision time — salience, recency decay, and prior
access count — never from ground-truth query labels.

## Consequences
- Greedy is O(n^2) per reduction and deterministic, with explicit tie-breaks.
- Greedy is **not optimal**, which we consider a feature: it keeps measured
  recall realistic rather than suspiciously perfect, and it matches what a
  cheap online policy can actually afford to compute per turn.
- The summarize tier is what structurally separates this policy from the
  recency/random/LRU baselines, which only keep or evict.
- Because value uses only noisy observable features, the policy can and does
  make mistakes; this is intended.
