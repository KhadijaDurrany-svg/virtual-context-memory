# ADR 0001: Measure recall under budget, not answer accuracy

## Status
Accepted

## Context
The problem area is managing an LLM's limited working memory: deciding what to
keep, summarize, or evict as content streams in under a fixed token budget. We
want a metric that is *causal to the policy* and does not depend on any model
being available at runtime.

Answer accuracy would require running an LLM, which pulls in network access,
API keys, non-determinism, and cost. It also confounds the retention policy
with the model's reasoning ability.

## Decision
The benchmark measures **recall under budget**: for each later query, was the
fact it needs still present in the working set (at full detail, or in a summary
that preserves the fact)? Recall is the fraction of queries whose fact survived.

This isolates exactly what the policy controls — what stays in the budget — and
is computable offline and deterministically.

## Consequences
- No LLM, no network, no keys. Runs anywhere, identically.
- We explicitly do **not** claim anything about downstream answer quality; a
  retained fact is necessary but not sufficient for a correct answer. This is
  stated in the README "Known gaps".
- The metric rewards policies that keep *the right* items, which is the whole
  point.
