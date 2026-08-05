# ADR 003: Judge output informational until holdout calibration clears

Status: accepted · Date: 2026-08-05  
Deciders: evalanche architects

## Context

LLM judges are biased. Phase 6 must land signals before Phase 7 gates without
letting uncalibrated or **dev-set** agreement clear a gate bit.

Dominating constraint: **fail closed on gating authority; mechanical
dev/holdout separation**.

## Decision

1. `judgment.json` always has `gating_allowed: false` until
   `evalctl judge attach-calibration` copies a passing calibration digest.
2. `evalctl judge validate` (file paths primary) writes **`calibration.json`**
   only; that file is the source of truth for the gate bit.
3. Gate bit is true only when agreement is computed on `split: holdout` labels,
   `n_holdout >= min_holdout_n` (default 150), a distinct dev calibration
   record is handled per rubric without entering the predicate as the score
   source, and **family separation is mandatory**
   (`judge_model_family != candidate_model_family`; empty fields fail closed).
4. Phase 6 CI uses synthetic holdout fixtures, never production `dev` labels.
5. Deterministic metrics are never overwritten by judge scores.
6. Phase 7 must honor `calibration.json` / attached digest.

## Consequences

Easy: safe early ship; no silent gate. Hard: real human holdout required before
production gating. Give up: clearing gates from convenient `dev` agreement.

## Alternatives considered

- **Single `n` without split:** Allows 150 `dev` labels to clear the bit.
  Rejected.
- **Flip judgment in place during validate:** Couples SoT; rejected in favor of
  calibration artifact + optional attach.
