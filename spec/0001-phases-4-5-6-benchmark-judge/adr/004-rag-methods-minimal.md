# ADR 004: RAG evidence ships minimal deterministic methods in Phase 6

Status: accepted · Date: 2026-08-05  
Deciders: evalanche architects

## Context

The Phase 6 RAG evidence artifact must be shippable with deterministic local
doubles and no Hugging Face runtime (ADR-001). Full RAGAS-style metrics
(answer-grounded context precision/recall, NLI-verified citation support,
model-based claim decomposition) require methodology choices and evaluation that
are not yet settled. Leaving those fields as unexplained `null` in the artifact
was the review finding.

Dominating constraint: **every Done-when must be meetable in CI with
deterministic doubles**, and no artifact field may be an unexplained null.

## Decision

1. Phase 6 ships these deterministic methods, each versioned:
   - `claim_split_v1`: lexical sentence segmentation of the answer (or explicit
     `claims`), no model.
   - `claim_nli_v1`: faithfulness through the Provider seam only; `unavailable`
     (`NLI_UNAVAILABLE`) when no NLI provider is configured.
   - `qrels_context_v1`: context precision/recall from `retrieved_contexts` +
     `qrels`, no model.
   - `cited_present_relevant_v1`: citation attribution as cited-present-and
     -relevant (NLI-verified when a provider is available).
2. The following are **deferred** and reported with `status: "deferred"`,
   `reason: "ADR_004_RAG_METHODS"`, never bare null:
   - answer-grounded context precision/recall (RAGAS-style, ground-truth
     grounded).
   - NLI-verified-only citation attribution (requires entailment, not merely
     qrels relevance).
   - model-based or coreference-aware claim decomposition.
3. Promoting a deferred method bumps its method id and the artifact
   `schema_version`, and updates this ADR and the RAG contract.

## Consequences

Easy: Phase 6 exit is concrete and CI-verifiable with mock doubles; no null
fields without an explanation. Hard: the shipped context/citation numbers are
retrieval-context signals, weaker than answer-grounded RAGAS metrics. Give up:
claiming RAGAS parity in Phase 6.

## Alternatives considered

- **Leave the fields null until designed:** Rejected; unexplained nulls hide
  scope and fail the Done-when meetability bar.
- **Implement full RAGAS now:** Rejected for Phase 6; needs research and would
  tempt HF or extra model dependencies against ADR-001.
