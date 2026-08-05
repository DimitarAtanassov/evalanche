# ADR 001: No runtime Hugging Face for dataset loading

Status: accepted · Date: 2026-08-05  
Deciders: evalanche architects

## Context

Phase 4 needs real benchmark sources often published via Hugging Face. Adding
`datasets` / `huggingface_hub` to the installable package would pull network,
cache, and license ambiguity onto every `evalctl run` and CI path. The Phase 4
template forbids a runtime HF dependency on `evalharness`. Phase 6 NLI must not
reintroduce HF downloads on the default pytest path.

Dominating constraint: **hermetic, license-auditable evaluation inputs**.

## Decision

1. Do not add Hugging Face libraries to `[project] dependencies` or to any extra
   required by `evalctl run` / suite / judge / default pytest paths.
2. Dataset adapters live under `tools/datasets/` and read **pinned local
   snapshots** (`canonical_url` + `revision_digest`) via
   `evalctl dataset materialize`.
3. Commit only MIT-compatible allow-listed smoke fixtures; rebuild larger tiers
   into cache outside git.
4. RAG faithfulness NLI uses the existing Provider seam or reports
   `NLI_UNAVAILABLE`; no HF model download on default pytest.
5. Optional future `dataset-build` extra, if ever needed, still must not be
   imported by the `evalharness` package; prefer avoiding HF even there.

## Consequences

Easy: CI stays reproducible; license review is per committed byte; loader seam
unchanged. Hard: authors must vendor or cache source snapshots. Give up:
one-command `load_dataset("squad")` convenience inside the harness.

## Alternatives considered

- **HF inside adapters only (build extra):** Loses if any import leaks into
  runtime; still complicates license story. Rejected for Phase 4; revisit only
  with a sealed offline cache and explicit audit.
- **Runtime HF with pinned revision:** Violates the phase constraint and CI
  hermeticity. Rejected.
