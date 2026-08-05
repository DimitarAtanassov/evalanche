# Spec 0001: Phases 4–6 (datasets → suite → judge/RAG)

Status: **accepted** · Tier: **T2** · Date: 2026-08-05  
Owner: evalanche · Links: `docs/private.templates/phase-{4,5,6}.md`, report schema 2.1  
Review: spec-reviewer blocking/major/minor findings resolved in place

## Problem

The harness can validate synthetic fixtures and emit honest single-run reports, but it
cannot yet prove task-fit metrics on licensed real data, summarize multi-run
benchmarks without bloating the per-run report, or add calibrated open-ended / RAG
signals without turning an uncalibrated judge into a silent gate.

## Decision (summary)

Ship in **strict dependency order — Phase 4 → Phase 5 → Phase 6**. No Phase 6
unit starts until Phase 5 **B3** (`suite.html` + CLI build) is green. This
matches the delivery requirement and the phase-6 template ("depends on Phase 5
suite contract").

1. **Phase 4** — Offline, pinned adapters via `evalctl dataset materialize`;
   MIT-compatible SPDX allow-list; NC/DUA/news-fulltext/PhraseBank/CNN/XSum
   cache-only; **no runtime HF**.
2. **Phase 5** — `suite.yaml` → `suite.json`/`suite.html` from report 2.1 +
   compare files only; run reports unchanged.
3. **Phase 6** — Only after Phase 5 B3 is green. Pointwise/pairwise+swap; RAG
   evidence separate; gate bit only from **holdout** `calibration.json` with
   mandatory family separation; judgment files stay informational until attach.

Dominating constraints: license safety (P4), artifact-only suite (P5),
fail-closed holdout calibration (P6).

## What already exists (do not reinvent)

| Surface | Today |
|---------|--------|
| Dataset contract | `datasets/loader.py` + `validator.py` |
| Metrics | `scoring/catalog.py`; `Requirement.JUDGE` reserved |
| Report | Schema **2.1**; PoC under `fixtures/poc/` |
| Compare | `evalctl runs compare` → `schema_version: "1.0"` |
| Store | `judgments`, `annotations` unwired |
| Verify | ruff, mypy `--strict`, pytest, PoC golden, `benchmark_100k` |

`docs/stack.md` is **Absent** (inferred from `pyproject.toml` + CI).

## Artifacts in this spec

| File | Purpose |
|------|---------|
| [design.md](design.md) | Constraints, approach, **Threats**, failure modes, ops |
| [patterns.md](patterns.md) | Selected / deferred / rejected |
| [decomposition.md](decomposition.md) | Independently green units |
| [contracts/dataset-adapter.md](contracts/dataset-adapter.md) | Adapters, SPDX, pins |
| [contracts/suite.md](contracts/suite.md) | Suite 0.1 |
| [contracts/judge.md](contracts/judge.md) | Rubric, BT rule, holdout gate |
| [contracts/rag-evidence.md](contracts/rag-evidence.md) | RAG evidence + NLI policy |
| [adr/001-no-runtime-huggingface.md](adr/001-no-runtime-huggingface.md) | No HF runtime |
| [adr/002-suite-reads-artifacts-only.md](adr/002-suite-reads-artifacts-only.md) | Suite isolation |
| [adr/003-judge-informational-until-calibrated.md](adr/003-judge-informational-until-calibrated.md) | Holdout gate |
| [adr/004-rag-methods-minimal.md](adr/004-rag-methods-minimal.md) | Minimal RAG methods; deferrals |

## How to implement next

1. **Build** → `orchestrator` / `python-dev` from **PR A1**.
2. Phase 5 after ≥2 committed Phase 4 families + A5.
3. Phase 6 **C1–C5** only after Phase 5 **B3** is green (strict sequence, no
   parallel start with Phase 5). **C6** after C2–C5 schemas are stable.
4. Optional finer split → `decomposer`.

## Explicitly deferred

- HF / `datasets` runtime; MIMIC; suite server; Phase 7 blocking gates; judge
  ensembles; `evald`; object storage; RunReport 2.1 changes; meteor as required CI metric
- Answer-grounded (RAGAS-style) context precision/recall and NLI-verified-only
  citation attribution → [ADR-004](adr/004-rag-methods-minimal.md)
