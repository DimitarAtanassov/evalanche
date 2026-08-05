# Patterns and tradeoffs: Phases 4–6

Tier: **T2** (new bounded contexts inside the existing monolith).  
Dominating constraint: license-safe reproducible inputs and honest artifact
boundaries without new runtime dependencies or silent judge gates.

## Selected

- **Offline adapter / transform scripts (ports at the filesystem boundary):**
  External corpora are awkward and license-sensitive. Emitting the existing
  dataset bundle beats in-process HF loaders because CI stays hermetic and the
  loader seam stays single.
- **Content addressing + pinned provenance:** `canonical_url` +
  `revision_digest` + seed beats floating tags and tampered snapshots.
- **Typed manifest expand:** Additive fields on `DatasetManifest` when
  `schema_version` is set; unknown keys error. Beats untyped passthrough bags.
- **Pure derived view (suite):** Multi-run visualization over immutable JSON
  files beats querying Postgres. Keeps report 2.1 frozen.
- **Expand-then-contract versioning:** Allow-listed `schema_version` values;
  incompatible changes bump version. Beats "0.x means anything goes."
- **Calibration artifact as gate SoT:** `calibration.json` from file-based
  validate; judgment stays false until attach. Beats flipping judgment in place.
- **Holdout-only gate predicate:** Dev agreement recorded, never clears the bit.
  Beats a single unlabeled `n`.
- **Mandatory family separation for true gate bits:** Beats optional
  `forbidden_*` allow-lists that default empty.
- **Provider-seam NLI or honest unavailable:** Beats HF model downloads on
  default pytest.
- **Separated RAG evidence artifact:** Retrieval vs faithfulness vs citation.
- **Published text bounds:** Truncation/redaction aligned with Phase 3 report
  and observability sanitizers.

## Deferred

- **Plugin entry points for dataset adapters:** Promote when a second team ships
  adapters out-of-repo.
- **CQRS / separate read store for suites:** Promote when file suites cannot
  index thousands of runs.
- **Message queue for judge jobs:** Promote when fan-out exceeds process bounds.
- **Judge ensembles:** Promote when release gates need inter-judge disagreement
  control.
- **Suite live server / auth:** Phase 8 adjacency.
- **Blocking gates (`gates.yaml`):** Phase 7 after real holdout calibration.
- **Object storage for evidence blobs:** Per `DEFERRED.md` triggers.
- **Runtime HF / `datasets`:** Never as run dependency; prefer pinned archives.
- **MIMIC / gated clinical:** Separate DUA design.
- **Changing RunReport 2.1:** Only with a new report schema version.
- **meteor as required CI metric:** Optional when NLTK resources Present;
  promote when offline pin is committed in an enabling PR.
- **DB-backed judge validate:** Optional after file path is green; promote when
  operators need annotation table joins in CI.

## Rejected

- **Microservices for suite or judge:** No ownership/scale constraint.
- **Embedding suite panels into run HTML:** Violates Phase 5 goal.
- **Single-implementation Protocol for suite backends:** Files only; concrete
  functions.
- **Treating judge score as default metric:** Invites uncalibrated gating.
- **Hugging Face as runtime dataset or default NLI loader:** Violates Phase 4/6
  hermeticity.
- **Clearing `gating_allowed` from `dev` labels:** Violates mechanical separation.
- **Committing NC / PhraseBank / CNN-DM / XSum fulltext into MIT fixtures:**
  Redistribution trap.
