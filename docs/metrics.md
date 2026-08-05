# Metric methodology

Every score records the metric semantic version, canonical configuration hash,
normalizer or model revision, and sufficient detail for corpus aggregation.
Generation rows are immutable and never encode scoring outcomes.

The core catalog includes configurable exact match, SQuAD token F1, normalized
Levenshtein, content and numeric assertions, JSON/schema/field metrics,
classification summaries, adaptive calibration, retrieval at fixed cutoffs,
ROUGE, SacreBLEU, chrF++, and semantic similarity through normalized pinned
embeddings. BERTScore is isolated in the `metrics-ml` extra.

Rate intervals use Wilson 95%. Mean intervals use seeded 10,000-resample BCa
bootstrap. Paired binary comparisons use exact McNemar and report absolute and
relative delta plus Cohen's h. Benjamini–Hochberg controls false discovery at
q=0.05 across reported metrics and slices. Cases that vary across repeats are
reported as flaky and excluded from claims, never from raw storage.

Threshold calibration is development-only:

```bash
uv run evalctl calibrate dev-similarities.jsonl
```

The output records ROC-AUC, PR-AUC, the selected operating threshold, and
development F1. Never select thresholds on holdout data.
