# Lexical & structured metrics

**The cheapest, most reproducible signals in the catalog.** These metrics look at the
*surface* of an output — its exact characters, its token bag, its edit distance, whether a
required phrase is present, whether it parses as JSON with the right fields — and return a
verdict with zero model calls and (mostly) zero randomness. They are your regression
tripwires and hard gates, not your final quality verdict.

All of these live in [`scoring/catalog.py`](../../../src/evalharness/scoring/catalog.py)
(except `exact_match`, which lives in
[`scoring/exact_match.py`](../../../src/evalharness/scoring/exact_match.py) so it can own the
versioned `Normalizer`).

## The metrics

| Doc | Registered name | One line | Requires |
|-----|-----------------|----------|----------|
| [Exact match](exact-match.md) | `exact_match` | Normalized string equality; the default headline pass rate | `REFERENCE` |
| [SQuAD token F1](squad-f1.md) | `squad_f1` | Token‑overlap F1 for paraphrase‑tolerant short answers | `REFERENCE` |
| [Normalized Levenshtein](normalized-levenshtein.md) | `normalized_levenshtein` | Character edit‑distance similarity for near‑copies / OCR noise | `REFERENCE` |
| [Assertions](assertions.md) | `assertions` | All `must_contain` present and all `must_not_contain` absent | none |
| [Numeric assertion](numeric-assertion.md) | `numeric_assertion` | Extract numbers from output & reference, compare within tolerance | `REFERENCE` |
| [JSON validity](json-validity.md) | `json_validity` | Output parses as JSON (and optionally matches a schema) | none |
| [JSON field F1](json-field-f1.md) | `json_field_f1` | Flatten nested JSON to leaf paths; per‑field precision/recall/F1 | none¹ |

¹ `json_field_f1` declares no `Requirement`, but returns `NULL` unless the case supplies
`expected_json`.

## When to reach for this family

- **Short‑form QA with a canonical answer** → `exact_match` as the gate, `squad_f1` for
  partial credit, `normalized_levenshtein` for near‑misses.
- **Content constraints & safety denylists** → `assertions`.
- **Numeric answers embedded in prose** → `numeric_assertion`.
- **Structured extraction / tool arguments** → `json_validity` (did it parse?) plus
  `json_field_f1` (how right were the fields?). Use both — validity tells you *whether*,
  field‑F1 tells you *how much*.

## Shared mechanics you should know

- **`ScalarMetric` base.** Every metric in `catalog.py` subclasses `ScalarMetric`, which
  provides the default `score()`/`aggregate()`: value in `[0,1]`, `passed = value ≥
  threshold` (default `0.5`), aggregate = **mean + Wilson 95% CI** over thresholded
  successes (`method="mean+wilson"`). `exact_match` uses its own Wilson‑only aggregate.
- **Missing data → `NULL`, not `0`.** When a reference or output is absent, these metrics
  return `value=None` with a `detail.reason`. A `NULL` is excluded from the denominator; a
  `0.0` counts as a failure. Confusing the two silently biases your rate — this family is
  careful about it, and so should you be.
- **Config is hashed.** The `Normalizer` config and any thresholds fold into
  `metric_config_sha256`. Change a rule and you get a *new* score row next to the old one,
  never a silent rewrite. See [principle #6](../../principles.md).

## Related

- [Family narrative](../../metrics.md#lexical--deterministic) in `metrics.md`
- [guide §6.1](../../guide.md#61-deterministic--lexical) — the compact reference
- [Semantic similarity](../semantic-similarity/README.md) — when surface overlap is too
  strict and you need meaning
