# `json_field_f1`

## TL;DR

Flatten the predicted and expected JSON to **leaf paths** (`user.name`, `items[0].id`), then
score how many leaves match — precision, recall, and their F1. A number in `[0, 1]`.

## What it measures & why you'd use it

Once [`json_validity`](json-validity.md) confirms the output parses, you want to know *how
right* the structured content is — not all‑or‑nothing, but field by field. `json_field_f1`
flattens both the predicted object and `expected_json` into a flat map of dotted / indexed
paths to leaf values, then treats correct leaves as true positives. This gives graceful,
partial credit for extraction: getting 4 of 5 fields right scores well, not zero.

## Intuition (tiny worked example)

Expected `{"a": 1, "nested": {"b": 2}}` flattens to `{"a": 1, "nested.b": 2}`. Output
`{"a": 1, "nested": {"b": 3}}` flattens to `{"a": 1, "nested.b": 3}`.

- Matching leaves: `a` (1 == 1) ✓, `nested.b` (3 ≠ 2) ✗ → 1 match.
- Precision = 1/2, recall = 1/2 → \(F_1 = 0.5.\)

That is exactly the value asserted in
[`tests/test_metric_catalog.py`](../../../tests/test_metric_catalog.py)
(`test_json_flattened_field_f1`).

## Formal definition

Flatten with `_flatten`: dicts recurse with `parent.key`, lists recurse with `parent[i]`,
scalars become leaves. With predicted leaf map \(P\), expected leaf map \(E\), and matches
\(m = |\{k \in E : P_k = E_k\}|\):

\[
\text{precision} = \frac{m}{|P|}, \quad \text{recall} = \frac{m}{|E|}, \quad F_1 = \frac{2PR}{P+R}.
\]

```200:218:src/evalharness/scoring/catalog.py
        expected = _flatten(case.expected_json)
        matches = sum(predicted.get(key) == value for key, value in expected.items())
        precision = matches / len(predicted) if predicted else 0.0
        recall = matches / len(expected) if expected else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return f1, {"precision": precision, "recall": recall}


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        ...
    if isinstance(value, list):
        ...
    return {prefix: value}
```

## Inputs & requirements

- **Task types:** `extraction`, `generation`.
- **Requires:** nothing declared, but returns `NULL` unless `case.expected_json` is set
  (`detail.reason = "missing_expected_json"`).
- **Invalid JSON output** → `0.0` (not `NULL`), `detail.reason = "invalid_json"`. This is a
  deliberate distinction: a *missing gold* is `NULL`; a *broken prediction* is a real `0`.

## Output & aggregation

- **Per‑case value:** \(F_1 \in [0, 1]\); `detail` has `precision` and `recall`.
- **Aggregate:** inherited **mean + Wilson** over thresholded successes (default `0.5`).
- **High vs low:** higher is better; `1.0` means every expected leaf matched and nothing
  extra was predicted.

## Registered name / version & config

- **Name / version:** `json_field_f1` / `1.0.0`.
- **Config:** inherited `threshold` (`0.5`) only.

## Pitfalls & gotchas

- **Exact leaf equality.** `1` vs `"1"`, `2.0` vs `2`, or a reordered list changes the leaf
  path/value and costs matches. List order matters because indices are part of the path
  (`items[0]` ≠ `items[1]`).
- **Invalid JSON scores `0`, not `NULL`.** Distinguish "wrong fields" from "unparseable"
  by reading `json_validity` alongside it — a `0.0` here with a `0.0` there means it never
  parsed.
- **Precision penalizes extra fields.** Emitting fields not in `expected_json` lowers
  precision even if every expected field is correct; keep the schema tight.
- **Nested vs flat gold.** The gold must be the *actual* expected structure; the flattener
  is faithful, so a mismatched nesting shape yields disjoint paths and a low score.

## How it composes

- The **content** half of structured scoring; [`json_validity`](json-validity.md) is the
  **structural** half. Report both.
- For numeric leaves that should match within tolerance rather than exactly, consider
  [`numeric_assertion`](numeric-assertion.md) on the relevant field.

## References & code

- Code: [`JsonFieldF1Metric`](../../../src/evalharness/scoring/catalog.py) and `_flatten`.
- Test: `test_json_flattened_field_f1` (asserts `0.5`).
- Guide: [§6.1](../../guide.md#61-deterministic--lexical).
