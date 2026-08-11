# `json_validity`

## TL;DR

Does the output **parse as JSON** — and, if the case supplies a schema, does it **validate**
against that schema? `1.0` for yes, `0.0` for no.

## What it measures & why you'd use it

For extraction and tool‑use, the first question is not "are the fields right" but "is this
even machine‑readable?" A model that emits prose around its JSON, or trailing commas, or a
truncated object, breaks every downstream consumer. `json_validity` is the **structural
gate**: it confirms the output is parseable JSON and, optionally, conforms to a JSON Schema
you attach to the case. Answer *whether* first; measure *how right* with
[`json_field_f1`](json-field-f1.md).

## Intuition (tiny worked example)

- Output `'{"name": "Ada", "age": 36}'` → parses → **`1.0`**.
- Output `'{"name": "Ada", "age": 36'` (missing brace) → `json.loads` raises → **`0.0`**,
  `detail.error` holds the parse message.
- With `case.inputs["json_schema"]` requiring `age` to be a string, the same valid‑JSON
  output fails schema validation → **`0.0`** with the `ValidationError` message.

## Formal definition

\[
\text{json\_validity} =
\begin{cases}
1.0 & \text{if } \texttt{json.loads(output)} \text{ succeeds (and validates against schema if present)} \\
0.0 & \text{if it raises } \texttt{JSONDecodeError} \text{ or } \texttt{ValidationError}
\end{cases}
\]

```23:30:src/evalharness/scoring/metrics/structured/json_validity.py
        try:
            parsed = json.loads(gen.output or "")
            schema = case.inputs.get("json_schema")
            if schema:
                validate(parsed, schema)
            return 1.0, {"parsed": parsed, "schema_valid": True}
        except (json.JSONDecodeError, ValidationError) as exc:
            return 0.0, {"error": str(exc)}
```

Schema validation uses the `jsonschema` library's `validate(parsed, schema)`.

## Inputs & requirements

- **Task types:** `extraction`, `generation`, `tool_use` (**not** the full text set — this
  is deliberately scoped to structured tasks).
- **Requires:** nothing (`requires = frozenset()`). The optional schema comes from
  `case.inputs["json_schema"]`.
- **Missing output** → `json.loads("")` raises → `0.0`. There is **no `NULL` path** here: a
  missing or empty output is a real validity failure.

## Output & aggregation

- **Per‑case value:** `1.0` / `0.0`; `detail` carries the `parsed` object on success or the
  `error` string on failure.
- **Aggregate:** inherited **mean + Wilson**, `config = {"threshold": 1.0}`.
- **High vs low:** higher is better; the rate is the fraction of outputs that are valid
  (schema‑valid, if a schema was provided).

## Registered name / version & config

- **Name / version:** `json_validity` / `1.0.0`.
- **Config:** `{"threshold": 1.0}`. The schema is **per‑case data** (`inputs.json_schema`),
  not metric config — so two cases can enforce different schemas under the same metric.

## Pitfalls & gotchas

- **Valid ≠ correct.** A perfectly parseable `{}` is valid but useless. Always pair with
  `json_field_f1` for content.
- **Prose around JSON fails.** `"Here is the JSON: {…}"` does not parse. Constrain decoding
  (response format / tool schema) or strip the wrapper before scoring.
- **Schema strictness is your call.** Only the keys/types your schema constrains are
  checked; a loose schema passes near‑anything. Put the contract in the schema, not in
  hope.
- **Scoped task types.** Trying to score `json_validity` on a `qa_short` case raises in
  `ScoringEngine.validate` — it's only registered for extraction/generation/tool_use.

## How it composes

- The **structural gate** that must pass before [`json_field_f1`](json-field-f1.md) is even
  meaningful — a `0.0` here explains a `0.0` there.
- Together they answer the two structured questions: validity (this metric) and field
  accuracy (field‑F1). See the family [README](README.md).

## References & code

- Code: [`JsonValidityMetric`](../../../src/evalharness/scoring/metrics/structured/json_validity.py); `jsonschema`.
- Guide: [§6.1](../../guide.md#61-deterministic--lexical).
