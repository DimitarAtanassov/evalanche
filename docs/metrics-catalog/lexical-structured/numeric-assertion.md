# `numeric_assertion`

## TL;DR

Pull every number out of the output and the reference, and check they match **as numbers**
(within a tiny tolerance) — same count, same values. Pass (`1.0`) or fail (`0.0`).

## What it measures & why you'd use it

`"The total is 42.0 dollars"` and `"$42"` are the same answer, but `exact_match` and string
metrics disagree. When the *number* is what matters — arithmetic answers, unit conversions,
extracted quantities — you want to compare the numeric content and ignore the surrounding
prose. `numeric_assertion` extracts all numbers from both sides and compares them pairwise
within floating‑point tolerance.

## Intuition (tiny worked example)

Reference `"about 42 items"`, output `"There are 42 items."`

- Numbers extracted from output: `[42.0]`; from reference: `[42.0]`.
- Same count (1 == 1) and `math.isclose(42.0, 42.0)` → **`1.0`**.

Now output `"There are 42 items in 3 boxes."` → extracted `[42.0, 3.0]` vs reference
`[42.0]`. Counts differ (2 ≠ 1) → **`0.0`**. The metric requires the *same set of numbers in
the same order*, not just the presence of the right one.

## Formal definition

Extract numbers with the regex `[-+]?\d*\.?\d+` from output and reference, giving lists
\(P\) and \(E\). Pass iff \(|P| = |E|\) **and** every aligned pair is close:

\[
\text{pass} = \big(|P| = |E|\big) \wedge \bigwedge_i \text{isclose}(P_i, E_i;\ \text{rel\_tol}=10^{-6},\ \text{abs\_tol}=10^{-6}).
\]

```python
        predicted_numbers = [float(value) for value in re.findall(r"[-+]?\d*\.?\d+", gen.output)]
        expected_numbers = [float(value) for value in re.findall(r"[-+]?\d*\.?\d+", reference)]
        passed = len(predicted_numbers) == len(expected_numbers) and all(
            math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-6)
            for left, right in zip(predicted_numbers, expected_numbers, strict=True)
        )
        return float(passed), {"prediction": predicted_numbers, "reference": expected_numbers}
```

## Inputs & requirements

- **Task types:** `generation`, `qa_short`, `summarization`, `rag`.
- **Requires:** `REFERENCE`.
- Missing output/reference → `value=None`, `detail={"reason": "missing"}`.

## Output & aggregation

- **Per‑case value:** `1.0` / `0.0`; `detail` carries the extracted `prediction` and
  `reference` number lists — inspect these when debugging a failure.
- **Aggregate:** inherited **mean + Wilson**; `config = {"threshold": 1.0, "abs_tol": 1e-6,
  "rel_tol": 1e-6}`, so the tolerances are part of the config hash.
- **High vs low:** higher is better; the rate is the fraction of outputs whose numbers all
  match.

## Registered name / version & config

- **Name / version:** `numeric_assertion` / `1.0.0`.
- **Config:** `threshold = 1.0`, `abs_tol = 1e-6`, `rel_tol = 1e-6`. **Caveat worth
  knowing:** the tolerances in `config` are advertised, but the comparison in code passes
  literal `1e-6` values to `math.isclose` rather than reading them from `config` — so today
  the tolerance is effectively fixed at `1e-6`. Treat the config values as documentation of
  intent; if you need a looser tolerance, that is a code change, not just a config flip.

## Pitfalls & gotchas

- **Count must match exactly.** Extra numbers anywhere (dates, footnotes, "3 boxes") fail
  the case even if the target number is present. Constrain the output or pre‑extract the
  field for extraction tasks.
- **Order matters.** Numbers are compared positionally via `zip(..., strict=True)`; a
  reordered list fails.
- **Regex quirks.** The pattern splits on non‑numeric characters, so `"1,000"` becomes
  `[1, 0]` (the comma breaks it) and `"1.2.3"` becomes `[1.2, 3]`. Currency symbols and
  thousands separators are not understood — normalize upstream if your data uses them.
- **Tight tolerance.** `1e-6` is strict; `0.1 + 0.2` style float answers may fail. It is
  meant for exact numeric answers, not measured quantities.

## How it composes

- The numeric counterpart to [`assertions`](assertions.md): use `assertions` for phrase
  constraints and `numeric_assertion` for the quantity.
- Common on extraction / tool‑use answers alongside [`json_field_f1`](json-field-f1.md)
  when the numeric field lives inside a JSON payload.

## References & code

- Code: [`NumericAssertionMetric`](../../../src/evalharness/scoring/metrics/lexical/numeric_assertion.py);
  `math.isclose`.
- Guide: [§6.1](../../guide.md#61-deterministic--lexical).
