# `exact_match`

## TL;DR

Did the model say **exactly** the canonical answer, after we clean up trivial differences
(case, punctuation, articles, spacing)? Returns `1.0` for yes, `0.0` for no. This is the
default headline pass rate for the whole harness.

## What it measures & why you'd use it

Exact match answers the strictest fair question: *ignoring formatting noise, is the output
character‑for‑character the reference?* It is the right tool when there is **one canonical
answer** — short‑form QA ("What is the capital of France?" → "Paris"), a normalized label,
a slot value. It is cheap, perfectly reproducible, and impossible to game with fluff, which
is exactly why `evalctl run` computes it by default and reports it as the primary rate.

The subtlety is that "exact" without normalization is uselessly brittle: `"Paris."` ≠
`"paris"` ≠ `" Paris"`. So the metric normalizes both sides with a **versioned
`Normalizer`** before comparing.

## Intuition (tiny worked example)

Reference: `"The Blue Whale"`. Output: `"a blue whale."`

Normalizer (default config) lowercases, strips punctuation, and strips the articles
`{a, an, the}`:

- reference → `blue whale`
- output → `blue whale`

They match → **`value = 1.0`, `passed = True`.** Now change the output to `"blue whales"` →
normalizes to `blue whales` ≠ `blue whale` → **`0.0`.** Exact match gives no partial credit
for "close"; that is [`squad_f1`](squad-f1.md)'s job.

## Formal definition

Let \(N(\cdot)\) be the normalizer. Then

\[
\text{exact\_match}(o, r) = \mathbb{1}\big[\,N(o) = N(r)\,\big] \in \{0, 1\}.
\]

The normalizer ([`scoring/normalizer.py`](../../../src/evalharness/scoring/normalizer.py))
applies, in order and each toggle‑able via `NormalizerConfig`:

1. Unicode **NFKC** normalization (`unicode_nfkc`)
2. lowercase (`lowercase`)
3. strip punctuation — keep `\w`, whitespace, `.` and `-` (`strip_punctuation`)
4. optional numeric canonicalization to a tolerance (`numeric_tol`, default `None`/off)
5. strip articles `{a, an, the}` (`strip_articles`)
6. collapse whitespace + trim (`collapse_whitespace`)

## Inputs & requirements

- **Task types:** `generation`, `qa_short`, `summarization`, `rag`.
- **Requires:** `REFERENCE`. The reference is `case.reference_answer`, falling back to
  `case.references[0]` ([`core/models.py`](../../../src/evalharness/core/models.py)).
- **Reads:** `gen.output`, `case.reference_answer` / `case.references`.
- If either the reference or the output is missing, it returns `value=None`,
  `passed=None`, `detail={"reason": "missing_reference_or_output"}` — a **`NULL`, not a
  `0`**.

## Output & aggregation

- **Per‑case value:** `1.0` or `0.0` (or `NULL`). `passed` mirrors the value.
- **Aggregate:** the pass **rate** with a **Wilson 95% score interval**
  (`method="wilson"`), computed by `ExactMatchMetric.aggregate` via
  [`wilson_interval`](../statistics/wilson.md). `NULL`s are dropped before counting.
- **High vs low:** higher is better; `1.0` means every kept case matched exactly.

```56:84:src/evalharness/scoring/exact_match.py
    def aggregate(self, values: list[ScoreValue]) -> AggregateValue:
        valid = [v for v in values if v.value is not None]
        n = len(valid)
        ...
        successes = sum(1 for v in valid if v.passed)
        rate = successes / n
        ci_low, ci_high = wilson_interval(successes, n)
```

## Registered name / version & config

- **Name / version:** `exact_match` / `1.0.0`.
- **Config hash:** `metric_config_sha256` **is the `Normalizer` config id** — the SHA‑256
  of the `NormalizerConfig` dict. Two runs scored with different normalizer rules produce
  distinct score rows over the same generations, so a normalizer change is auditable rather
  than silent.
- **Knobs (`NormalizerConfig`):** `lowercase`, `strip_articles`, `strip_punctuation`,
  `collapse_whitespace`, `unicode_nfkc`, `numeric_tol`, `version`.
- **CLI:** default in `evalctl run`; also `evalctl runs rescore "$RUN_ID" --metrics
  exact_match` and `evalctl score outputs.jsonl --metrics exact_match`.

## Pitfalls & gotchas

- **Over‑normalizing hides real errors.** Stripping too much (e.g. turning `numeric_tol`
  on with a coarse tolerance) can mark a wrong answer as correct. Under‑normalizing inflates
  false failures. The config hash is your audit trail — never change the ruleset silently.
- **One canonical answer only.** Multiple acceptable phrasings need `references[]` +
  `squad_f1`/semantic similarity, not exact match.
- **Whitespace/casing in the *reference* matters too** — both sides are normalized, so a
  messy gold answer still normalizes, but a gold answer with meaningful punctuation (code,
  math) may be damaged by `strip_punctuation`. Consider a stricter config for those tasks.
- **`NULL` ≠ `0`.** A run full of missing references can show a deceptively high rate over
  a tiny denominator — always read `n` alongside the rate (see [Wilson](../statistics/wilson.md)).

## How it composes

- The **cheap gate** in the [composition story](../README.md#how-metrics-compose-the-honesty-layering):
  headline pass rate first, then a quality metric.
- Pairs with [`squad_f1`](squad-f1.md) for partial credit and
  [semantic similarity](../semantic-similarity/README.md) for paraphrase tolerance when a
  single canonical string is too strict.
- Feeds [`runs compare`](../statistics/mcnemar.md): the paired McNemar test operates on
  `exact_match.passed` booleans by default (see SQL query 11 in
  [guide §5.5](../../guide.md#55-query-library)).

## References & code

- Code: [`scoring/exact_match.py`](../../../src/evalharness/scoring/exact_match.py),
  normalizer [`scoring/normalizer.py`](../../../src/evalharness/scoring/normalizer.py).
- Test (golden behavior): [`tests/test_exact_match.py`](../../../tests/test_exact_match.py),
  [`tests/test_normalizer.py`](../../../tests/test_normalizer.py).
- Guide: [§6.1](../../guide.md#61-deterministic--lexical). SQL to read it back:
  [§5.5 query 1](../../guide.md#55-query-library).
- Lineage: the SQuAD evaluation script popularized normalized exact match for QA
  (Rajpurkar et al., 2016).
