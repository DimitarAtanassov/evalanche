# `assertions`

## TL;DR

Does the output **contain every required phrase** and **avoid every forbidden phrase**?
Pass (`1.0`) only if all checks hold; otherwise fail (`0.0`). A hard constraint gate, not a
quality score.

## What it measures & why you'd use it

Some requirements are binary and non‑negotiable: the answer must mention the safety
disclaimer; it must never leak the customer's SSN; it must include the ticket number. These
are **constraints**, not gradations of quality. `assertions` checks two lists on the case —
`must_contain` and `must_not_contain` — and passes only if *all* required terms are present
and *all* forbidden terms are absent. It is the natural tool for safety smoke tests and
content‑policy denylists.

## Intuition (tiny worked example)

Case: `must_contain = ["refund", "5-7 days"]`, `must_not_contain = ["guarantee"]`.

- Output `"Your refund arrives in 5-7 days."` → both required present, forbidden absent →
  **`1.0`**.
- Output `"We guarantee your refund in 5-7 days."` → forbidden `guarantee` present →
  **`0.0`**, even though both required phrases are there. One violated check fails the whole
  assertion.

## Formal definition

With case‑folded output \(o\), required set \(C\), forbidden set \(F\):

\[
\text{assertions} = \mathbb{1}\Big[\big(\forall t \in C:\ t \in o\big) \wedge \big(\forall t \in F:\ t \notin o\big)\Big].
\]

```19:25:src/evalharness/scoring/metrics/lexical/assertions.py
        if gen.output is None:
            return 0.0, {"reason": "missing_output"}
        folded = gen.output.casefold()
        required = {term: term.casefold() in folded for term in case.must_contain}
        forbidden = {term: term.casefold() not in folded for term in case.must_not_contain}
        checks = [*required.values(), *forbidden.values()]
        return float(all(checks)), {"contains": required, "forbidden": forbidden}
```

Matching is **case‑insensitive substring** containment (`casefold`), not token or regex
matching.

## Inputs & requirements

- **Task types:** `generation`, `qa_short`, `summarization`, `rag` (`ALL_TEXT_TASKS`).
- **Requires:** nothing (`requires = frozenset()`) — it reads only `gen.output`,
  `case.must_contain`, `case.must_not_contain`. No reference needed.
- **Missing output** → `0.0` (an *empty* output fails required‑term checks), with
  `detail.reason = "missing_output"`. Note: this is a **hard `0.0`, not `NULL`** — an absent
  output is a real constraint failure here.
- **Empty lists** → vacuously true; with both lists empty the metric always returns `1.0`.

## Output & aggregation

- **Per‑case value:** `1.0` / `0.0`; `detail` shows which `contains`/`forbidden` checks
  passed.
- **Aggregate:** inherited **mean + Wilson**, with `config = {"threshold": 1.0}` — a case
  only "passes" at a perfect `1.0`.
- **High vs low:** higher is better; the rate is the fraction of outputs satisfying *all*
  constraints.

## Registered name / version & config

- **Name / version:** `assertions` / `1.0.0`.
- **Config:** `{"threshold": 1.0}` (all‑or‑nothing pass).

## Pitfalls & gotchas

- **Substring, not word‑boundary.** `must_not_contain = ["ass"]` will fire on `"assistant"`.
  Choose terms carefully or the check over‑triggers.
- **No fuzzy match.** A required `"5-7 days"` won't match `"5 to 7 days"`. Assertions are
  literal by design — for tolerance, use a different family.
- **All‑or‑nothing.** One failed check zeroes the case; the `detail` payload is where you
  see *which* one, so always inspect it when a case fails.
- **Not a quality metric.** Passing all assertions doesn't mean the answer is good — pair
  it with a quality metric.

## How it composes

- A **cheap gate** alongside `exact_match` / `json_validity` in step 1 of the
  [composition story](../README.md#how-metrics-compose-the-honesty-layering).
- On `safety` and policy tasks it's often the *primary* signal; combine with the
  [outcome taxonomy](../../dataplane.md#outcome-taxonomy) to separate refusals from
  violations.
- Numeric constraints belong in [`numeric_assertion`](numeric-assertion.md), which compares
  extracted numbers within tolerance rather than as substrings.

## References & code

- Code: [`AssertionMetric`](../../../src/evalharness/scoring/metrics/lexical/assertions.py).
- Guide: [§6.1](../../guide.md#61-deterministic--lexical).
