# Metric catalog

## Purpose

This document explains **what evalanche measures, why each metric exists, and how the
metrics compose** into an honest verdict. It is the conceptual map. For the deep
drill‑down — beginner‑friendly intuition, formulas, edge cases, registered names/versions,
and code links — go to the **[metrics catalog](metrics-catalog/README.md)** (one
subdirectory per family). The compact formula/threshold reference also lives in
[guide.md §6](guide.md#6-metric-catalog--statistics), and the statistics are detailed in
[guide.md §6.7](guide.md#67-statistics-package). Read this first to know *which* metric to
reach for; follow the catalog when you need the math.

## The mental model

A metric is not a number — it is an **opinion about a generation, versioned so you can
trust the diff.** Every metric implements the `Metric` protocol
(`core/protocols.py`): a `name`, a `version`, the `task_types` it applies to, a
`requires` set (data prerequisites like `REFERENCE` or `QRELS`), a `score()` that
produces `ScoreValue`s for one generation, and an `aggregate()` that rolls many
`ScoreValue`s into an `AggregateValue`. **Aggregation is metric‑specific — never assume
`mean()`** (BLEU aggregates as a corpus score; classification aggregates as accuracy
with a confusion‑matrix detail; a pass rate aggregates with a Wilson interval).

Because scoring is a [separate stage](dataplane.md#the-two-stages), you can add or
change metrics and rescore historical runs with **zero inference cost**. Each score row
records `metric_config_sha256`, so a changed normalizer produces a *new* opinion beside
the old one rather than overwriting it.

Discover what is registered at any time:

```bash
uv run python -c "from evalharness.scoring.registry import MetricRegistry; \
print(MetricRegistry.defaults().names())"
```

`MetricRegistry.defaults()` registers `exact_match` plus the full catalog from
`scoring/catalog.py`. That is how every built‑in metric is registered; a built‑in never
goes through an entry point. The `evalharness.metrics` entry‑point group is reserved for
optional external extras that ship behind their own dependency group, today only
`bertscore` behind `metrics-ml`. `MetricRegistry.discover()` loads an entry point with no
arguments, so an entry‑point metric must be constructible without injected collaborators
(`exact_match` needs a `Normalizer`, which is why it lives in `defaults()`).
`evalctl run` scores `exact_match` by default; reach for the rest via
`evalctl runs rescore --metrics …`.

## The families and when to use them

Metrics fall into families that answer different questions. A good evaluation combines
several: a cheap deterministic tripwire, a task‑appropriate quality metric, and honest
statistics around the comparison.

| Family | Question it answers | Metrics (registry name) | Reach for it when… | Deep dive |
|--------|---------------------|-------------------------|--------------------|-----------|
| **Lexical / deterministic** | "Does the text match the reference exactly or almost?" | `exact_match`, `squad_f1`, `normalized_levenshtein` | Short‑form QA with canonical answers; fast regression gates | [lexical-structured/](metrics-catalog/lexical-structured/README.md) |
| **Assertions** | "Does the output contain / avoid required content?" | `assertions`, `numeric_assertion` | Content constraints, safety denylists, numeric answers | [lexical-structured/](metrics-catalog/lexical-structured/README.md) |
| **Structured** | "Is the output valid, schema‑correct JSON with the right fields?" | `json_validity`, `json_field_f1` | Extraction and tool‑use tasks | [lexical-structured/](metrics-catalog/lexical-structured/README.md) |
| **Classification** | "How good are the predicted labels, accounting for imbalance?" | `classification` | Label tasks (report MCC / balanced accuracy, not just accuracy) | [classification/](metrics-catalog/classification/README.md) |
| **Calibration** | "Does the model know when it doesn't know?" | `calibration.py` helpers, `evalctl calibrate` | You have confidences/logprobs and care about selective prediction | [calibration/](metrics-catalog/calibration/README.md) |
| **Retrieval / ranking** | "Are the right documents ranked highly?" | `retrieval_ndcg_10` | Retrieval / RAG with graded `qrels` | [retrieval-ranking/](metrics-catalog/retrieval-ranking/README.md) |
| **Overlap** | "How much surface content is shared with the reference?" | `rouge_l`, `sacrebleu`, `chrf_pp`, `meteor`, `bertscore_f1` (extra) | Summarization / translation regression tripwires | [text-overlap/](metrics-catalog/text-overlap/README.md) |
| **Semantic similarity** | "Is the meaning close, beyond exact wording?" | `scoring/embeddings.py` (`EmbeddingService`) | Paraphrase‑tolerant QA / semantic checks with a calibrated threshold | [semantic-similarity/](metrics-catalog/semantic-similarity/README.md) |
| **Statistics** | "Is the difference real, or noise?" | `statistics/` (Wilson, BCa, McNemar, BH, Cohen's h, pass@k, power) | Every published number and every A/B comparison | [statistics/](metrics-catalog/statistics/README.md) |

## Lexical / deterministic

The cheapest, most reproducible signals. `exact_match` v1.0.0 normalizes both sides
with a **versioned `Normalizer`** (unicode NFKC, lowercase, strip punctuation, strip
articles, collapse whitespace, optional numeric tolerance) and reports a **pass rate
with a Wilson 95% CI**. `squad_f1` relaxes to token‑overlap F1 for paraphrase tolerance;
`normalized_levenshtein` handles near‑copies and OCR‑ish noise with a calibrated
threshold. *Gotcha:* over‑normalizing hides real errors; the normalizer ruleset is
hashed into every score so changes are auditable, not silent. Full detail:
[metrics-catalog/lexical-structured/](metrics-catalog/lexical-structured/README.md),
[guide.md §6.1](guide.md#61-deterministic--lexical).

## Assertions

`assertions` checks `must_contain` / `must_not_contain` term presence;
`numeric_assertion` extracts numbers and compares them within tolerance. These are
constraint checks, not quality scores — ideal for safety smoke tests and answers with a
required numeric value. Detail:
[metrics-catalog/lexical-structured/](metrics-catalog/lexical-structured/README.md),
[guide.md §6.1](guide.md#61-deterministic--lexical).

## Structured

`json_validity` verifies the output parses (and optionally validates against an
`inputs.json_schema`); `json_field_f1` flattens nested objects/arrays and computes
per‑key precision/recall/F1 against `expected_json`. Use both together on extraction
tasks: validity tells you *whether* it parsed, field‑F1 tells you *how right* it was.
Detail: [metrics-catalog/lexical-structured/](metrics-catalog/lexical-structured/README.md),
[guide.md §6.1](guide.md#61-deterministic--lexical).

## Classification

`classification` scores per‑case label equality, but its `aggregate()` is where the
value is: accuracy with a Wilson CI plus a detail payload carrying balanced accuracy,
macro/micro/weighted F1, **Matthews correlation coefficient**, and Cohen's κ. Prefer MCC
and balanced accuracy under class imbalance — raw accuracy flatters a model that always
predicts the majority class. Detail:
[metrics-catalog/classification/](metrics-catalog/classification/README.md),
[guide.md §6.2](guide.md#62-classification).

## Calibration

Calibration answers a different question than accuracy: *is a confident answer actually
more likely to be right?* The `scoring/calibration.py` helpers compute adaptive ECE,
Brier score, NLL, a risk–coverage curve, AURC, accuracy at 80% coverage, and ROC/PR
AUC; `evalctl calibrate` fits an operating threshold on **development** data and reports
ROC‑AUC / PR‑AUC / dev F1. Never fit thresholds on holdout. This family needs real
confidences (logprobs or elicited) — skip it rather than invent them. Detail:
[metrics-catalog/calibration/](metrics-catalog/calibration/README.md),
[guide.md §6.3](guide.md#63-calibration-helpers-not-registry-metrics).

## Retrieval / ranking

`retrieval_ndcg_10` requires graded `qrels` and reports NDCG@10 as its primary value
(with P@k, R@k, Hit@k, MRR, MAP, and a recall ceiling in the detail across cutoffs
{1,3,5,10,20}). It uses the **exponential‑gain** DCG form — state that in reports,
because the linear‑gain variant gives different numbers and is a classic source of
cross‑team disagreement. Zero‑relevance queries are excluded, not scored as 0. Detail:
[metrics-catalog/retrieval-ranking/](metrics-catalog/retrieval-ranking/README.md),
[guide.md §6.4](guide.md#64-ranking--retrieval).

## Overlap (summarization / translation)

`rouge_l`, `sacrebleu`, `chrf_pp`, `meteor`, and the optional `bertscore_f1` measure how
much surface content a generation shares with a reference. Two honesty rules: report
**corpus** BLEU (not the mean of sentence BLEUs) with its SacreBLEU signature, and state
plainly that these metrics **correlate weakly with human judgment on abstractive
tasks** — treat them as regression tripwires, not quality verdicts. `bertscore_f1` lives
behind the `metrics-ml` extra because it pulls heavy model weights. Detail:
[metrics-catalog/text-overlap/](metrics-catalog/text-overlap/README.md),
[guide.md §6.5](guide.md#65-summarization--generation-overlap).

## Semantic similarity

When exact wording is too strict, `EmbeddingService` (`scoring/embeddings.py`) pins an
embedding model + revision, **L2‑normalizes** before cosine, dedupes by content hash,
and offers max‑reference and asymmetric‑centroid variants. Cosine has no absolute
meaning on its own — pick the operating threshold on **dev** via `evalctl calibrate` and
report ROC‑AUC, not a magic `0.8`. Note there is no registered `semantic_similarity`
metric today; the release evidence script writes those scores explicitly with BCa CIs.
Detail: [metrics-catalog/semantic-similarity/](metrics-catalog/semantic-similarity/README.md),
[guide.md §6.6](guide.md#66-semantic-similarity-embeddings).

## Statistics: honest comparison

Metrics produce point estimates; statistics decide whether a difference is real. The
`evalharness.statistics` package provides:

- **Wilson interval** — the default CI for any pass rate / Bernoulli aggregate.
- **BCa bootstrap** — bias‑corrected accelerated CIs for means of continuous metrics
  (ROUGE, cosine, …); seed it when publishing.
- **Paired bootstrap + exact McNemar** — for run‑vs‑run comparison of aligned
  `(case, repeat)` outcomes (`evalctl runs compare`).
- **Benjamini–Hochberg** — controls false discovery across many metrics/slices.
- **Cohen's h** — effect size beside the p‑value, so "significant" also means
  "meaningful".
- **pass@k** — the unbiased HumanEval‑style estimator when any of k samples may succeed.
- **Power / required sample size** — `evalctl power`, to size a study *before* spending
  inference.
- **Flaky‑case detection** — cases that disagree across repeats are excluded from
  comparison claims (never from raw storage).

The recurring theme is [principle #4](principles.md): **no point estimate without an
interval.** Small `n` yields honestly wide intervals — widen the dataset, don't shrink
the CI. Detail: [metrics-catalog/statistics/](metrics-catalog/statistics/README.md),
[guide.md §6.7](guide.md#67-statistics-package).

## How they compose (a worked shape)

A trustworthy evaluation typically layers families rather than picking one metric:

1. **Gate cheaply.** `exact_match` (or `assertions` / `json_validity`) as a fast,
   deterministic tripwire and the headline pass rate.
2. **Measure quality appropriately.** Add the task‑fit metric — `squad_f1` for QA,
   `classification` for labels, `retrieval_ndcg_10` for ranking, `rouge_l`/`sacrebleu`
   for summaries, semantic similarity for paraphrase tolerance.
3. **Separate infra from model.** Coverage (from the [outcome
   taxonomy](dataplane.md#outcome-taxonomy)) confirms harness failures didn't distort
   the denominator.
4. **Compare honestly.** Use `evalctl runs compare` for paired McNemar + BCa deltas +
   Cohen's h, with BH across everything you report, and exclude flaky cases.

Because all of this runs over stored generations, you can add step 2's metrics to a
completed run months later without re‑inferring a single token.

## Related

- **[metrics-catalog/](metrics-catalog/README.md)** — the in‑depth per‑family, per‑metric drill‑down
- [guide.md §6](guide.md#6-metric-catalog--statistics) — formulas, ranges, thresholds, gotchas
- [reports.md](reports.md) — where these numbers surface, and for whom
- [dataplane.md](dataplane.md) — coverage, publishability, and the scoring stage
- [principles.md](principles.md) — versioning and interval requirements
