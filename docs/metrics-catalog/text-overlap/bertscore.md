# `bertscore_f1` (optional `metrics-ml` extra)

> **Availability.** BERTScore is **not** in the default registry. It is registered via the
> `evalharness.metrics` entry point and only importable when the **`metrics-ml`** extra is
> installed (it pulls heavy transformer weights). This is a deliberately isolated,
> opt‑in metric — see [guide §8.4](../../guide.md#84-known-gaps--deferred) and
> [`scoring/ml.py`](../../../src/evalharness/scoring/ml.py). It is **shipped**, not
> roadmap — just gated behind an extra so the core install stays light.

## TL;DR

Instead of matching surface words, BERTScore matches **contextual embeddings** of tokens
between output and reference, so paraphrases with different words but the same meaning score
high. Reported as an F1 in roughly `[0, 1]`.

## What it measures & why you'd use it

Every other metric in this family counts surface overlap and therefore punishes valid
paraphrases. BERTScore embeds each token with a pretrained transformer, greedily matches
each output token to its most similar reference token (and vice versa) by cosine similarity,
and aggregates into precision/recall/F1. It correlates with human judgment substantially
better than BLEU/ROUGE on abstractive tasks — at the cost of loading a large model. Use it
when overlap tripwires are too blunt and you can pay for GPU/CPU inference.

## Intuition (tiny worked example)

Reference `"the film was excellent"`, output `"the movie was superb"`. Word‑overlap metrics
score near‑zero. BERTScore matches `movie`↔`film` and `superb`↔`excellent` by embedding
similarity → a high F1. That robustness to synonymy is the reason to reach for it.

## Formal definition

With contextual embeddings, greedy‑match each token to its max‑cosine counterpart:
precision over output tokens, recall over reference tokens, F1 their harmonic mean; optional
**baseline rescaling** shifts scores off the compressed high range into a more
interpretable spread.

```43:56:src/evalharness/scoring/ml.py
        precision, recall, f1 = score(
            [gen.output],
            [reference],
            model_type=str(self.config["model_type"]),
            num_layers=int(self.config["num_layers"]),
            lang=str(self.config["language"]),
            rescale_with_baseline=bool(self.config["rescale_with_baseline"]),
            verbose=False,
        )
        return float(f1[0]), {
            **self.config,
            "precision": float(precision[0]),
            "recall": float(recall[0]),
        }
```

## Inputs & requirements

- **Task types:** `ALL_TEXT_TASKS`. **Requires:** `REFERENCE`.
- **Requires the `metrics-ml` extra** (`bert_score` + torch + the pinned model). Without it,
  importing the metric fails — it is intentionally absent from `MetricRegistry.defaults()`.
- Missing output/reference → `NULL`.

## Output & aggregation

- **Per‑case value:** BERTScore **F1** (with rescale‑with‑baseline, values can dip slightly
  below 0 / above the naive range); `detail` carries `precision`, `recall`, and the full
  pinned config.
- **Aggregate:** inherited **mean + Wilson**; for a published mean use a
  [BCa CI](../statistics/bootstrap.md).

## Registered name / version & config

- **Name / version:** `bertscore_f1` / `1.0.0`.
- **Config (all pinned into `metric_config_sha256`):**
  - `model_type = "microsoft/deberta-xlarge-mnli"`
  - `revision = "7d9f5b4"`
  - `num_layers = 40`
  - `language = "en"`
  - `rescale_with_baseline = True`

  **Pin everything.** BERTScore numbers are only comparable across runs with the *same*
  model, revision, and layer — which is exactly why all of it is in the config hash.

## Pitfalls & gotchas

- **Heavy and slow.** Pulls large weights; isolate behind the extra and don't put it in a
  fast CI gate.
- **Not comparable across model/layer changes.** Change the backbone or `num_layers` and you
  have a *different* metric config (new hash) — never compare across them.
- **Rescaling matters.** Raw BERTScore is compressed near 1.0; `rescale_with_baseline`
  spreads it out. Report whether rescaling was on (it is, here).
- **Still reference‑based.** It rewards similarity to *a* reference; a correct answer unlike
  the reference still scores lower.

## How it composes

- The meaning‑aware top of the overlap family; where [ROUGE](rouge.md)/[BLEU](sacrebleu.md)
  are cheap tripwires, BERTScore is the paraphrase‑tolerant quality signal — analogous to
  [semantic cosine](../semantic-similarity/README.md) but reference‑anchored and token‑level.

## References & code

- Code: [`BERTScoreMetric`](../../../src/evalharness/scoring/ml.py); `bert_score`.
- Guide: [§6.5](../../guide.md#65-summarization--generation-overlap),
  [§8.4](../../guide.md#84-known-gaps--deferred).
- Lineage: Zhang et al. (2020), "BERTScore: Evaluating Text Generation with BERT."
