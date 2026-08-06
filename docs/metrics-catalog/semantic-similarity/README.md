# Semantic similarity (embeddings)

**Is the *meaning* close, beyond exact wording?** When surface metrics are too strict — a
correct paraphrase that shares few words — embeddings let you compare *meaning*. Text is
mapped to a high‑dimensional vector; two texts are "similar" when their vectors point in
nearly the same direction (cosine similarity).

This family is **not a registered `Metric`.** It is the
[`EmbeddingService`](../../../src/evalharness/scoring/embeddings.py) — an async service that
turns text into normalized vectors and offers similarity helpers. There is **no
`semantic_similarity` entry in `MetricRegistry`** today; the release evidence script
(`scripts/run_release_e2e.py`) writes these scores explicitly with
[BCa CIs](../statistics/bootstrap.md). That is a deliberate, documented gap — see
[guide §8.4](../../guide.md#84-known-gaps--deferred).

This README is the in‑depth doc for the whole family.

---

## `EmbeddingService`

### 1. TL;DR

Embed prediction and references into L2‑normalized vectors (dedup by content hash, pinned
model + revision), then score similarity as cosine — either the max over references, or
against the reference centroid.

### 2. What it measures & why you'd use it

Cosine similarity of embeddings captures semantic closeness that lexical overlap misses:
`"the film was great"` and `"loved the movie"` land near each other in embedding space.
Reach for it on paraphrase‑tolerant QA and semantic equivalence checks — but only with a
**calibrated threshold**, because a raw cosine has no absolute meaning.

### 3. Intuition (tiny worked example)

Prediction `"a large marine mammal"` vs references `["the blue whale", "a big sea animal"]`.
`cosine_max_reference` embeds all three, L2‑normalizes, and returns the **best** match — say
`0.71` against `"a big sea animal"`. Whether `0.71` counts as "correct" is *not* inherent; you
decide by fitting a threshold on dev data (§8).

### 4. Formal definition

Vectors are L2‑normalized, so cosine is a plain dot product. For prediction \(p\) and
references \(r_1..r_m\):

\[
\text{cosine\_max\_reference} = \max_j \langle \hat p, \hat r_j \rangle, \qquad
\text{asymmetric\_similarity} = \Big\langle \hat p,\ \widehat{\tfrac{1}{m}\textstyle\sum_j \hat r_j} \Big\rangle.
\]

```44:58:src/evalharness/scoring/embeddings.py
                norm = math.sqrt(sum(value * value for value in vector))
                if norm == 0:
                    raise ValueError("Zero-norm embedding")
                self._cache[key] = [value / norm for value in vector]
        return [self._cache[key] for key in keys]

    async def cosine_max_reference(self, prediction: str, references: list[str]) -> float:
        if not references:
            raise ValueError("At least one reference is required")
        vectors = await self.embed([prediction, *references])
        prediction_vector = vectors[0]
        return max(
            sum(left * right for left, right in zip(prediction_vector, reference, strict=True))
            for reference in vectors[1:]
        )
```

### 5. Inputs & requirements

- **Construction:** `EmbeddingService(provider, model, revision, *, dimension=1024,
  batch_size=64)`. Needs a `Provider` that implements `embed()`
  ([`core/protocols.py`](../../../src/evalharness/core/protocols.py)) and corresponds to
  `Requirement.EMBEDDINGS`.
- **Contracts enforced in code:** every returned vector must match `dimension` (else
  `ValueError`), and zero‑norm vectors are rejected. Content is **deduped by SHA‑256** in an
  in‑memory cache, so repeated texts are embedded once.
- **Variants:** `asymmetric_similarity(prediction, references, *, variant=...)` accepts
  `"prediction_to_reference"` or `"reference_to_prediction"`; the cosine itself is
  symmetric, but the **variant label is carried for threshold provenance** so you calibrate
  per direction.

### 6. Output & aggregation

- **Per‑call value:** a cosine in `[-1, 1]` (typically `[0, 1]` for these models).
- **Aggregation is not built in** — there's no `Metric.aggregate`. The release script
  aggregates the mean cosine with a **[BCa bootstrap CI](../statistics/bootstrap.md)** and
  seeds it. Do the same in research code.
- **High vs low:** higher = more semantically similar; the *decision boundary* is the
  calibrated threshold, not a fixed number.

### 7. Config knobs & provenance

- `model`, `revision` — **pinned**; different weights ⇒ different geometry ⇒ not comparable.
- `dimension` — default **1024**. The release e2e overrides to **768** for
  `nomic-embed-text`. Keep dimension in the metric config hash and never force mismatched
  inserts (see the pgvector caveat below).
- `batch_size` — throughput only.

### 8. Pitfalls & gotchas

- **Cosine has no absolute meaning.** `0.8` is not "correct." **Fit the threshold on dev**
  with [`calibrate_threshold`](../calibration/threshold-calibration.md) and report ROC‑AUC /
  PR‑AUC / operating point — never ship a magic constant to holdout.
- **Calibrate per variant/direction.** Because provenance distinguishes
  `prediction_to_reference` vs `reference_to_prediction`, maintain a separate threshold per
  variant; don't reuse one across directions.
- **Max‑reference vs centroid differ.** `cosine_max_reference` rewards matching *any* one
  reference (lenient); the asymmetric centroid rewards matching the *average* of references
  (stricter, smoother). Pick deliberately.
- **Dimension mismatch is a real trap.** Local embedders disagree on width (nomic is
  768‑d; other models differ). The hot path is **in‑memory** by design; a durable
  pgvector write path + HNSW is [deferred](../../guide.md#84-known-gaps--deferred).
  Pin dimension into any future metric config hash.
- **Not registered.** You can't `runs rescore --metrics semantic_similarity`; score it in
  research/release code until a metric is added.

### 9. How it composes

- The meaning layer above the [lexical family](../lexical-structured/README.md): gate with
  `exact_match`, give partial credit with `squad_f1`, then catch valid paraphrases with
  cosine.
- Shares the **dev‑set threshold calibration** workflow with
  [`normalized_levenshtein`](../lexical-structured/normalized-levenshtein.md) and confidence
  [calibration](../calibration/threshold-calibration.md).
- Reference‑anchored, token‑level meaning matching lives in
  [BERTScore](../text-overlap/bertscore.md) — a related but distinct tool.

### 10. References & code

- Code: [`EmbeddingService`](../../../src/evalharness/scoring/embeddings.py); release usage
  in `scripts/run_release_e2e.py`.
- Guide: [§6.6](../../guide.md#66-semantic-similarity-embeddings),
  [§8.4](../../guide.md#84-known-gaps--deferred).
- Narrative: [`metrics.md`](../../metrics.md#semantic-similarity).
