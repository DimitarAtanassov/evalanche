"""Emit deterministic catalog conformance evidence."""

from __future__ import annotations

import json
from pathlib import Path

import sacrebleu

from evalharness.domain.dataset import Case
from evalharness.domain.enums import FailureOutcome, TaskType
from evalharness.domain.generation import Generation
from evalharness.domain.scoring import ScoringContext
from evalharness.scoring.calibration import calibration_metrics
from evalharness.scoring.metrics.classification.labels import ClassificationMetric
from evalharness.scoring.metrics.overlap.rouge_l import RougeLMetric
from evalharness.scoring.metrics.retrieval.ndcg import RetrievalMetric


def generation(output: str) -> Generation:
    return Generation(
        None,
        "conformance",
        "case",
        0,
        output,
        [],
        None,
        FailureOutcome.PASSED,
        None,
        None,
        0.0,
        None,
        None,
        None,
        1,
        [],
        False,
        None,
        None,
    )


def main() -> None:
    context = ScoringContext("conformance")
    retrieval = RetrievalMetric().score(
        generation('["d1","d2","d3"]'),
        Case("retrieval", TaskType.RETRIEVAL, {}, qrels={"d1": 3, "d2": 2, "d3": 1}),
        context,
    )[0]
    classification = ClassificationMetric().aggregate(
        [
            ClassificationMetric().score(
                generation(predicted),
                Case(str(index), TaskType.CLASSIFICATION, {}, expected_label=expected),
                context,
            )[0]
            for index, (predicted, expected) in enumerate(
                [("a", "a"), ("b", "b"), ("a", "b"), ("c", "c")]
            )
        ]
    )
    rouge = RougeLMetric().score(
        generation("the cat sat on the mat"),
        Case("summary", TaskType.SUMMARIZATION, {}, reference_answer="the cat is on the mat"),
        context,
    )[0]
    bleu_metric = sacrebleu.metrics.BLEU()
    bleu = bleu_metric.corpus_score(["the cat sat on the mat"], [["the cat is on the mat"]])
    evidence = {
        "schema_version": "1.0",
        "classification": {
            "accuracy": classification.value,
            "method_detail": json.loads(classification.method),
        },
        "calibration": calibration_metrics([True, True, False, True], [0.95, 0.8, 0.7, 0.6]),
        "retrieval": {
            "ndcg_at_10": retrieval.value,
            "trec_eval_reference": 1.0,
            "absolute_error": abs(float(retrieval.value or 0) - 1.0),
            "tolerance": 1e-6,
            **retrieval.detail,
        },
        "summarization": {
            "rouge": rouge.detail,
            "sacrebleu": bleu.score,
            "sacrebleu_signature": str(bleu_metric.get_signature()),
        },
    }
    path = Path("release/v0.2.0/catalog-conformance.json")
    path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
