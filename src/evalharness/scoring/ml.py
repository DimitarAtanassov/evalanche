"""Optional ML-backed metrics (installed with ``metrics-ml``)."""

from __future__ import annotations

from typing import Any

from evalharness.core.enums import Requirement
from evalharness.core.models import Case, Generation, ScoringContext
from evalharness.scoring.catalog import ALL_TEXT_TASKS, ScalarMetric, _reference


class BERTScoreMetric(ScalarMetric):
    name = "bertscore_f1"
    version = "1.0.0"
    task_types = ALL_TEXT_TASKS
    requires = frozenset({Requirement.REFERENCE})

    def __init__(
        self,
        *,
        model_type: str = "microsoft/deberta-xlarge-mnli",
        revision: str = "7d9f5b4",
        num_layers: int = 40,
        language: str = "en",
        rescale_with_baseline: bool = True,
    ) -> None:
        self.config: dict[str, Any] = {
            "model_type": model_type,
            "revision": revision,
            "num_layers": num_layers,
            "language": language,
            "rescale_with_baseline": rescale_with_baseline,
        }

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        reference = _reference(case)
        if gen.output is None or reference is None:
            return None, {"reason": "missing"}
        from bert_score import score

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
