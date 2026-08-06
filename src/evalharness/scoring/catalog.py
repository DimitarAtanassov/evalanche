"""Built-in lexical, structured, classification, retrieval, and overlap metrics."""

from __future__ import annotations

import json
import math
import re
import warnings
from collections import Counter
from typing import Any

import numpy as np
import sacrebleu
from jsonschema import ValidationError, validate
from rapidfuzz.distance import Levenshtein
from sacrebleu.metrics.bleu import BLEU
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

try:
    from rouge_score import rouge_scorer as _rouge_scorer
except ImportError:  # pragma: no cover - optional path when rouge_score is unavailable
    _rouge_scorer = None

try:
    from nltk.translate.meteor_score import meteor_score as _meteor_score
except ImportError:  # pragma: no cover - optional path when nltk is unavailable
    _meteor_score = None

from evalharness.core.constants import OVERALL_SLICE
from evalharness.core.enums import Requirement, TaskType
from evalharness.core.models import AggregateValue, Case, Generation, ScoreValue, ScoringContext
from evalharness.hashing import sha256_canonical
from evalharness.statistics import wilson_interval

ALL_TEXT_TASKS = frozenset(
    {TaskType.GENERATION, TaskType.QA_SHORT, TaskType.SUMMARIZATION, TaskType.RAG}
)


class ScalarMetric:
    name = "scalar"
    version = "1.0.0"
    task_types = ALL_TEXT_TASKS
    requires = frozenset({Requirement.REFERENCE})
    config: dict[str, Any] = {}

    @property
    def config_id(self) -> str:
        return sha256_canonical({"metric": self.name, "version": self.version, **self.config})

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        raise NotImplementedError

    def score(self, gen: Generation, case: Case, ctx: ScoringContext) -> list[ScoreValue]:
        value, detail = self.value(gen, case, ctx)
        return [
            ScoreValue(
                metric_name=self.name,
                metric_version=self.version,
                metric_config_sha256=self.config_id,
                value=value,
                passed=None if value is None else value >= float(self.config.get("threshold", 0.5)),
                detail=detail,
            )
        ]

    def aggregate(self, values: list[ScoreValue]) -> AggregateValue:
        valid = [float(value.value) for value in values if value.value is not None]
        mean = float(np.mean(valid)) if valid else 0.0
        low, high = (
            wilson_interval(
                sum(value >= float(self.config.get("threshold", 0.5)) for value in valid),
                len(valid),
            )
            if valid
            else (0.0, 0.0)
        )
        return AggregateValue(
            self.name,
            self.version,
            OVERALL_SLICE,
            len(valid),
            mean,
            low,
            high,
            float(np.std(valid)) if valid else None,
            "mean+wilson",
        )


def _reference(case: Case) -> str | None:
    return case.reference_answer or (case.references[0] if case.references else None)


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


class SquadMetric(ScalarMetric):
    name = "squad_f1"

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        reference = _reference(case)
        if gen.output is None or reference is None:
            return None, {"reason": "missing"}
        predicted, expected = _tokens(gen.output), _tokens(reference)
        common = Counter(predicted) & Counter(expected)
        overlap = sum(common.values())
        precision = overlap / len(predicted) if predicted else float(not expected)
        recall = overlap / len(expected) if expected else float(not predicted)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return f1, {"precision": precision, "recall": recall, "f1": f1}


class LevenshteinMetric(ScalarMetric):
    name = "normalized_levenshtein"

    def __init__(self, threshold: float = 0.8) -> None:
        self.threshold = threshold
        self.config = {"threshold": threshold}

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        reference = _reference(case)
        if gen.output is None or reference is None:
            return None, {"reason": "missing"}
        value = float(Levenshtein.normalized_similarity(gen.output, reference))
        return value, {"threshold": self.threshold}


class AssertionMetric(ScalarMetric):
    name = "assertions"
    requires = frozenset()
    config = {"threshold": 1.0}

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        if gen.output is None:
            return 0.0, {"reason": "missing_output"}
        folded = gen.output.casefold()
        required = {term: term.casefold() in folded for term in case.must_contain}
        forbidden = {term: term.casefold() not in folded for term in case.must_not_contain}
        checks = [*required.values(), *forbidden.values()]
        return float(all(checks)), {"contains": required, "forbidden": forbidden}


class NumericAssertionMetric(ScalarMetric):
    name = "numeric_assertion"
    config = {"threshold": 1.0, "abs_tol": 1e-6, "rel_tol": 1e-6}

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        reference = _reference(case)
        if gen.output is None or reference is None:
            return None, {"reason": "missing"}
        predicted_numbers = [float(value) for value in re.findall(r"[-+]?\d*\.?\d+", gen.output)]
        expected_numbers = [float(value) for value in re.findall(r"[-+]?\d*\.?\d+", reference)]
        passed = len(predicted_numbers) == len(expected_numbers) and all(
            math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-6)
            for left, right in zip(predicted_numbers, expected_numbers, strict=True)
        )
        return float(passed), {"prediction": predicted_numbers, "reference": expected_numbers}


class JsonValidityMetric(ScalarMetric):
    name = "json_validity"
    task_types = frozenset({TaskType.EXTRACTION, TaskType.GENERATION, TaskType.TOOL_USE})
    requires = frozenset()
    config = {"threshold": 1.0}

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        try:
            parsed = json.loads(gen.output or "")
            schema = case.inputs.get("json_schema")
            if schema:
                validate(parsed, schema)
            return 1.0, {"parsed": parsed, "schema_valid": True}
        except (json.JSONDecodeError, ValidationError) as exc:
            return 0.0, {"error": str(exc)}


class JsonFieldF1Metric(ScalarMetric):
    name = "json_field_f1"
    task_types = frozenset({TaskType.EXTRACTION, TaskType.GENERATION})
    requires = frozenset()

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        if case.expected_json is None:
            return None, {"reason": "missing_expected_json"}
        try:
            predicted = _flatten(json.loads(gen.output or ""))
        except json.JSONDecodeError:
            return 0.0, {"reason": "invalid_json"}
        expected = _flatten(case.expected_json)
        matches = sum(predicted.get(key) == value for key, value in expected.items())
        precision = matches / len(predicted) if predicted else 0.0
        recall = matches / len(expected) if expected else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return f1, {"precision": precision, "recall": recall}


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            result.update(_flatten(child, f"{prefix}.{key}" if prefix else key))
        return result
    if isinstance(value, list):
        result = {}
        for index, child in enumerate(value):
            result.update(_flatten(child, f"{prefix}[{index}]"))
        return result
    return {prefix: value}


class ClassificationMetric(ScalarMetric):
    name = "classification"
    task_types = frozenset({TaskType.CLASSIFICATION})
    requires = frozenset()
    config = {"threshold": 1.0}

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        if gen.output is None or case.expected_label is None:
            return None, {"reason": "missing"}
        predicted = gen.output.strip()
        return float(predicted == case.expected_label), {
            "predicted": predicted,
            "expected": case.expected_label,
        }

    def aggregate(self, values: list[ScoreValue]) -> AggregateValue:
        details = [value.detail for value in values if value.value is not None]
        expected = [str(item["expected"]) for item in details]
        predicted = [str(item["predicted"]) for item in details]
        accuracy = float(accuracy_score(expected, predicted)) if expected else 0.0
        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                expected, predicted, average="weighted", zero_division=0
            )
            if expected
            else (0.0, 0.0, 0.0, None)
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="y_pred contains classes not in y_true",
                category=UserWarning,
            )
            balanced_accuracy = (
                float(balanced_accuracy_score(expected, predicted)) if expected else 0.0
            )
        detail = {
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": float(f1_score(expected, predicted, average="macro")) if expected else 0.0,
            "micro_f1": float(f1_score(expected, predicted, average="micro")) if expected else 0.0,
            "weighted_precision": float(precision),
            "weighted_recall": float(recall),
            "weighted_f1": float(f1),
            "mcc": float(matthews_corrcoef(expected, predicted)) if expected else 0.0,
            "cohen_kappa": float(cohen_kappa_score(expected, predicted)) if expected else 0.0,
        }
        low, high = wilson_interval(
            sum(x == y for x, y in zip(expected, predicted, strict=True)), len(expected)
        )
        return AggregateValue(
            self.name,
            self.version,
            OVERALL_SLICE,
            len(expected),
            accuracy,
            low,
            high,
            None,
            json.dumps(detail, sort_keys=True),
        )


class RetrievalMetric(ScalarMetric):
    name = "retrieval_ndcg_10"
    task_types = frozenset({TaskType.RETRIEVAL, TaskType.RAG})
    requires = frozenset({Requirement.QRELS})
    config = {"threshold": 0.0, "cutoffs": [1, 3, 5, 10, 20]}

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        if not case.qrels:
            return None, {"excluded": "zero_relevance"}
        try:
            ranking = json.loads(gen.output or "[]")
        except json.JSONDecodeError:
            ranking = [part.strip() for part in (gen.output or "").split(",") if part.strip()]
        ranking = list(dict.fromkeys(map(str, ranking)))
        relevant = {str(doc): int(gain) for doc, gain in case.qrels.items() if int(gain) > 0}
        detail: dict[str, Any] = {}
        for cutoff in self.config["cutoffs"]:
            selected = ranking[:cutoff]
            hits = sum(doc in relevant for doc in selected)
            detail[f"precision@{cutoff}"] = hits / cutoff
            detail[f"recall@{cutoff}"] = hits / len(relevant)
            detail[f"hit@{cutoff}"] = float(hits > 0)
        ranks = [index + 1 for index, doc in enumerate(ranking) if doc in relevant]
        detail["mrr"] = 1 / min(ranks) if ranks else 0.0
        precisions = [
            sum(item in relevant for item in ranking[:rank]) / rank
            for rank, item in enumerate(ranking, start=1)
            if item in relevant
        ]
        detail["map"] = sum(precisions) / len(relevant)
        dcg = sum(
            (2 ** relevant.get(doc, 0) - 1) / math.log2(index + 2)
            for index, doc in enumerate(ranking[:10])
        )
        ideal = sorted(relevant.values(), reverse=True)[:10]
        idcg = sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(ideal))
        ndcg = dcg / idcg if idcg else 0.0
        detail["recall_ceiling"] = min(1.0, len(ranking) / len(relevant))
        return ndcg, detail


class RougeLMetric(ScalarMetric):
    name = "rouge_l"

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        reference = _reference(case)
        if gen.output is None or reference is None:
            return None, {"reason": "missing"}
        if _rouge_scorer is not None:
            scores = _rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"]).score(
                reference, gen.output
            )
            detail = {
                name: {
                    "precision": value.precision,
                    "recall": value.recall,
                    "fmeasure": value.fmeasure,
                }
                for name, value in scores.items()
            }
            return float(scores["rougeL"].fmeasure), detail
        # Deterministic fallback when rouge_score cannot be imported (e.g. NLTK
        # blocked on some Python 3.14 paths).
        predicted = _tokens(gen.output)
        expected = _tokens(reference)
        fallback_detail: dict[str, dict[str, float]] = {}
        for size, name in ((1, "rouge1"), (2, "rouge2")):
            predicted_ngrams = Counter(
                zip(*(predicted[index:] for index in range(size)), strict=False)
            )
            expected_ngrams = Counter(
                zip(*(expected[index:] for index in range(size)), strict=False)
            )
            overlap = sum((predicted_ngrams & expected_ngrams).values())
            fallback_detail[name] = _prf(
                overlap, sum(predicted_ngrams.values()), sum(expected_ngrams.values())
            )
        lcs = _lcs_length(predicted, expected)
        fallback_detail["rougeL"] = _prf(lcs, len(predicted), len(expected))
        fallback_detail["rougeLsum"] = fallback_detail["rougeL"]
        return fallback_detail["rougeL"]["fmeasure"], fallback_detail


def _prf(overlap: int, predicted: int, expected: int) -> dict[str, float]:
    precision = overlap / predicted if predicted else 0.0
    recall = overlap / expected if expected else 0.0
    fmeasure = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "fmeasure": fmeasure}


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            current.append(
                previous[index - 1] + 1 if token == other else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


class ChrfMetric(ScalarMetric):
    name = "chrf_pp"

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        reference = _reference(case)
        if gen.output is None or reference is None:
            return None, {"reason": "missing"}
        score = sacrebleu.sentence_chrf(gen.output, [reference], word_order=2)
        return score.score / 100, {"signature": "chrF2++", "raw_score": score.score}


class BleuMetric(ScalarMetric):
    name = "sacrebleu"

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        reference = _reference(case)
        if gen.output is None or reference is None:
            return None, {"reason": "missing"}
        score = sacrebleu.sentence_bleu(gen.output, [reference])
        return score.score / 100, {
            "sentence_score": score.score,
            "hypothesis": gen.output,
            "reference": reference,
        }

    def aggregate(self, values: list[ScoreValue]) -> AggregateValue:
        valid = [
            value for value in values if value.value is not None and "hypothesis" in value.detail
        ]
        metric = BLEU()
        score = (
            metric.corpus_score(
                [str(value.detail["hypothesis"]) for value in valid],
                [[str(value.detail["reference"]) for value in valid]],
            )
            if valid
            else None
        )
        return AggregateValue(
            self.name,
            self.version,
            OVERALL_SLICE,
            len(valid),
            (score.score / 100) if score else 0.0,
            None,
            None,
            None,
            f"corpus BLEU; {metric.get_signature()}" if score else "corpus BLEU",
        )


class MeteorMetric(ScalarMetric):
    name = "meteor"
    config = {"language": "en", "resources": ["wordnet"]}

    def value(
        self, gen: Generation, case: Case, ctx: ScoringContext
    ) -> tuple[float | None, dict[str, Any]]:
        reference = _reference(case)
        if gen.output is None or reference is None:
            return None, {"reason": "missing"}
        if _meteor_score is None:
            return None, {
                "reason": "nltk_resource_unavailable",
                "language": "en",
                "resource": "wordnet",
                "error": "nltk.translate.meteor_score is not installed",
            }
        try:
            value = float(_meteor_score([_tokens(reference)], _tokens(gen.output)))
            return value, {"language": "en", "resources": ["wordnet"]}
        except LookupError as exc:
            return None, {
                "reason": "nltk_resource_unavailable",
                "language": "en",
                "resource": "wordnet",
                "error": str(exc),
            }


def builtin_metrics() -> list[ScalarMetric]:
    return [
        SquadMetric(),
        LevenshteinMetric(),
        AssertionMetric(),
        NumericAssertionMetric(),
        JsonValidityMetric(),
        JsonFieldF1Metric(),
        ClassificationMetric(),
        RetrievalMetric(),
        RougeLMetric(),
        ChrfMetric(),
        BleuMetric(),
        MeteorMetric(),
    ]
