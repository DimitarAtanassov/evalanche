"""Metric families, the unit the operator enables or disables.

The map is keyed by entry-point name so ``MetricRegistry.discover`` can drop a disabled
family without importing its module (and its dependencies) first.
"""

from __future__ import annotations

from enum import StrEnum

METRIC_ENTRY_POINT_GROUP = "evalharness.metrics"


class MetricFamily(StrEnum):
    LEXICAL = "lexical"
    STRUCTURED = "structured"
    CLASSIFICATION = "classification"
    RETRIEVAL = "retrieval"
    OVERLAP = "overlap"
    ML = "ml"
    # Anything registered on the entry-point group by another distribution.
    EXTERNAL = "external"


METRIC_FAMILIES: dict[str, MetricFamily] = {
    "exact_match": MetricFamily.LEXICAL,
    "squad_f1": MetricFamily.LEXICAL,
    "normalized_levenshtein": MetricFamily.LEXICAL,
    "assertions": MetricFamily.LEXICAL,
    "numeric_assertion": MetricFamily.LEXICAL,
    "json_validity": MetricFamily.STRUCTURED,
    "json_field_f1": MetricFamily.STRUCTURED,
    "classification": MetricFamily.CLASSIFICATION,
    "retrieval_ndcg_10": MetricFamily.RETRIEVAL,
    "retrieval_precision_at_k": MetricFamily.RETRIEVAL,
    "retrieval_mrr": MetricFamily.RETRIEVAL,
    "retrieval_map": MetricFamily.RETRIEVAL,
    "rouge_l": MetricFamily.OVERLAP,
    "chrf_pp": MetricFamily.OVERLAP,
    "sacrebleu": MetricFamily.OVERLAP,
    "meteor": MetricFamily.OVERLAP,
    "bertscore_f1": MetricFamily.ML,
}


def family_of(metric_name: str) -> MetricFamily:
    """The family a metric belongs to; unknown names are treated as third-party."""
    return METRIC_FAMILIES.get(metric_name, MetricFamily.EXTERNAL)
