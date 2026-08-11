"""Domain public surface."""

from evalharness.domain.artifacts import StrictModel
from evalharness.domain.constants import (
    BASELINE_SCHEMA_VERSION,
    COMPARE_SCHEMA_VERSION,
    GATES_SCHEMA_VERSION,
    MATRIX_SCHEMA_VERSION,
    OVERALL_SLICE,
    PRIMARY_METRIC,
    REPORT_SCHEMA_VERSION,
    SUITE_SCHEMA_VERSION,
    SUPPLEMENT_SCHEMA_VERSION,
)
from evalharness.domain.dataset import Case, DatasetRef
from evalharness.domain.enums import (
    ErrorClass,
    FailureOutcome,
    FinishReason,
    Requirement,
    TaskType,
)
from evalharness.domain.generation import (
    Capabilities,
    Generation,
    GenerationRequest,
    GenerationResponse,
    Message,
    ModelVersion,
    StoredGeneration,
    TokenLogprob,
    ToolCall,
    ToolSpec,
)
from evalharness.domain.metric import Metric
from evalharness.domain.provider import Provider
from evalharness.domain.run import ModelVersionRef, PromptTemplateRef, RunRecord
from evalharness.domain.scoring import (
    AggregateValue,
    ScoreValue,
    ScoringContext,
    StoredAggregate,
    StoredScore,
)

__all__ = [
    "AggregateValue",
    "BASELINE_SCHEMA_VERSION",
    "COMPARE_SCHEMA_VERSION",
    "Capabilities",
    "Case",
    "DatasetRef",
    "ErrorClass",
    "FailureOutcome",
    "FinishReason",
    "GATES_SCHEMA_VERSION",
    "Generation",
    "GenerationRequest",
    "GenerationResponse",
    "MATRIX_SCHEMA_VERSION",
    "Message",
    "Metric",
    "ModelVersion",
    "ModelVersionRef",
    "OVERALL_SLICE",
    "PRIMARY_METRIC",
    "PromptTemplateRef",
    "Provider",
    "REPORT_SCHEMA_VERSION",
    "Requirement",
    "RunRecord",
    "SUITE_SCHEMA_VERSION",
    "SUPPLEMENT_SCHEMA_VERSION",
    "ScoreValue",
    "ScoringContext",
    "StoredAggregate",
    "StoredGeneration",
    "StoredScore",
    "StrictModel",
    "TaskType",
    "TokenLogprob",
    "ToolCall",
    "ToolSpec",
]
