"""Build ``rag_evidence.json`` schema 0.1 from report + evidence files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from evalharness.domain.constants import OVERALL_SLICE, REPORT_SCHEMA_VERSION
from evalharness.hashing import canonical_json, sha256_hex
from evalharness.observability import get_logger
from evalharness.rag.citations import citation_attribution
from evalharness.rag.context import context_precision_recall
from evalharness.rag.errors import RagError
from evalharness.rag.faithfulness import NliLabel, build_faithfulness, load_mock_nli_responses
from evalharness.rag.text import EXAMPLE_LIMIT, MAX_CONTEXTS_PER_CASE, truncate_span

logger = get_logger(__name__)


class RetrievedContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1)
    text: str = ""
    rank: int = Field(ge=1)


class CitationRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_index: int = Field(ge=0)
    doc_id: str = Field(min_length=1)


class EvidenceCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    answer_text: str
    retrieved_contexts: list[RetrievedContext]
    qrels: dict[str, int] | None = None
    citations: list[CitationRef] | None = None
    claims: list[str] | None = None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RagError("MISSING_ARTIFACT", str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RagError("INVALID_ARTIFACT", f"{path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RagError("INVALID_ARTIFACT", f"{path}: expected a JSON object")
    return payload


def _load_evidence(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RagError("MISSING_ARTIFACT", str(path))
    cases: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RagError("INVALID_EVIDENCE", f"{path}: {exc}") from exc
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            case = EvidenceCase.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise RagError("INVALID_EVIDENCE", f"{path}:{index}: {exc}") from exc
        bounded = case.model_dump(mode="json")
        bounded["retrieved_contexts"] = [
            {
                **ctx,
                "text": truncate_span(str(ctx.get("text") or "")),
            }
            for ctx in bounded["retrieved_contexts"][:MAX_CONTEXTS_PER_CASE]
        ]
        cases.append(bounded)
    if not cases:
        raise RagError("INVALID_EVIDENCE", f"{path}: no cases found")
    return cases


def _retrieval_section(report: dict[str, Any]) -> dict[str, Any]:
    aggregates = report.get("metric_aggregates")
    if not isinstance(aggregates, list):
        aggregates = []
    match: dict[str, Any] | None = None
    for row in aggregates:
        if (
            isinstance(row, dict)
            and row.get("metric") == "retrieval_ndcg_10"
            and row.get("slice") in {None, OVERALL_SLICE, "overall"}
        ):
            match = row
            break
    if match is None:
        for row in aggregates:
            if isinstance(row, dict) and row.get("metric") == "retrieval_ndcg_10":
                match = row
                break
    if match is None:
        return {
            "metric": "retrieval_ndcg_10",
            "metric_version": None,
            "status": "missing",
            "reason": "QRELS_MISSING",
            "aggregate": {"value": None, "n": 0, "ci_low": None, "ci_high": None},
            "notes": "From scores table; independent of faithfulness",
        }
    return {
        "metric": "retrieval_ndcg_10",
        "metric_version": match.get("version"),
        "status": "ok",
        "aggregate": {
            "value": match.get("value"),
            "n": match.get("n", 0),
            "ci_low": match.get("ci_low"),
            "ci_high": match.get("ci_high"),
        },
        "notes": "From scores table; independent of faithfulness",
    }


def _mock_nli_identity(model: str) -> dict[str, str]:
    digest = "".join(f"{ord(char):02x}" for char in model)[:64].ljust(64, "0")
    return {
        "provider": "mock",
        "model": model,
        "resolved_version": f"sha256:{digest}",
    }


def _write_rag_artifact(
    *,
    report: dict[str, Any],
    cases: list[dict[str, Any]],
    output_path: Path,
    nli_labels: dict[tuple[str, int, str], NliLabel] | None,
    nli_config: dict[str, Any],
    cost_usd_total: float,
    missing_label_code: str,
) -> dict[str, Any]:
    faithfulness, used_labels, claim_notes = build_faithfulness(
        cases,
        nli_labels=nli_labels,
        missing_label_code=missing_label_code,
    )
    context = context_precision_recall(cases)
    citations = citation_attribution(
        cases,
        nli_labels=used_labels if nli_labels is not None else None,
    )
    retrieval = _retrieval_section(report)

    bounded_examples: list[dict[str, Any]] = []
    for case in cases:
        if len(bounded_examples) >= EXAMPLE_LIMIT:
            break
        bounded_examples.append(
            {
                "case_id": case["case_id"],
                "generation_id": case["generation_id"],
                "answer_text": truncate_span(str(case.get("answer_text") or "")),
                "retrieved_doc_ids": [
                    str(ctx.get("doc_id")) for ctx in case.get("retrieved_contexts") or []
                ],
            }
        )

    artifact: dict[str, Any] = {
        "schema_version": "0.1",
        "run_id": report.get("run_id"),
        "model_digest": report.get("model_digest"),
        "dataset_sha256": report.get("dataset_sha256"),
        "config": nli_config,
        "retrieval": retrieval,
        "faithfulness": faithfulness,
        "context": context,
        "citations": citations,
        "deferred": {
            "answer_grounded_context": {
                "status": "deferred",
                "reason": "ADR_004_RAG_METHODS",
                "value": None,
            },
            "nli_verified_citations": {
                "status": "deferred",
                "reason": "ADR_004_RAG_METHODS",
                "value": None,
            },
        },
        "bounded_examples": bounded_examples,
        "claim_parse_notes": claim_notes,
        "cost_usd_total": cost_usd_total,
        "gating_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "rag_evidence_finished",
        run_id=artifact["run_id"],
        faithfulness_status=faithfulness.get("status"),
        retrieval_status=retrieval.get("status"),
        gating_allowed=False,
    )
    return artifact


def build_rag_evidence(
    *,
    report_path: Path,
    evidence_path: Path,
    output_path: Path,
    nli_provider: str | None = None,
    nli_model: str | None = None,
    nli_responses_path: Path | None = None,
) -> dict[str, Any]:
    """Assemble the RAG evidence artifact from a run report and local evidence JSONL."""
    report = _read_json(report_path)
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise RagError(
            "UNSUPPORTED_SCHEMA",
            f"{report_path}: expected report schema {REPORT_SCHEMA_VERSION}, "
            f"got {report.get('schema_version')!r}",
        )
    cases = _load_evidence(evidence_path)

    nli_labels = None
    nli_config: dict[str, Any] = {
        "nli_model": None,
        "nli_config_sha256": None,
    }
    if nli_provider is None and (nli_model is not None or nli_responses_path is not None):
        raise RagError(
            "NLI_UNAVAILABLE",
            "--nli-model and --nli-responses require --nli-provider",
        )
    if nli_provider is not None:
        if nli_provider != "mock":
            raise RagError(
                "PROVIDER_UNSUPPORTED",
                "the deterministic evidence path supports --nli-provider mock only",
            )
        if nli_model is None or nli_responses_path is None:
            raise RagError(
                "MISSING_ARTIFACT",
                "--nli-model and --nli-responses are required when --nli-provider mock",
            )
        nli_labels = load_mock_nli_responses(nli_responses_path)
        identity = _mock_nli_identity(nli_model)
        nli_config = {
            "nli_model": identity,
            "nli_config_sha256": f"sha256:{sha256_hex(canonical_json(identity))}",
        }

    return _write_rag_artifact(
        report=report,
        cases=cases,
        output_path=output_path,
        nli_labels=nli_labels,
        nli_config=nli_config,
        cost_usd_total=0.0,
        missing_label_code="MOCK_RESPONSE_MISSING",
    )
