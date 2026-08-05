"""Phase 6 RAG evidence artifact tests."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalharness.cli import app
from evalharness.rag import RagError, build_rag_evidence
from evalharness.rag.claims import split_claims
from evalharness.rag.context import context_precision_recall
from evalharness.rag.text import CLAIM_TEXT_LIMIT

ROOT = Path(__file__).parents[2]
RAG = ROOT / "fixtures" / "rag"
runner = CliRunner()


def test_rag_evidence_nli_unavailable_by_default(tmp_path: Path) -> None:
    output = tmp_path / "rag_evidence.json"
    artifact = build_rag_evidence(
        report_path=RAG / "report.json",
        evidence_path=RAG / "evidence.jsonl",
        output_path=output,
    )

    assert artifact["gating_allowed"] is False
    assert artifact["faithfulness"]["status"] == "unavailable"
    assert artifact["faithfulness"]["reason"] == "NLI_UNAVAILABLE"
    assert artifact["retrieval"]["status"] == "ok"
    assert artifact["retrieval"]["aggregate"]["value"] == 0.42
    assert artifact["context"]["precision"]["status"] == "ok"
    assert artifact["citations"]["attribution"]["status"] == "ok"
    assert artifact["deferred"]["answer_grounded_context"]["status"] == "deferred"
    assert artifact["deferred"]["nli_verified_citations"]["reason"] == "ADR_004_RAG_METHODS"


def test_rag_evidence_with_mock_nli_separates_failures(tmp_path: Path) -> None:
    output = tmp_path / "rag_evidence.json"
    artifact = build_rag_evidence(
        report_path=RAG / "report.json",
        evidence_path=RAG / "evidence.jsonl",
        output_path=output,
        nli_provider="mock",
        nli_model="mock-nli",
        nli_responses_path=RAG / "mock-nli-responses.jsonl",
    )

    assert artifact["faithfulness"]["status"] == "ok"
    aggregate = artifact["faithfulness"]["aggregate"]
    assert aggregate["unsupported_claim_rate"] == pytest.approx(2 / 3)
    assert aggregate["n"] == 3
    examples = {row["case_id"]: row for row in artifact["faithfulness"]["examples"]}
    claim0 = examples["case-00001"]["claims"][0]
    assert claim0["supported"] is True
    claim_bad = examples["case-00002"]["claims"][0]
    assert claim_bad["supported"] is False
    missing = artifact["citations"]["missing_support_examples"]
    assert any(row["doc_id"] == "d9" for row in missing)
    report = json.loads((RAG / "report.json").read_text(encoding="utf-8"))
    expected = next(
        row for row in report["metric_aggregates"] if row["metric"] == "retrieval_ndcg_10"
    )
    assert artifact["retrieval"]["aggregate"]["value"] == expected["value"]
    assert artifact["retrieval"]["aggregate"]["n"] == expected["n"]


def test_rag_retrieval_missing_while_faithfulness_ok(tmp_path: Path) -> None:
    """Retrieval QRELS_MISSING must not poison an otherwise-ok faithfulness section."""
    report = json.loads((RAG / "report.json").read_text(encoding="utf-8"))
    report["metric_aggregates"] = [
        row for row in report["metric_aggregates"] if row.get("metric") != "retrieval_ndcg_10"
    ]
    report_path = tmp_path / "report-no-retrieval.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    artifact = build_rag_evidence(
        report_path=report_path,
        evidence_path=RAG / "evidence.jsonl",
        output_path=tmp_path / "rag_evidence.json",
        nli_provider="mock",
        nli_model="mock-nli",
        nli_responses_path=RAG / "mock-nli-responses.jsonl",
    )

    assert artifact["retrieval"]["status"] == "missing"
    assert artifact["retrieval"]["reason"] == "QRELS_MISSING"
    assert artifact["retrieval"]["aggregate"] == {
        "value": None,
        "n": 0,
        "ci_low": None,
        "ci_high": None,
    }
    assert artifact["faithfulness"]["status"] == "ok"
    assert artifact["faithfulness"]["aggregate"]["unsupported_claim_rate"] == pytest.approx(2 / 3)
    assert artifact["faithfulness"]["aggregate"]["n"] == 3
    assert artifact["gating_allowed"] is False


def test_rag_context_qrels_missing_when_no_case_supplies_qrels(tmp_path: Path) -> None:
    """Omit-key (not empty ``{}``) must make context unavailable.

    Smell: ``test_context_empty_qrels_is_available_with_zero_values`` still locks
    ``qrels: {}`` as ok zeros. Contract intent is absent qrels ⇒ unavailable; do not
    strengthen the empty-dict path until product decides.
    """
    evidence_lines = []
    for line in (RAG / "evidence.jsonl").read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        case.pop("qrels", None)
        evidence_lines.append(json.dumps(case))
    evidence_path = tmp_path / "evidence-no-qrels.jsonl"
    evidence_path.write_text("\n".join(evidence_lines) + "\n", encoding="utf-8")

    artifact = build_rag_evidence(
        report_path=RAG / "report.json",
        evidence_path=evidence_path,
        output_path=tmp_path / "rag_evidence.json",
        nli_provider="mock",
        nli_model="mock-nli",
        nli_responses_path=RAG / "mock-nli-responses.jsonl",
    )

    assert artifact["context"]["precision"]["status"] == "unavailable"
    assert artifact["context"]["precision"]["reason"] == "QRELS_MISSING"
    assert artifact["context"]["recall"]["status"] == "unavailable"
    assert artifact["context"]["recall"]["reason"] == "QRELS_MISSING"
    assert artifact["faithfulness"]["status"] == "ok"
    assert artifact["retrieval"]["status"] == "ok"


def test_published_rag_text_is_bounded_and_omits_full_source_docs(tmp_path: Path) -> None:
    long_claim = ("word " * 80).strip() + "."
    long_context = ("source " * 80).strip() + "."
    evidence_path = tmp_path / "evidence-long.jsonl"
    evidence_path.write_text(
        json.dumps(
            {
                "case_id": "case-long",
                "generation_id": "gen-long",
                "answer_text": long_claim,
                "retrieved_contexts": [
                    {"doc_id": "d1", "text": long_context, "rank": 1},
                ],
                "qrels": {"d1": 1},
                "citations": [{"claim_index": 0, "doc_id": "d1"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    nli_path = tmp_path / "nli-long.jsonl"
    nli_path.write_text(
        json.dumps(
            {
                "case_id": "case-long",
                "claim_index": 0,
                "doc_id": "d1",
                "label": "entailment",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    artifact = build_rag_evidence(
        report_path=RAG / "report.json",
        evidence_path=evidence_path,
        output_path=tmp_path / "rag_evidence.json",
        nli_provider="mock",
        nli_model="mock-nli",
        nli_responses_path=nli_path,
    )
    serialized = json.dumps(artifact)

    assert "full_source_document" not in serialized
    assert long_context not in serialized
    assert long_claim not in serialized
    claim_text = artifact["faithfulness"]["examples"][0]["claims"][0]["text"]
    assert len(claim_text) <= CLAIM_TEXT_LIMIT + 1
    assert claim_text.endswith("…")
    span_text = artifact["faithfulness"]["examples"][0]["claims"][0]["evidence_spans"][0]["text"]
    assert len(span_text) <= CLAIM_TEXT_LIMIT + 1
    assert span_text.endswith("…")
    answer = artifact["bounded_examples"][0]["answer_text"]
    assert len(answer) <= CLAIM_TEXT_LIMIT + 1
    assert answer.endswith("…")


def test_nli_mock_response_missing_on_incomplete_fixture(tmp_path: Path) -> None:
    incomplete_lines: list[str] = []
    dropped = False
    for line in (RAG / "mock-nli-responses.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            row.get("case_id") == "case-00002"
            and row.get("claim_index") == 0
            and row.get("doc_id") == "d3"
        ):
            dropped = True
            continue
        incomplete_lines.append(line)
    assert dropped
    nli_path = tmp_path / "nli-incomplete.jsonl"
    nli_path.write_text("\n".join(incomplete_lines) + "\n", encoding="utf-8")

    with pytest.raises(RagError) as exc:
        build_rag_evidence(
            report_path=RAG / "report.json",
            evidence_path=RAG / "evidence.jsonl",
            output_path=tmp_path / "rag_evidence.json",
            nli_provider="mock",
            nli_model="mock-nli",
            nli_responses_path=nli_path,
        )

    assert exc.value.code == "MOCK_RESPONSE_MISSING"


def test_claim_split_respects_abbreviations_and_bounds() -> None:
    claims, error = split_claims("Dr. Smith met Mr. Jones. They agreed. e.g. this holds.")
    assert error is None
    assert claims[0].startswith("Dr. Smith")
    long = "x" * 500
    truncated, _ = split_claims(long)
    assert len(truncated[0]) <= CLAIM_TEXT_LIMIT + 1


def test_context_recall_counts_each_relevant_document_once() -> None:
    context = context_precision_recall(
        [
            {
                "retrieved_contexts": [{"doc_id": "d1"}, {"doc_id": "d1"}],
                "qrels": {"d1": 1},
            }
        ]
    )

    assert context["recall"] == {"status": "ok", "value": 1.0, "n": 1}


def test_context_empty_qrels_is_available_with_zero_values() -> None:
    context = context_precision_recall([{"retrieved_contexts": [], "qrels": {}}])

    assert context["precision"] == {"status": "ok", "value": 0.0, "n": 1}
    assert context["recall"] == {"status": "ok", "value": 0.0, "n": 1}


def test_rag_rejects_nli_configuration_without_provider(tmp_path: Path) -> None:
    with pytest.raises(RagError) as exc:
        build_rag_evidence(
            report_path=RAG / "report.json",
            evidence_path=RAG / "evidence.jsonl",
            output_path=tmp_path / "rag_evidence.json",
            nli_model="mock-nli",
            nli_responses_path=RAG / "mock-nli-responses.jsonl",
        )

    assert exc.value.code == "NLI_UNAVAILABLE"


def test_cli_rag_evidence(tmp_path: Path) -> None:
    output = tmp_path / "rag_evidence.json"
    result = runner.invoke(
        app,
        [
            "rag",
            "evidence",
            "--report",
            str(RAG / "report.json"),
            "--evidence",
            str(RAG / "evidence.jsonl"),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["faithfulness_status"] == "unavailable"
    assert payload["gating_allowed"] is False


def test_rag_package_has_no_hf_or_store_imports() -> None:
    rag_root = ROOT / "src" / "evalharness" / "rag"
    forbidden = ("evalharness.store", "huggingface_hub", "datasets", "transformers", "torch")
    for path in rag_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                assert not any(name == item or name.startswith(item + ".") for item in forbidden)
