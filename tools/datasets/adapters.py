"""Pure transforms from pinned local snapshots to harness case dictionaries."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from evalharness.datasets.validator import INPUT_TEXT_LIMIT, REFERENCE_TEXT_LIMIT

CaseRecord = dict[str, Any]
Parser = Callable[[Path], list[CaseRecord]]


@dataclass(frozen=True, slots=True)
class AdapterSpec:
    """Static policy and transform metadata for an offline source adapter."""

    name: str
    version: str
    source_id: str
    source_revision: str
    dataset_name: str
    dataset_version: str
    license: str
    redistributable_smoke: bool
    attribution: str
    pii_scrubbed: bool
    contamination_risk: str
    pii_scrub_procedure: str
    task_metrics: tuple[str, ...]
    slices: tuple[str, ...]
    created_at: str
    parser: Parser
    requires_external_pin: bool = True
    canonical_url: str | None = None


def _text_within(value: Any, limit: int) -> bool:
    if isinstance(value, str):
        return len(value) <= limit
    if isinstance(value, dict):
        return all(_text_within(child, limit) for child in cast(dict[str, Any], value).values())
    if isinstance(value, list):
        return all(_text_within(child, limit) for child in value)
    return True


def fits_field_bounds(record: CaseRecord) -> bool:
    """True when a source record fits the published input and reference bounds.

    Source documents that exceed the bounds are dropped from the sampling pool rather
    than published and rejected downstream: a pinned corpus of long articles should
    sample its publishable records, not abort the whole materialization.
    """
    if not _text_within(record.get("inputs", {}), INPUT_TEXT_LIMIT):
        return False
    references = [
        record.get("reference_answer"),
        record.get("expected_label"),
        record.get("references", []),
    ]
    return all(_text_within(value, REFERENCE_TEXT_LIMIT) for value in references)


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[CaseRecord]:
    records: list[CaseRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} must be an object")
        records.append(cast(CaseRecord, value))
    return records


def _squad(path: Path) -> list[CaseRecord]:
    root = _read_json(path)
    if not isinstance(root, dict) or not isinstance(root.get("data"), list):
        raise ValueError("SQuAD source must contain a data array")
    records: list[CaseRecord] = []
    for article in root["data"]:
        if not isinstance(article, dict):
            continue
        title = str(article.get("title", ""))
        for paragraph in article.get("paragraphs", []):
            if not isinstance(paragraph, dict):
                continue
            context = str(paragraph.get("context", ""))
            for qa in paragraph.get("qas", []):
                if not isinstance(qa, dict):
                    continue
                answers = qa.get("answers", [])
                if not answers or not isinstance(answers[0], dict):
                    continue
                records.append(
                    {
                        "id": str(qa["id"]),
                        "task_type": "qa_short",
                        "inputs": {
                            "question": str(qa["question"]),
                            "context": context,
                        },
                        "reference_answer": str(answers[0]["text"]),
                        "references": [
                            str(answer["text"])
                            for answer in answers
                            if isinstance(answer, dict) and "text" in answer
                        ],
                        "slices": {"domain": "wikipedia", "source_title": title[:100]},
                    }
                )
    return records


def _ag_news(path: Path) -> list[CaseRecord]:
    labels = {"1": "world", "2": "sports", "3": "business", "4": "science_technology"}
    records: list[CaseRecord] = []
    with path.open(encoding="utf-8", newline="") as source_file:
        for index, row in enumerate(csv.reader(source_file)):
            if len(row) != 3 or row[0] not in labels:
                raise ValueError(f"AG News row {index + 1} must contain label,title,description")
            records.append(
                {
                    "id": f"ag-news-{index:07d}",
                    "task_type": "classification",
                    "inputs": {
                        "text": f"{row[1].strip()}\n{row[2].strip()}",
                        "labels": list(labels.values()),
                    },
                    "expected_label": labels[row[0]],
                    "slices": {"domain": "news"},
                }
            )
    return records


def _pubmedqa(path: Path) -> list[CaseRecord]:
    root = _read_json(path)
    if not isinstance(root, dict):
        raise ValueError("PubMedQA source must be an object keyed by record id")
    records: list[CaseRecord] = []
    for record_id, value in root.items():
        if not isinstance(value, dict):
            continue
        context_value = value.get("CONTEXT", [])
        context = (
            "\n".join(map(str, context_value))
            if isinstance(context_value, list)
            else str(context_value)
        )
        records.append(
            {
                "id": str(record_id),
                "task_type": "classification",
                "inputs": {
                    "question": str(value.get("QUESTION", "")),
                    "context": context,
                    "labels": ["yes", "no", "maybe"],
                },
                "expected_label": str(value.get("final_decision", "")).lower(),
                "slices": {"domain": "healthcare"},
            }
        )
    return records


def _phrasebank(path: Path) -> list[CaseRecord]:
    records: list[CaseRecord] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            sentence, label = line.rsplit("@", 1)
        except ValueError as exc:
            raise ValueError(f"PhraseBank line {index + 1} must use sentence@label") from exc
        records.append(
            {
                "id": f"phrasebank-{index:07d}",
                "task_type": "classification",
                "inputs": {
                    "text": sentence,
                    "labels": ["negative", "neutral", "positive"],
                },
                "expected_label": label.strip(),
                "slices": {"domain": "finance"},
            }
        )
    return records


def _finqa(path: Path) -> list[CaseRecord]:
    root = _read_json(path)
    if not isinstance(root, list):
        raise ValueError("FinQA source must be an array")
    records: list[CaseRecord] = []
    for index, item in enumerate(root):
        if not isinstance(item, dict):
            continue
        qa = item.get("qa")
        if not isinstance(qa, dict):
            continue
        record_id = str(item.get("id", index))
        records.append(
            {
                "id": record_id,
                "task_type": "qa_short",
                "inputs": {
                    "question": str(qa.get("question", "")),
                    "context": "\n".join(
                        map(str, [*item.get("pre_text", []), *item.get("post_text", [])])
                    ),
                    "table": item.get("table", []),
                },
                "reference_answer": str(qa.get("answer", "")),
                "slices": {"domain": "finance"},
            }
        )
    return records


def _summaries(path: Path) -> list[CaseRecord]:
    records: list[CaseRecord] = []
    for index, row in enumerate(_read_jsonl(path)):
        document = row.get("document", row.get("article", ""))
        summary = row.get("summary", row.get("highlights", ""))
        records.append(
            {
                "id": str(row.get("id", f"summary-{index:07d}")),
                "task_type": "summarization",
                "inputs": {"document": str(document)},
                "reference_answer": str(summary),
                "slices": {"domain": str(row.get("domain", "news"))},
            }
        )
    return records


def _synthetic(path: Path) -> list[CaseRecord]:
    return _read_jsonl(path)


def _spec(
    *,
    name: str,
    source_id: str,
    dataset_name: str,
    license_id: str,
    redistributable: bool,
    risk: str,
    metrics: tuple[str, ...],
    parser: Parser,
    pii_scrubbed: bool = False,
    slices: tuple[str, ...] = ("domain",),
    revision: str = "operator-pinned",
    attribution: str = "",
    requires_external_pin: bool = True,
    canonical_url: str | None = None,
) -> AdapterSpec:
    return AdapterSpec(
        name=name,
        version="1.0.0",
        source_id=source_id,
        source_revision=revision,
        dataset_name=dataset_name,
        dataset_version="1.0.0",
        license=license_id,
        redistributable_smoke=redistributable,
        attribution=attribution,
        pii_scrubbed=pii_scrubbed,
        contamination_risk=risk,
        pii_scrub_procedure="docs/datasets.md#privacy-and-pii",
        task_metrics=metrics,
        slices=slices,
        created_at="2026-08-05",
        parser=parser,
        requires_external_pin=requires_external_pin,
        canonical_url=canonical_url,
    )


ADAPTERS: dict[str, AdapterSpec] = {
    "squad_v1_1": _spec(
        name="squad_v1_1",
        source_id="squad_v1.1",
        dataset_name="squad-v1.1",
        license_id="unknown",
        redistributable=False,
        risk="high",
        metrics=("squad_f1", "exact_match"),
        parser=_squad,
        revision="dev-v1.1",
        canonical_url="https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
    ),
    "ag_news": _spec(
        name="ag_news",
        source_id="ag_news",
        dataset_name="ag-news",
        license_id="unknown",
        redistributable=False,
        risk="high",
        metrics=("classification",),
        parser=_ag_news,
    ),
    "pubmedqa": _spec(
        name="pubmedqa",
        source_id="pubmedqa",
        dataset_name="pubmedqa",
        license_id="unknown",
        redistributable=False,
        risk="medium",
        metrics=("classification",),
        parser=_pubmedqa,
    ),
    "financial_phrasebank": _spec(
        name="financial_phrasebank",
        source_id="financial_phrasebank",
        dataset_name="financial-phrasebank",
        license_id="CC-BY-NC-SA-3.0",
        redistributable=False,
        risk="high",
        metrics=("classification",),
        parser=_phrasebank,
    ),
    "finqa": _spec(
        name="finqa",
        source_id="finqa",
        dataset_name="finqa",
        license_id="unknown",
        redistributable=False,
        risk="high",
        metrics=("numeric_assertion",),
        parser=_finqa,
    ),
    "cnn_dailymail": _spec(
        name="cnn_dailymail",
        source_id="cnn_dailymail",
        dataset_name="cnn-dailymail",
        license_id="unknown",
        redistributable=False,
        risk="high",
        metrics=("rouge_l", "chrf_pp"),
        parser=_summaries,
    ),
    "xsum": _spec(
        name="xsum",
        source_id="xsum",
        dataset_name="xsum",
        license_id="unknown",
        redistributable=False,
        risk="high",
        metrics=("rouge_l", "chrf_pp"),
        parser=_summaries,
    ),
    "scifact": _spec(
        name="scifact",
        source_id="scifact",
        dataset_name="scifact",
        license_id="unknown",
        redistributable=False,
        risk="medium",
        metrics=("retrieval_ndcg_10",),
        parser=_synthetic,
    ),
    "docred": _spec(
        name="docred",
        source_id="docred",
        dataset_name="docred",
        license_id="unknown",
        redistributable=False,
        risk="high",
        metrics=("json_validity", "json_field_f1"),
        parser=_synthetic,
    ),
}


def synthetic_spec(name: str, metrics: tuple[str, ...]) -> AdapterSpec:
    """Build metadata for a repository-authored synthetic smoke adapter."""
    return _spec(
        name=name,
        source_id=name,
        dataset_name=name.replace("_", "-"),
        license_id="CC0-1.0",
        redistributable=True,
        risk="low",
        metrics=metrics,
        parser=_synthetic,
        pii_scrubbed=True,
        revision="synthetic-v1",
        attribution="Repository-authored synthetic fixture; no public-source attribution.",
        requires_external_pin=False,
    )


ADAPTERS.update(
    {
        "synthetic_qa": synthetic_spec("synthetic_qa", ("squad_f1", "exact_match")),
        "synthetic_news": synthetic_spec("synthetic_news", ("classification",)),
        "synthetic_healthcare": synthetic_spec("synthetic_healthcare", ("classification",)),
        "synthetic_finance": synthetic_spec("synthetic_finance", ("numeric_assertion",)),
        "synthetic_summarization": synthetic_spec(
            "synthetic_summarization", ("rouge_l", "chrf_pp")
        ),
        "synthetic_extraction": synthetic_spec(
            "synthetic_extraction", ("json_validity", "json_field_f1")
        ),
        "synthetic_retrieval": synthetic_spec("synthetic_retrieval", ("retrieval_ndcg_10",)),
        "synthetic_math": synthetic_spec("synthetic_math", ("numeric_assertion",)),
    }
)
