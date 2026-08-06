"""Judge: rubrics, pairwise swap, calibration gate, truncation."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalharness.cli import app
from evalharness.config import get_settings
from evalharness.judge import JudgeError, attach_calibration, run_judgment, validate_calibration
from evalharness.judge.agreement import compute_agreement
from evalharness.judge.models import AgreementMetric, JudgeMode
from evalharness.judge.pairwise import resolve_original_preference
from evalharness.judge.text import EVIDENCE_TEXT_LIMIT, REASONING_LIMIT

ROOT = Path(__file__).parents[2]
JUDGE = ROOT / "fixtures" / "judge"
runner = CliRunner()


def test_pointwise_run_is_informational_and_truncates_reasoning(tmp_path: Path) -> None:
    output = tmp_path / "judgment.json"
    artifact = run_judgment(
        mode=JudgeMode.POINTWISE,
        rubric_path=JUDGE / "rubric-pointwise.yaml",
        candidates_path=JUDGE / "candidates-pointwise.jsonl",
        pairs_path=None,
        provider="mock",
        model="mock-judge",
        judge_family="qwen",
        candidate_family="llama",
        responses_path=JUDGE / "mock-judge-responses-pointwise.jsonl",
        seed=42,
        output_path=output,
    )

    assert artifact.gating_allowed is False
    assert artifact.calibration_digest is None
    assert artifact.judge_model.resolved_version.startswith("sha256:")
    assert artifact.cost_usd_total == 0.0
    assert "p50" in artifact.latency_ms.model_dump()
    source_reasoning = json.loads(
        (JUDGE / "mock-judge-responses-pointwise.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )["reasoning"]
    assert len(source_reasoning) > REASONING_LIMIT
    long_reasoning = artifact.items[0]["reasoning"]
    assert isinstance(long_reasoning, str)
    assert len(long_reasoning) == REASONING_LIMIT
    assert long_reasoning.endswith("…")
    assert long_reasoning[:-1] == " ".join(source_reasoning.split())[: REASONING_LIMIT - 1]


def test_pointwise_evidence_candidate_text_is_bounded_to_the_evidence_limit(
    tmp_path: Path,
) -> None:
    """Published evidence inherits the 280-char cap, ellipsis included."""
    overlong = "candidate " * 60
    candidates_path = tmp_path / "candidates-long.jsonl"
    candidates_path.write_text(
        json.dumps(
            {
                "case_id": "case-00001",
                "generation_id": "gen-00001",
                "candidate_text": overlong,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    artifact = run_judgment(
        mode=JudgeMode.POINTWISE,
        rubric_path=JUDGE / "rubric-pointwise.yaml",
        candidates_path=candidates_path,
        pairs_path=None,
        provider="mock",
        model="mock-judge",
        judge_family="qwen",
        candidate_family="llama",
        responses_path=JUDGE / "mock-judge-responses-pointwise.jsonl",
        seed=42,
        output_path=tmp_path / "judgment-long-evidence.json",
    )

    evidence_text = artifact.items[0]["evidence"]["candidate_text"]
    assert len(evidence_text) == EVIDENCE_TEXT_LIMIT
    assert evidence_text.endswith("…")
    assert evidence_text[:-1] == " ".join(overlong.split())[: EVIDENCE_TEXT_LIMIT - 1]
    assert overlong.strip() not in json.dumps(artifact.model_dump(mode="json"))


def test_pointwise_run_rejects_score_without_required_reasoning(tmp_path: Path) -> None:
    responses = []
    for index, line in enumerate(
        (JUDGE / "mock-judge-responses-pointwise.jsonl").read_text(encoding="utf-8").splitlines()
    ):
        response = json.loads(line)
        if index == 0:
            response["reasoning"] = " "
        responses.append(json.dumps(response))
    responses_path = tmp_path / "responses.jsonl"
    responses_path.write_text("\n".join(responses) + "\n", encoding="utf-8")

    with pytest.raises(JudgeError) as exc:
        run_judgment(
            mode=JudgeMode.POINTWISE,
            rubric_path=JUDGE / "rubric-pointwise.yaml",
            candidates_path=JUDGE / "candidates-pointwise.jsonl",
            pairs_path=None,
            provider="mock",
            model="mock-judge",
            judge_family="qwen",
            candidate_family="llama",
            responses_path=responses_path,
            seed=42,
            output_path=tmp_path / "judgment.json",
        )

    assert exc.value.code == "INVALID_MOCK_RESPONSES"
    assert "missing required reasoning" in str(exc.value)


def test_pairwise_swap_consistency_and_flip_becomes_tie(tmp_path: Path) -> None:
    output = tmp_path / "judgment-pairwise.json"
    artifact = run_judgment(
        mode=JudgeMode.PAIRWISE,
        rubric_path=JUDGE / "rubric-pairwise.yaml",
        candidates_path=None,
        pairs_path=JUDGE / "pairs.jsonl",
        provider="mock",
        model="mock-judge",
        judge_family="qwen",
        candidate_family="llama",
        responses_path=JUDGE / "mock-judge-responses-pairwise.jsonl",
        seed=42,
        output_path=output,
    )

    assert artifact.gating_allowed is False
    assert artifact.pairwise_summary is not None
    items = {item["case_id"]: item for item in artifact.items}
    assert items["pair-00001"]["consistent"] is True
    assert items["pair-00001"]["final_preference"] == "A"
    assert items["pair-flip"]["consistent"] is False
    assert items["pair-flip"]["final_preference"] == "tie"
    assert len(items["pair-00001"]["orderings"]) == 2
    # Fixture: 2/3 pairs consistent; 4 of 6 non-tie orderings prefer first-shown "A".
    assert artifact.pairwise_summary.swap_consistency == pytest.approx(2 / 3)
    assert artifact.pairwise_summary.position_bias == pytest.approx(2 / 3)
    graph = artifact.pairwise_summary.bradley_terry
    assert graph is not None
    # A connected graph yields raw win rates; Phase 6 never fits BT strengths.
    assert graph.status == "win_rates_only"
    assert graph.win_rates == {  # type: ignore[union-attr]
        "llama3.2:3b": pytest.approx(1.0),
        "mistral:7b": pytest.approx(0.0),
        "qwen2.5:3b": pytest.approx(0.0),
    }


def test_pairwise_self_pair_hard_fails(tmp_path: Path) -> None:
    pairs_path = tmp_path / "self-pair.jsonl"
    pairs_path.write_text(
        json.dumps(
            {
                "case_id": "pair-self",
                "a_generation_id": "gen-A",
                "b_generation_id": "gen-B",
                "a_model_label": "llama3.2:3b",
                "b_model_label": "llama3.2:3b",
                "a_text": "Same model left.",
                "b_text": "Same model right.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    responses_path = tmp_path / "responses.jsonl"
    responses_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "case_id": "pair-self",
                        "swap_position": 0,
                        "preference": "A",
                        "reasoning": "unused",
                    }
                ),
                json.dumps(
                    {
                        "case_id": "pair-self",
                        "swap_position": 1,
                        "preference": "B",
                        "reasoning": "unused",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(JudgeError) as exc:
        run_judgment(
            mode=JudgeMode.PAIRWISE,
            rubric_path=JUDGE / "rubric-pairwise.yaml",
            candidates_path=None,
            pairs_path=pairs_path,
            provider="mock",
            model="mock-judge",
            judge_family="qwen",
            candidate_family="llama",
            responses_path=responses_path,
            seed=42,
            output_path=tmp_path / "judgment.json",
        )

    assert exc.value.code == "SELF_PAIR"


def test_pointwise_mock_response_missing_when_fixture_key_absent(tmp_path: Path) -> None:
    responses = (JUDGE / "mock-judge-responses-pointwise.jsonl").read_text(encoding="utf-8")
    lines = [line for line in responses.splitlines() if line.strip()]
    assert len(lines) >= 2
    responses_path = tmp_path / "responses-incomplete.jsonl"
    responses_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(JudgeError) as exc:
        run_judgment(
            mode=JudgeMode.POINTWISE,
            rubric_path=JUDGE / "rubric-pointwise.yaml",
            candidates_path=JUDGE / "candidates-pointwise.jsonl",
            pairs_path=None,
            provider="mock",
            model="mock-judge",
            judge_family="qwen",
            candidate_family="llama",
            responses_path=responses_path,
            seed=42,
            output_path=tmp_path / "judgment.json",
        )

    assert exc.value.code == "MOCK_RESPONSE_MISSING"


@pytest.mark.parametrize(
    ("judge_family", "candidate_family"),
    [
        (" ", "llama"),
        ("qwen", "\t"),
        ("", "llama"),
        ("qwen", ""),
    ],
)
def test_empty_or_whitespace_family_fail_closed_at_run(
    tmp_path: Path,
    judge_family: str,
    candidate_family: str,
) -> None:
    with pytest.raises(JudgeError) as exc:
        run_judgment(
            mode=JudgeMode.POINTWISE,
            rubric_path=JUDGE / "rubric-pointwise.yaml",
            candidates_path=JUDGE / "candidates-pointwise.jsonl",
            pairs_path=None,
            provider="mock",
            model="mock-judge",
            judge_family=judge_family,
            candidate_family=candidate_family,
            responses_path=JUDGE / "mock-judge-responses-pointwise.jsonl",
            seed=42,
            output_path=tmp_path / "judgment.json",
        )

    assert exc.value.code == "JUDGE_FAMILY_CONFLICT"


def test_bradley_terry_refuses_disconnected_graph(tmp_path: Path) -> None:
    output = tmp_path / "judgment-disc.json"
    artifact = run_judgment(
        mode=JudgeMode.PAIRWISE,
        rubric_path=JUDGE / "rubric-pairwise.yaml",
        candidates_path=None,
        pairs_path=JUDGE / "pairs-disconnected.jsonl",
        provider="mock",
        model="mock-judge",
        judge_family="qwen",
        candidate_family="llama",
        responses_path=JUDGE / "mock-judge-responses-disconnected.jsonl",
        seed=7,
        output_path=output,
    )

    bt = artifact.pairwise_summary.bradley_terry  # type: ignore[union-attr]
    assert bt is not None
    assert bt.status == "refused"
    assert bt.reason == "DISCONNECTED_PAIRWISE_GRAPH"  # type: ignore[union-attr]
    assert bt.n_models == 4
    assert len(bt.component_sizes) >= 2


def test_resolve_original_preference_maps_swapped_positions() -> None:
    assert resolve_original_preference(swap_position=0, preference="A") == "A"
    assert resolve_original_preference(swap_position=1, preference="A") == "B"
    assert resolve_original_preference(swap_position=1, preference="B") == "A"


def _calibration_judgment(tmp_path: Path, *, judge_family: str, candidate_family: str) -> Path:
    output = tmp_path / "judgment-cal.json"
    run_judgment(
        mode=JudgeMode.POINTWISE,
        rubric_path=JUDGE / "rubric-pointwise.yaml",
        candidates_path=JUDGE / "candidates-calibration.jsonl",
        pairs_path=None,
        provider="mock",
        model="mock-judge",
        judge_family=judge_family,
        candidate_family=candidate_family,
        responses_path=JUDGE / "mock-judge-responses-calibration.jsonl",
        seed=1,
        output_path=output,
    )
    return output


def test_calibration_gate_true_only_on_passing_holdout(tmp_path: Path) -> None:
    judgment = _calibration_judgment(tmp_path, judge_family="qwen", candidate_family="llama")
    calibration_path = tmp_path / "calibration.json"

    artifact = validate_calibration(
        judgment_path=judgment,
        labels_dev_path=JUDGE / "labels-dev.jsonl",
        labels_holdout_path=JUDGE / "labels-holdout.jsonl",
        rubric_path=JUDGE / "rubric-pointwise.yaml",
        output_path=calibration_path,
    )

    assert artifact.gating_allowed is True
    assert artifact.family_separation_ok is True
    assert artifact.holdout.n == 150
    assert artifact.dev is not None
    assert artifact.dev.n == 50
    assert artifact.min_dev_n == 50
    assert artifact.holdout.label_set_id != artifact.dev.label_set_id
    assert artifact.holdout.agreement is not None
    assert artifact.holdout.agreement >= 0.60
    # Dev agreement must never be the gate source; presence only.
    assert artifact.dev.agreement is not None

    calibrated = tmp_path / "judgment-calibrated.json"
    attached = attach_calibration(
        judgment_path=judgment,
        calibration_path=calibration_path,
        output_path=calibrated,
    )
    assert attached.gating_allowed is True
    assert attached.calibration_digest == artifact.calibration_digest

    raw = json.loads(judgment.read_text(encoding="utf-8"))
    assert raw["gating_allowed"] is False


@pytest.mark.parametrize(
    ("metric", "human", "judge"),
    [
        (AgreementMetric.COHEN_KAPPA, [5, 5, 5], [5, 5, 5]),
        (AgreementMetric.SPEARMAN, [4.0, 4.0, 4.0], [1.0, 2.0, 3.0]),
        (AgreementMetric.SPEARMAN, [1.0, 2.0, 3.0], [4.0, 4.0, 4.0]),
        (AgreementMetric.KRIPPENDORFF_ALPHA, ["yes", "yes"], ["yes", "yes"]),
    ],
)
def test_compute_agreement_is_none_when_undefined(
    metric: AgreementMetric,
    human: list[int | float | str],
    judge: list[int | float | str],
) -> None:
    """A degenerate single-category holdout has no chance term, so agreement is undefined."""
    assert compute_agreement(metric, human, judge) is None


def test_calibration_gate_false_when_holdout_is_a_single_category(tmp_path: Path) -> None:
    """Perfect but degenerate holdout agreement must not clear the gate.

    Every human label and every judge score is 5, so Cohen's kappa is undefined.
    Scoring that as 1.0 would clear the 0.60 threshold on zero evidence.
    """
    judgment = _calibration_judgment(tmp_path, judge_family="qwen", candidate_family="llama")
    payload = json.loads(judgment.read_text(encoding="utf-8"))
    for item in payload["items"]:
        if str(item["case_id"]).startswith("holdout-"):
            item["score"] = 5
    judgment.write_text(json.dumps(payload), encoding="utf-8")

    holdout = tmp_path / "labels-holdout-identical.jsonl"
    holdout.write_text(
        "".join(
            json.dumps({**json.loads(line), "value": 5}) + "\n"
            for line in (JUDGE / "labels-holdout.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ),
        encoding="utf-8",
    )

    artifact = validate_calibration(
        judgment_path=judgment,
        labels_dev_path=JUDGE / "labels-dev.jsonl",
        labels_holdout_path=holdout,
        rubric_path=JUDGE / "rubric-pointwise.yaml",
        output_path=tmp_path / "calibration-degenerate.json",
    )

    assert artifact.holdout.n == 150
    assert artifact.holdout.agreement is None
    assert artifact.family_separation_ok is True
    assert artifact.gating_allowed is False
    assert any(reason.startswith("AGREEMENT_UNDEFINED") for reason in artifact.block_reasons)


def test_attach_refuses_judgment_whose_holdout_agreement_was_undefined(tmp_path: Path) -> None:
    """The undefined-agreement calibration stays unattachable, not merely unflagged."""
    judgment = _calibration_judgment(tmp_path, judge_family="qwen", candidate_family="llama")
    payload = json.loads(judgment.read_text(encoding="utf-8"))
    for item in payload["items"]:
        if str(item["case_id"]).startswith("holdout-"):
            item["score"] = 5
    judgment.write_text(json.dumps(payload), encoding="utf-8")
    holdout = tmp_path / "labels-holdout-identical.jsonl"
    holdout.write_text(
        "".join(
            json.dumps({**json.loads(line), "value": 5}) + "\n"
            for line in (JUDGE / "labels-holdout.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ),
        encoding="utf-8",
    )
    calibration_path = tmp_path / "calibration-degenerate.json"
    validate_calibration(
        judgment_path=judgment,
        labels_dev_path=JUDGE / "labels-dev.jsonl",
        labels_holdout_path=holdout,
        rubric_path=JUDGE / "rubric-pointwise.yaml",
        output_path=calibration_path,
    )

    with pytest.raises(JudgeError) as exc:
        attach_calibration(
            judgment_path=judgment,
            calibration_path=calibration_path,
            output_path=tmp_path / "judgment-degenerate.json",
        )

    assert exc.value.code == "UNCALIBRATED_JUDGE"


def test_calibration_gate_false_when_holdout_agreement_below_threshold(
    tmp_path: Path,
) -> None:
    """High-dev / low-holdout agreement must not clear the gate bit.

    If the predicate wrongly used agreement_dev, this fixture would pass.
    """
    judgment = _calibration_judgment(tmp_path, judge_family="qwen", candidate_family="llama")
    payload = json.loads(judgment.read_text(encoding="utf-8"))
    for item in payload["items"]:
        case_id = str(item["case_id"])
        if case_id.startswith("holdout-"):
            # Force systematic disagreement on holdout only; leave perfect dev matches.
            item["score"] = 1 if int(item["score"]) != 1 else 5
    judgment.write_text(json.dumps(payload), encoding="utf-8")

    artifact = validate_calibration(
        judgment_path=judgment,
        labels_dev_path=JUDGE / "labels-dev.jsonl",
        labels_holdout_path=JUDGE / "labels-holdout.jsonl",
        rubric_path=JUDGE / "rubric-pointwise.yaml",
        output_path=tmp_path / "calibration-low-holdout.json",
    )

    assert artifact.dev is not None
    assert artifact.dev.agreement is not None
    assert artifact.dev.agreement >= 0.60
    assert artifact.holdout.agreement is not None
    assert artifact.holdout.agreement < 0.60
    assert artifact.holdout.n == 150
    assert artifact.family_separation_ok is True
    assert artifact.gating_allowed is False
    assert any(
        "agreement_holdout=" in reason and "threshold=" in reason
        for reason in artifact.block_reasons
    )


def test_calibration_gate_true_when_holdout_high_even_if_dev_low(
    tmp_path: Path,
) -> None:
    """Holdout-high / dev-low still clears the gate (predicate is holdout-only)."""
    judgment = _calibration_judgment(tmp_path, judge_family="qwen", candidate_family="llama")
    payload = json.loads(judgment.read_text(encoding="utf-8"))
    for item in payload["items"]:
        case_id = str(item["case_id"])
        if case_id.startswith("dev-"):
            item["score"] = 1 if int(item["score"]) != 1 else 5
    judgment.write_text(json.dumps(payload), encoding="utf-8")

    artifact = validate_calibration(
        judgment_path=judgment,
        labels_dev_path=JUDGE / "labels-dev.jsonl",
        labels_holdout_path=JUDGE / "labels-holdout.jsonl",
        rubric_path=JUDGE / "rubric-pointwise.yaml",
        output_path=tmp_path / "calibration-holdout-only.json",
    )

    assert artifact.dev is not None
    assert artifact.dev.agreement is not None
    assert artifact.dev.agreement < 0.60
    assert artifact.holdout.agreement is not None
    assert artifact.holdout.agreement >= 0.60
    assert artifact.holdout.n == 150
    assert artifact.dev.n == 50
    assert artifact.family_separation_ok is True
    assert artifact.gating_allowed is True
    assert artifact.block_reasons == []


def test_duplicate_judgment_case_ids_are_hard_error_not_holdout_padding(tmp_path: Path) -> None:
    """Repeated judgment items for one labeled case must be refused outright.

    Without the guard, 150 identical case_id rows plus one holdout label would
    clear min_holdout_n on volume alone and open the gate.
    """
    judgment = _calibration_judgment(tmp_path, judge_family="qwen", candidate_family="llama")
    payload = json.loads(judgment.read_text(encoding="utf-8"))
    holdout_case = "holdout-dup-00001"
    template = {
        "evidence": {"candidate_text": "dup"},
        "reasoning": "Matches human label.",
        "outcome": None,
    }
    payload["items"] = [
        {**template, "case_id": holdout_case, "generation_id": f"gen-h-{index}", "score": 5}
        for index in range(150)
    ]
    judgment.write_text(json.dumps(payload), encoding="utf-8")

    labels_holdout = tmp_path / "labels-holdout-dup.jsonl"
    labels_holdout.write_text(
        json.dumps(
            {
                "schema_version": "0.1",
                "rubric_name": "helpfulness",
                "rubric_version": "1.0.0",
                "case_id": holdout_case,
                "label_shape": "ordinal_score",
                "value": 5,
                "split": "holdout",
                "label_set_id": "helpfulness-holdout-dup-v1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calibration_path = tmp_path / "calibration-dup.json"

    with pytest.raises(JudgeError) as exc:
        validate_calibration(
            judgment_path=judgment,
            labels_dev_path=JUDGE / "labels-dev.jsonl",
            labels_holdout_path=labels_holdout,
            rubric_path=JUDGE / "rubric-pointwise.yaml",
            output_path=calibration_path,
        )

    assert exc.value.code == "DUPLICATE_CASE_ID"
    assert holdout_case in str(exc.value)
    assert not calibration_path.exists()


def test_calibration_gate_false_when_dev_n_below_min_dev_n(tmp_path: Path) -> None:
    judgment = _calibration_judgment(tmp_path, judge_family="qwen", candidate_family="llama")
    slim_dev = tmp_path / "labels-dev-slim.jsonl"
    slim_dev.write_text(
        "\n".join((JUDGE / "labels-dev.jsonl").read_text(encoding="utf-8").splitlines()[:10])
        + "\n",
        encoding="utf-8",
    )

    artifact = validate_calibration(
        judgment_path=judgment,
        labels_dev_path=slim_dev,
        labels_holdout_path=JUDGE / "labels-holdout.jsonl",
        rubric_path=JUDGE / "rubric-pointwise.yaml",
        output_path=tmp_path / "calibration-min-dev.json",
    )

    assert artifact.dev is not None
    assert artifact.dev.n == 10
    assert artifact.holdout.n == 150
    assert artifact.holdout.agreement is not None
    assert artifact.holdout.agreement >= 0.60
    assert artifact.min_dev_n == 50
    assert artifact.gating_allowed is False
    assert any(
        "n_dev=10" in reason and "min_dev_n=50" in reason for reason in artifact.block_reasons
    )


def test_calibration_gate_false_when_dev_missing(tmp_path: Path) -> None:
    judgment = _calibration_judgment(tmp_path, judge_family="qwen", candidate_family="llama")
    artifact = validate_calibration(
        judgment_path=judgment,
        labels_dev_path=None,
        labels_holdout_path=JUDGE / "labels-holdout.jsonl",
        rubric_path=JUDGE / "rubric-pointwise.yaml",
        output_path=tmp_path / "calibration.json",
    )
    assert artifact.gating_allowed is False
    assert "missing --labels-dev" in artifact.block_reasons


@pytest.mark.parametrize(
    ("judge_family", "candidate_family"),
    [
        ("llama", "llama"),
        ("Qwen", "qwen"),
        ("QWEN", "qwen"),
        ("llama", "LLaMA"),
        (" Llama ", "llama"),
    ],
)
def test_calibration_gate_false_on_family_conflict(
    tmp_path: Path,
    judge_family: str,
    candidate_family: str,
) -> None:
    """Family separation is case- and whitespace-insensitive, and fails closed.

    Dropping the normalization would let ``Qwen`` judge ``qwen`` clear the gate.
    """
    judgment = _calibration_judgment(
        tmp_path,
        judge_family=judge_family,
        candidate_family=candidate_family,
    )
    artifact = validate_calibration(
        judgment_path=judgment,
        labels_dev_path=JUDGE / "labels-dev.jsonl",
        labels_holdout_path=JUDGE / "labels-holdout.jsonl",
        rubric_path=JUDGE / "rubric-pointwise.yaml",
        output_path=tmp_path / "calibration.json",
    )
    assert artifact.family_separation_ok is False
    assert "JUDGE_FAMILY_CONFLICT" in artifact.block_reasons
    assert artifact.gating_allowed is False


def test_calibration_gate_false_below_holdout_n(tmp_path: Path) -> None:
    judgment = _calibration_judgment(tmp_path, judge_family="qwen", candidate_family="llama")
    artifact = validate_calibration(
        judgment_path=judgment,
        labels_dev_path=JUDGE / "labels-dev-smoke.jsonl",
        labels_holdout_path=JUDGE / "labels-holdout-smoke.jsonl",
        rubric_path=JUDGE / "rubric-pointwise.yaml",
        output_path=tmp_path / "calibration.json",
    )
    assert artifact.gating_allowed is False
    assert any("n_holdout=" in reason for reason in artifact.block_reasons)


def test_holdout_n_counts_paired_labels_not_judgment_items(tmp_path: Path) -> None:
    """One holdout label against 200 judgment items yields n_holdout=1, not 200."""
    judgment = _calibration_judgment(tmp_path, judge_family="qwen", candidate_family="llama")
    single = tmp_path / "labels-holdout-single.jsonl"
    single.write_text(
        (JUDGE / "labels-holdout.jsonl").read_text(encoding="utf-8").splitlines()[0] + "\n",
        encoding="utf-8",
    )

    artifact = validate_calibration(
        judgment_path=judgment,
        labels_dev_path=JUDGE / "labels-dev.jsonl",
        labels_holdout_path=single,
        rubric_path=JUDGE / "rubric-pointwise.yaml",
        output_path=tmp_path / "calibration-single.json",
    )

    assert artifact.holdout.n == 1
    assert artifact.gating_allowed is False
    assert any(
        "n_holdout=1" in reason and "min_holdout_n=150" in reason
        for reason in artifact.block_reasons
    )


def test_duplicate_label_rows_for_one_case_are_hard_error(tmp_path: Path) -> None:
    judgment = _calibration_judgment(tmp_path, judge_family="qwen", candidate_family="llama")
    lines = (JUDGE / "labels-holdout.jsonl").read_text(encoding="utf-8").splitlines()
    padded = tmp_path / "labels-holdout-padded.jsonl"
    padded.write_text("\n".join(lines + [lines[0]]) + "\n", encoding="utf-8")

    with pytest.raises(JudgeError) as exc:
        validate_calibration(
            judgment_path=judgment,
            labels_dev_path=JUDGE / "labels-dev.jsonl",
            labels_holdout_path=padded,
            rubric_path=JUDGE / "rubric-pointwise.yaml",
            output_path=tmp_path / "calibration-padded.json",
        )

    assert exc.value.code == "DUPLICATE_CASE_ID"
    assert "holdout-case-00000" in str(exc.value)


def test_run_rejects_duplicate_candidate_case_ids(tmp_path: Path) -> None:
    lines = (JUDGE / "candidates-pointwise.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    duplicate = {**first, "generation_id": f"{first['generation_id']}-again"}
    candidates = tmp_path / "candidates-dup.jsonl"
    candidates.write_text("\n".join([*lines, json.dumps(duplicate)]) + "\n", encoding="utf-8")

    with pytest.raises(JudgeError) as exc:
        run_judgment(
            mode=JudgeMode.POINTWISE,
            rubric_path=JUDGE / "rubric-pointwise.yaml",
            candidates_path=candidates,
            pairs_path=None,
            provider="mock",
            model="mock-judge",
            judge_family="qwen",
            candidate_family="llama",
            responses_path=JUDGE / "mock-judge-responses-pointwise.jsonl",
            seed=42,
            output_path=tmp_path / "judgment-dup.json",
        )

    assert exc.value.code == "DUPLICATE_CASE_ID"
    assert first["case_id"] in str(exc.value)


def test_stricter_rubric_threshold_blocks_a_judgment_that_clears_the_default(
    tmp_path: Path,
) -> None:
    """The rubric that produced the judgment owns the threshold, not the 0.60 default."""
    judgment = _calibration_judgment(tmp_path, judge_family="qwen", candidate_family="llama")
    payload = json.loads(judgment.read_text(encoding="utf-8"))
    # Flip a minority of holdout scores so agreement lands between 0.60 and 0.95.
    for index, item in enumerate(payload["items"]):
        if str(item["case_id"]).startswith("holdout-") and index % 5 == 0:
            item["score"] = 1 if int(item["score"]) != 1 else 5
    judgment.write_text(json.dumps(payload), encoding="utf-8")

    strict_rubric = tmp_path / "rubric-strict.yaml"
    strict_rubric.write_text(
        (JUDGE / "rubric-pointwise.yaml")
        .read_text(encoding="utf-8")
        .replace("agreement_threshold: 0.6", "agreement_threshold: 0.95"),
        encoding="utf-8",
    )

    lenient = validate_calibration(
        judgment_path=judgment,
        labels_dev_path=JUDGE / "labels-dev.jsonl",
        labels_holdout_path=JUDGE / "labels-holdout.jsonl",
        rubric_path=JUDGE / "rubric-pointwise.yaml",
        output_path=tmp_path / "calibration-lenient.json",
    )
    strict = validate_calibration(
        judgment_path=judgment,
        labels_dev_path=JUDGE / "labels-dev.jsonl",
        labels_holdout_path=JUDGE / "labels-holdout.jsonl",
        rubric_path=strict_rubric,
        output_path=tmp_path / "calibration-strict.json",
    )

    assert lenient.threshold == 0.60
    assert lenient.gating_allowed is True
    assert strict.threshold == 0.95
    assert strict.gating_allowed is False
    assert any("threshold=0.95" in reason for reason in strict.block_reasons)


def test_cli_judge_validate_requires_rubric(tmp_path: Path) -> None:
    judgment = _calibration_judgment(tmp_path, judge_family="qwen", candidate_family="llama")
    result = runner.invoke(
        app,
        [
            "judge",
            "validate",
            "--judgments",
            str(judgment),
            "--labels-dev",
            str(JUDGE / "labels-dev.jsonl"),
            "--labels-holdout",
            str(JUDGE / "labels-holdout.jsonl"),
            "--output",
            str(tmp_path / "calibration.json"),
        ],
    )

    assert result.exit_code != 0
    assert not (tmp_path / "calibration.json").exists()


def test_dev_labels_in_holdout_file_are_hard_error(tmp_path: Path) -> None:
    judgment = _calibration_judgment(tmp_path, judge_family="qwen", candidate_family="llama")
    bad = tmp_path / "bad-holdout.jsonl"
    bad.write_text((JUDGE / "labels-dev.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(JudgeError) as exc:
        validate_calibration(
            judgment_path=judgment,
            labels_dev_path=JUDGE / "labels-dev.jsonl",
            labels_holdout_path=bad,
            rubric_path=JUDGE / "rubric-pointwise.yaml",
            output_path=tmp_path / "calibration.json",
        )
    assert exc.value.code == "DEV_USED_FOR_GATE"


def test_attach_refuses_failing_calibration(tmp_path: Path) -> None:
    judgment = _calibration_judgment(tmp_path, judge_family="llama", candidate_family="llama")
    calibration_path = tmp_path / "calibration.json"
    validate_calibration(
        judgment_path=judgment,
        labels_dev_path=JUDGE / "labels-dev.jsonl",
        labels_holdout_path=JUDGE / "labels-holdout.jsonl",
        rubric_path=JUDGE / "rubric-pointwise.yaml",
        output_path=calibration_path,
    )
    with pytest.raises(JudgeError) as exc:
        attach_calibration(
            judgment_path=judgment,
            calibration_path=calibration_path,
            output_path=tmp_path / "out.json",
        )
    assert exc.value.code == "UNCALIBRATED_JUDGE"


def test_attach_refuses_tampered_passing_calibration(tmp_path: Path) -> None:
    judgment = _calibration_judgment(tmp_path, judge_family="qwen", candidate_family="llama")
    calibration_path = tmp_path / "calibration.json"
    validate_calibration(
        judgment_path=judgment,
        labels_dev_path=JUDGE / "labels-dev.jsonl",
        labels_holdout_path=JUDGE / "labels-holdout.jsonl",
        rubric_path=JUDGE / "rubric-pointwise.yaml",
        output_path=calibration_path,
    )
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    payload["threshold"] = 0.0
    calibration_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(JudgeError) as exc:
        attach_calibration(
            judgment_path=judgment,
            calibration_path=calibration_path,
            output_path=tmp_path / "out.json",
        )

    assert exc.value.code == "INVALID_ARTIFACT"
    assert "calibration_digest" in str(exc.value)


def test_attach_refuses_calibration_judgment_digest_mismatch(tmp_path: Path) -> None:
    judgment = _calibration_judgment(tmp_path, judge_family="qwen", candidate_family="llama")
    calibration_path = tmp_path / "calibration.json"
    validate_calibration(
        judgment_path=judgment,
        labels_dev_path=JUDGE / "labels-dev.jsonl",
        labels_holdout_path=JUDGE / "labels-holdout.jsonl",
        rubric_path=JUDGE / "rubric-pointwise.yaml",
        output_path=calibration_path,
    )
    payload = json.loads(judgment.read_text(encoding="utf-8"))
    payload["items"][0]["score"] = 1 if int(payload["items"][0]["score"]) != 1 else 5
    judgment.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(JudgeError) as exc:
        attach_calibration(
            judgment_path=judgment,
            calibration_path=calibration_path,
            output_path=tmp_path / "out.json",
        )

    assert exc.value.code == "CALIBRATION_JUDGMENT_MISMATCH"


def test_attach_refuses_calibration_from_a_different_judgment(tmp_path: Path) -> None:
    """Calibration for judgment A must not attach onto a distinct judgment B."""
    judgment_a = _calibration_judgment(tmp_path, judge_family="qwen", candidate_family="llama")
    calibration_path = tmp_path / "calibration-a.json"
    validate_calibration(
        judgment_path=judgment_a,
        labels_dev_path=JUDGE / "labels-dev.jsonl",
        labels_holdout_path=JUDGE / "labels-holdout.jsonl",
        rubric_path=JUDGE / "rubric-pointwise.yaml",
        output_path=calibration_path,
    )

    judgment_b = tmp_path / "judgment-b.json"
    run_judgment(
        mode=JudgeMode.POINTWISE,
        rubric_path=JUDGE / "rubric-pointwise.yaml",
        candidates_path=JUDGE / "candidates-pointwise.jsonl",
        pairs_path=None,
        provider="mock",
        model="mock-judge",
        judge_family="qwen",
        candidate_family="llama",
        responses_path=JUDGE / "mock-judge-responses-pointwise.jsonl",
        seed=99,
        output_path=judgment_b,
    )

    with pytest.raises(JudgeError) as exc:
        attach_calibration(
            judgment_path=judgment_b,
            calibration_path=calibration_path,
            output_path=tmp_path / "out-b.json",
        )

    assert exc.value.code == "CALIBRATION_JUDGMENT_MISMATCH"


def test_cli_judge_validate_file_primary_without_database_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judgment = _calibration_judgment(tmp_path, judge_family="qwen", candidate_family="llama")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    get_settings.cache_clear()
    calibration = tmp_path / "calibration.json"

    result = runner.invoke(
        app,
        [
            "judge",
            "validate",
            "--judgments",
            str(judgment),
            "--labels-dev",
            str(JUDGE / "labels-dev.jsonl"),
            "--labels-holdout",
            str(JUDGE / "labels-holdout.jsonl"),
            "--rubric",
            str(JUDGE / "rubric-pointwise.yaml"),
            "--output",
            str(calibration),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["gating_allowed"] is True
    assert calibration.is_file()
    get_settings.cache_clear()


def test_cli_judge_commands(tmp_path: Path) -> None:
    judgment = tmp_path / "judgment.json"
    result = runner.invoke(
        app,
        [
            "judge",
            "run",
            "--mode",
            "pointwise",
            "--rubric",
            str(JUDGE / "rubric-pointwise.yaml"),
            "--candidates",
            str(JUDGE / "candidates-pointwise.jsonl"),
            "--provider",
            "mock",
            "--model",
            "mock-judge",
            "--judge-family",
            "qwen",
            "--candidate-family",
            "llama",
            "--responses",
            str(JUDGE / "mock-judge-responses-pointwise.jsonl"),
            "--seed",
            "42",
            "--output",
            str(judgment),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["gating_allowed"] is False


def test_cli_judge_validate_and_attach_calibration(tmp_path: Path) -> None:
    judgment = _calibration_judgment(
        tmp_path,
        judge_family="qwen",
        candidate_family="llama",
    )
    calibration = tmp_path / "calibration.json"
    validate_result = runner.invoke(
        app,
        [
            "judge",
            "validate",
            "--judgments",
            str(judgment),
            "--labels-dev",
            str(JUDGE / "labels-dev.jsonl"),
            "--labels-holdout",
            str(JUDGE / "labels-holdout.jsonl"),
            "--rubric",
            str(JUDGE / "rubric-pointwise.yaml"),
            "--output",
            str(calibration),
        ],
    )

    assert validate_result.exit_code == 0, validate_result.output
    assert json.loads(validate_result.stdout)["gating_allowed"] is True

    attached = tmp_path / "judgment-calibrated.json"
    attach_result = runner.invoke(
        app,
        [
            "judge",
            "attach-calibration",
            "--judgment",
            str(judgment),
            "--calibration",
            str(calibration),
            "--output",
            str(attached),
        ],
    )

    assert attach_result.exit_code == 0, attach_result.output
    assert json.loads(attach_result.stdout)["gating_allowed"] is True
    assert json.loads(attached.read_text(encoding="utf-8"))["calibration_digest"]


def test_judge_package_does_not_import_store_or_hf() -> None:
    judge_root = ROOT / "src" / "evalharness" / "judge"
    forbidden = ("evalharness.store", "huggingface_hub", "datasets", "transformers")
    for path in judge_root.rglob("*.py"):
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


def test_suite_path_still_avoids_judge_store_coupling() -> None:
    suite_root = ROOT / "src" / "evalharness" / "suite"
    for path in suite_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "evalharness.store" not in text
        assert "evalharness.providers" not in text
        assert "huggingface" not in text
