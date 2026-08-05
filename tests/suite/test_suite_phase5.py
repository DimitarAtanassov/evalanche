"""Phase 5 contract, determinism, privacy, and isolation regression tests."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from evalharness.cli import app
from evalharness.suite import (
    SuiteValidationError,
    build_suite,
    load_suite,
    suite_to_html,
    suite_to_json,
    write_suite_artifacts,
)
from evalharness.suite.render import (
    LATENCY_CHART_DIV_ID,
    LEADERBOARD_CHART_DIV_ID,
    SLICE_CHART_DIV_ID,
)

ROOT = Path(__file__).parents[2]
GOLDEN = ROOT / "fixtures" / "suite" / "golden"
CLI = CliRunner()
MEMBER_ARTIFACTS = (
    "qa-baseline.json",
    "qa-candidate.json",
    "classification-baseline.json",
    "classification-candidate.json",
    "qa-compare.json",
)


@pytest.fixture
def mutable_suite(tmp_path: Path) -> Path:
    """Copy the golden suite so invalid cases cannot mutate shared fixtures."""
    suite_dir = tmp_path / "suite"
    shutil.copytree(GOLDEN, suite_dir)
    return suite_dir / "suite.yaml"


def _manifest(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_manifest(path: Path, value: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _accessible_table(html: str, caption: str) -> str:
    marker = f"<caption>{caption}</caption>"
    start = html.index(marker)
    end = html.index("</table>", start)
    return html[start:end]


def test_golden_suite_json_is_byte_identical() -> None:
    report = build_suite(GOLDEN / "suite.yaml")

    actual = suite_to_json(report)

    assert actual == (GOLDEN / "suite.json").read_text(encoding="utf-8")


def test_golden_suite_html_is_byte_identical() -> None:
    actual = suite_to_html(build_suite(GOLDEN / "suite.yaml"))

    assert actual == (GOLDEN / "suite.html").read_text(encoding="utf-8")


def test_suite_outputs_are_deterministic_and_offline(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    write_suite_artifacts(GOLDEN / "suite.yaml", first)
    write_suite_artifacts(GOLDEN / "suite.yaml", second)

    assert (first / "suite.json").read_bytes() == (second / "suite.json").read_bytes()
    assert (first / "suite.html").read_bytes() == (second / "suite.html").read_bytes()
    html = (first / "suite.html").read_text(encoding="utf-8")
    assert 'src="http' not in html
    assert "cdn." not in html
    assert "altair-viz-" not in html
    assert html.count("<table") >= 6
    for chart_id in (
        LEADERBOARD_CHART_DIV_ID,
        SLICE_CHART_DIV_ID,
        LATENCY_CHART_DIV_ID,
    ):
        assert f'id="{chart_id}"' in html


def test_write_suite_artifacts_leaves_member_and_compare_bytes_unchanged(
    tmp_path: Path,
) -> None:
    suite_dir = tmp_path / "suite"
    shutil.copytree(GOLDEN, suite_dir)
    before = {name: (suite_dir / name).read_bytes() for name in MEMBER_ARTIFACTS}

    write_suite_artifacts(suite_dir / "suite.yaml", tmp_path / "out")

    for name, content in before.items():
        assert (suite_dir / name).read_bytes() == content


def test_suite_view_excludes_non_publishable_member_with_visible_reason() -> None:
    view = build_suite(GOLDEN / "suite.yaml").model_dump(mode="json")

    excluded_id = "20000000-0000-4000-8000-000000000001"
    leaderboard_ids = {
        entry["run_id"] for board in view["leaderboards"] for entry in board["entries"]
    }

    assert excluded_id not in leaderboard_ids
    assert view["exclusions"] == [
        {
            "run_id": excluded_id,
            "reason": (
                "run status is failed; written generations 8/10; coverage 0.8000 below floor 0.9800"
            ),
        }
    ]


def test_quality_claim_row_is_fully_pinned() -> None:
    """Pin one full quality row; golden suite.json already covers the rest."""
    report = build_suite(GOLDEN / "suite.yaml")

    domain_general = next(
        table
        for table in report.quality_tables
        if table["dimension"] == "domain" and table["value"] == "general"
    )
    pinned = next(
        row
        for row in domain_general["rows"]
        if row["run_id"] == "10000000-0000-4000-8000-000000000002"
    )

    assert pinned == {
        "ci_high": 0.9433,
        "ci_low": 0.4902,
        "dataset": "squad-smoke",
        "dataset_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
        "label": "mock-large / squad-smoke",
        "method": "wilson",
        "metric": "exact_match",
        "model_digest": "mock-large-v2",
        "n": 10,
        "run_id": "10000000-0000-4000-8000-000000000002",
        "value": 0.8,
    }


def test_slices_are_overall_then_weakest_first_per_member() -> None:
    report = build_suite(GOLDEN / "suite.yaml")

    candidate = [
        row for row in report.slices if row["run_id"] == "10000000-0000-4000-8000-000000000002"
    ]

    assert [row["slice"] for row in candidate] == [
        "__overall__",
        "difficulty=hard",
        "difficulty=easy",
    ]


def test_failure_gallery_is_bounded_redacted_and_raw_payload_free() -> None:
    report = build_suite(GOLDEN / "suite.yaml")

    serialized = suite_to_json(report)
    baseline = next(
        row
        for row in report.failure_gallery
        if row["run_id"] == "10000000-0000-4000-8000-000000000001"
    )

    assert len(report.failure_gallery) <= 24
    assert "[REDACTED]" in str(baseline["input"])
    assert "private-token" not in serialized
    assert "raw_response" not in serialized
    assert all(
        len(value) <= 280
        for row in report.failure_gallery
        for key in ("input", "reference", "output")
        if isinstance((value := row.get(key)), str)
    )


def test_failure_gallery_text_stays_within_280_chars_including_ellipsis(
    mutable_suite: Path,
) -> None:
    artifact_path = mutable_suite.parent / "qa-baseline.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["case_examples"][0]["input"] = "word " * 400
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    report = build_suite(mutable_suite)

    truncated = next(
        str(row["input"])
        for row in report.failure_gallery
        if row["run_id"] == "10000000-0000-4000-8000-000000000001"
    )
    assert truncated.endswith("…")
    assert len(truncated) == 280


def test_accessible_leaderboard_and_slice_tables_include_identity_columns() -> None:
    html = suite_to_html(build_suite(GOLDEN / "suite.yaml"))

    leaderboard = _accessible_table(
        html, "Accessible leaderboard data, non-publishable members excluded"
    )
    slices = _accessible_table(html, "Accessible slice data, weakest first within each member")

    assert leaderboard.startswith(
        "<caption>Accessible leaderboard data, non-publishable members excluded</caption>"
    )
    assert (
        "<th>Dataset / metric</th><th>Member</th><th>Score</th>"
        "<th>95% CI</th><th>n</th><th>Dataset digest</th><th>Model digest</th>"
    ) in leaderboard
    assert (
        "<th>Member</th><th>Dataset</th><th>Metric</th><th>Slice</th><th>Score</th>"
        "<th>95% CI</th><th>n</th><th>Dataset digest</th><th>Model digest</th>"
    ) in slices
    assert "mock-large-v2" in leaderboard
    assert "1111111111111111111111111111111111111111111111111111111111111111" in slices


def test_latency_chart_has_adjacent_accessible_table() -> None:
    html = suite_to_html(build_suite(GOLDEN / "suite.yaml"))

    assert f'id="{LATENCY_CHART_DIV_ID}"' in html
    latency = _accessible_table(html, "Accessible p95 latency data")
    assert latency.startswith("<caption>Accessible p95 latency data</caption>")
    assert "<th>Member</th><th>p95 latency (ms)</th>" in latency
    assert "mock-large / squad-smoke" in latency
    assert "24.00" in latency


def test_latency_chart_and_table_omitted_when_all_p95_are_zero(
    mutable_suite: Path,
) -> None:
    for filename in MEMBER_ARTIFACTS[:-1]:
        path = mutable_suite.parent / filename
        artifact = json.loads(path.read_text(encoding="utf-8"))
        zeros = {key: 0.0 for key in artifact["latency_ms"]}
        artifact["latency_ms"] = zeros
        path.write_text(json.dumps(artifact), encoding="utf-8")

    html = suite_to_html(build_suite(mutable_suite))

    assert f'id="{LATENCY_CHART_DIV_ID}"' not in html
    assert "Accessible p95 latency data" not in html


@pytest.mark.parametrize(
    ("target", "version"),
    [
        ("suite", "0.2"),
        ("run", "2.0"),
        ("compare", "9.9"),
    ],
)
def test_unknown_schema_versions_are_rejected(
    mutable_suite: Path,
    target: str,
    version: str,
) -> None:
    if target == "suite":
        manifest = _manifest(mutable_suite)
        manifest["schema_version"] = version
        _write_manifest(mutable_suite, manifest)
    else:
        filename = "qa-baseline.json" if target == "run" else "qa-compare.json"
        artifact_path = mutable_suite.parent / filename
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact["schema_version"] = version
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(SuiteValidationError, match="UNSUPPORTED_SCHEMA"):
        load_suite(mutable_suite)


def test_unknown_manifest_fields_are_rejected(mutable_suite: Path) -> None:
    manifest = _manifest(mutable_suite)
    manifest["database_url"] = "postgresql://must-not-be-read"
    _write_manifest(mutable_suite, manifest)

    with pytest.raises(SuiteValidationError, match="INVALID_MANIFEST"):
        load_suite(mutable_suite)


def test_raw_response_in_case_examples_is_rejected(mutable_suite: Path) -> None:
    artifact_path = mutable_suite.parent / "qa-baseline.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["case_examples"][0]["raw_response"] = {"authorization": "secret"}
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(SuiteValidationError, match="raw_response"):
        load_suite(mutable_suite)


def test_primary_metric_must_exist_for_every_matching_member(mutable_suite: Path) -> None:
    manifest = _manifest(mutable_suite)
    primary_metrics = manifest["primary_metrics"]
    assert isinstance(primary_metrics, list)
    first = primary_metrics[0]
    assert isinstance(first, dict)
    first["metric"] = "invented_metric"
    _write_manifest(mutable_suite, manifest)

    with pytest.raises(SuiteValidationError, match="PRIMARY_METRIC_UNKNOWN"):
        load_suite(mutable_suite)


def test_missing_member_artifact_raises_missing_artifact(mutable_suite: Path) -> None:
    (mutable_suite.parent / "qa-baseline.json").unlink()

    with pytest.raises(SuiteValidationError, match="MISSING_ARTIFACT"):
        load_suite(mutable_suite)


def test_member_dataset_without_a_primary_metric_fails_validate_and_build(
    mutable_suite: Path,
) -> None:
    manifest = _manifest(mutable_suite)
    primary_metrics = manifest["primary_metrics"]
    assert isinstance(primary_metrics, list)
    manifest["primary_metrics"] = [
        primary
        for primary in primary_metrics
        if isinstance(primary, dict) and primary.get("dataset") != "news-smoke"
    ]
    _write_manifest(mutable_suite, manifest)

    with pytest.raises(SuiteValidationError, match="PRIMARY_METRIC_UNKNOWN.*news-smoke"):
        load_suite(mutable_suite)
    with pytest.raises(SuiteValidationError, match="PRIMARY_METRIC_UNKNOWN.*news-smoke"):
        build_suite(mutable_suite)


def test_comparison_must_reference_declared_members(mutable_suite: Path) -> None:
    comparison_path = mutable_suite.parent / "qa-compare.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison["candidate_run_id"] = "not-a-member"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")

    with pytest.raises(SuiteValidationError, match="declared suite members"):
        load_suite(mutable_suite)


def test_cli_suite_validate_happy_path() -> None:
    result = CLI.invoke(app, ["suite", "validate", str(GOLDEN / "suite.yaml")])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "compares": 1,
        "members": 4,
        "name": "phase-5-golden",
        "schema_version": "0.1",
        "valid": True,
    }


def test_cli_suite_build_writes_artifacts_and_surfaces_digest(tmp_path: Path) -> None:
    output = tmp_path / "out"
    report = build_suite(GOLDEN / "suite.yaml")

    result = CLI.invoke(
        app,
        [
            "suite",
            "build",
            "--manifest",
            str(GOLDEN / "suite.yaml"),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    # Rich soft-wraps long JSON values with newlines inside string literals.
    assert json.loads(result.stdout.replace("\n", "")) == {
        "members": 4,
        "output": str(output),
        "suite_digest": report.suite_digest,
    }
    assert (output / "suite.json").read_text(encoding="utf-8") == suite_to_json(report)
    assert (output / "suite.html").read_text(encoding="utf-8") == suite_to_html(report)


def test_cli_suite_validate_unsupported_schema_exits_1(mutable_suite: Path) -> None:
    manifest = _manifest(mutable_suite)
    manifest["schema_version"] = "0.2"
    _write_manifest(mutable_suite, manifest)

    result = CLI.invoke(app, ["suite", "validate", str(mutable_suite)])

    assert result.exit_code == 1
    assert "UNSUPPORTED_SCHEMA" in result.stdout


def test_cli_suite_build_missing_artifact_exits_1(
    mutable_suite: Path,
    tmp_path: Path,
) -> None:
    (mutable_suite.parent / "qa-baseline.json").unlink()

    result = CLI.invoke(
        app,
        [
            "suite",
            "build",
            "--manifest",
            str(mutable_suite),
            "--output",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 1
    assert "MISSING_ARTIFACT" in result.stdout


def test_cli_suite_validate_and_build_agree_on_missing_primary_metric(
    mutable_suite: Path,
    tmp_path: Path,
) -> None:
    manifest = _manifest(mutable_suite)
    primary_metrics = manifest["primary_metrics"]
    assert isinstance(primary_metrics, list)
    manifest["primary_metrics"] = [
        row
        for row in primary_metrics
        if isinstance(row, dict) and row.get("dataset") != "news-smoke"
    ]
    _write_manifest(mutable_suite, manifest)

    validate = CLI.invoke(app, ["suite", "validate", str(mutable_suite)])
    build = CLI.invoke(
        app,
        [
            "suite",
            "build",
            "--manifest",
            str(mutable_suite),
            "--output",
            str(tmp_path / "out"),
        ],
    )

    assert validate.exit_code == 1
    assert build.exit_code == 1
    assert "PRIMARY_METRIC_UNKNOWN" in validate.stdout
    assert "PRIMARY_METRIC_UNKNOWN" in build.stdout


def test_suite_package_has_no_runtime_or_persistence_imports() -> None:
    package = ROOT / "src" / "evalharness" / "suite"
    forbidden_prefixes = (
        "evalharness.providers",
        "evalharness.reporting",
        "evalharness.scoring",
        "evalharness.store",
    )
    forbidden_names = {"sqlalchemy", "asyncpg"}
    imports: set[str] = set()
    for path in package.rglob("*"):
        if path.suffix not in {".py", ".j2", ".html"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert "DATABASE_URL" not in text
        assert "evalharness.store" not in text
        assert "evalharness.providers" not in text
        if path.suffix != ".py":
            continue
        tree = ast.parse(text)
        imports.update(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        imports.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )

    assert not {
        name
        for name in imports
        if name in forbidden_names or any(name.startswith(item) for item in forbidden_prefixes)
    }


def test_html_omits_empty_comparison_and_failure_panels(mutable_suite: Path) -> None:
    manifest = _manifest(mutable_suite)
    manifest["compares"] = []
    _write_manifest(mutable_suite, manifest)
    for filename in MEMBER_ARTIFACTS[:-1]:
        path = mutable_suite.parent / filename
        artifact = json.loads(path.read_text(encoding="utf-8"))
        artifact["case_examples"] = []
        path.write_text(json.dumps(artifact), encoding="utf-8")

    html = suite_to_html(build_suite(mutable_suite))

    assert "Paired comparisons" not in html
    assert "Bounded failure examples" not in html


def test_html_omits_empty_leaderboard_when_no_publishable_members(
    mutable_suite: Path,
) -> None:
    for filename in MEMBER_ARTIFACTS[:-1]:
        path = mutable_suite.parent / filename
        artifact = json.loads(path.read_text(encoding="utf-8"))
        artifact["publishable"] = False
        artifact["case_examples"] = []
        path.write_text(json.dumps(artifact), encoding="utf-8")
    manifest = _manifest(mutable_suite)
    manifest["compares"] = []
    _write_manifest(mutable_suite, manifest)

    html = suite_to_html(build_suite(mutable_suite))

    assert "Declared-member leaderboards" not in html
    assert "Paired comparisons" not in html
    assert "Bounded failure examples" not in html


def test_optional_phase6_artifacts_render_bounded_separate_panels(
    mutable_suite: Path,
) -> None:
    calibration_body = {
        "schema_version": "0.1",
        "judgment_digest": "sha256:judgment",
        "rubric_name": "helpfulness",
        "rubric_version": "1.0.0",
        "holdout": {
            "label_set_id": "holdout-v1",
            "n": 150,
            "agreement_metric": "cohen_kappa",
            "agreement": 0.72,
            "agreement_ci": None,
        },
        "dev": {
            "label_set_id": "dev-v1",
            "n": 50,
            "agreement_metric": "cohen_kappa",
            "agreement": 0.70,
            "agreement_ci": None,
        },
        "threshold": 0.60,
        "min_holdout_n": 150,
        "min_dev_n": 50,
        "family_separation_ok": True,
        "gating_allowed": True,
        "plain_language": "Passing holdout calibration.",
        "block_reasons": [],
    }
    calibration_digest = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                calibration_body,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()
    )
    calibration = {**calibration_body, "calibration_digest": calibration_digest}
    judgment = {
        "schema_version": "0.1",
        "mode": "pointwise",
        "rubric_name": "helpfulness",
        "rubric_version": "1.0.0",
        "judge_model": {
            "provider": "mock",
            "model": "mock-judge",
            "resolved_version": "sha256:judge",
        },
        "gating_allowed": False,
        "calibration_digest": None,
        "gating_block_reason": "Informational until calibration is attached.",
        "items": [{"reasoning": "must not be copied into suite"}],
    }
    rag = {
        "schema_version": "0.1",
        "run_id": "rag-run",
        "model_digest": "sha256:candidate",
        "retrieval": {"status": "ok", "aggregate": {"value": 0.42, "n": 10}},
        "faithfulness": {
            "status": "unavailable",
            "aggregate": {"unsupported_claim_rate": None, "n": 0},
        },
        "citations": {
            "attribution": {"status": "ok", "value": 0.75, "n": 4},
            "full_source_document": "must not be copied into suite",
        },
        "gating_allowed": False,
    }
    for name, payload in (
        ("calibration.json", calibration),
        ("judgment.json", judgment),
        ("rag.json", rag),
    ):
        (mutable_suite.parent / name).write_text(json.dumps(payload), encoding="utf-8")
    manifest = _manifest(mutable_suite)
    manifest["calibrations"] = [{"path": "calibration.json"}]
    manifest["judge_artifacts"] = [{"path": "judgment.json"}]
    manifest["rag_artifacts"] = [{"path": "rag.json"}]
    _write_manifest(mutable_suite, manifest)

    report = build_suite(mutable_suite)
    serialized = suite_to_json(report)
    html = suite_to_html(report)

    assert report.calibrations is not None
    assert report.calibrations[0]["gating_allowed"] is True
    assert report.judge_artifacts is not None
    assert report.judge_artifacts[0]["gating_allowed"] is False
    assert report.rag_artifacts is not None
    assert report.rag_artifacts[0]["gating_allowed"] is False
    assert "Judge calibration" in html
    assert "Judge artifacts" in html
    assert "RAG evidence" in html
    assert "must not be copied into suite" not in serialized
    assert "must not be copied into suite" not in html


def _passing_calibration_payload() -> dict[str, object]:
    calibration_body: dict[str, object] = {
        "schema_version": "0.1",
        "judgment_digest": "sha256:judgment",
        "rubric_name": "helpfulness",
        "rubric_version": "1.0.0",
        "holdout": {
            "label_set_id": "holdout-v1",
            "n": 150,
            "agreement_metric": "cohen_kappa",
            "agreement": 0.72,
            "agreement_ci": None,
        },
        "dev": {
            "label_set_id": "dev-v1",
            "n": 50,
            "agreement_metric": "cohen_kappa",
            "agreement": 0.70,
            "agreement_ci": None,
        },
        "threshold": 0.60,
        "min_holdout_n": 150,
        "min_dev_n": 50,
        "family_separation_ok": True,
        "gating_allowed": True,
        "plain_language": "Passing holdout calibration.",
        "block_reasons": [],
    }
    calibration_digest = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                calibration_body,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()
    )
    return {**calibration_body, "calibration_digest": calibration_digest}


def test_attached_judgment_lights_suite_badge_when_calibration_digest_matches(
    mutable_suite: Path,
) -> None:
    """Suite badge lights only when judgment digest matches a passing calibrations[] entry."""
    calibration = _passing_calibration_payload()
    digest = calibration["calibration_digest"]
    assert isinstance(digest, str)
    judgment = {
        "schema_version": "0.1",
        "mode": "pointwise",
        "rubric_name": "helpfulness",
        "rubric_version": "1.0.0",
        "judge_model": {
            "provider": "mock",
            "model": "mock-judge",
            "resolved_version": "sha256:judge",
        },
        "gating_allowed": True,
        "calibration_digest": digest,
        "gating_block_reason": "Passing holdout calibration.",
        "items": [],
    }
    (mutable_suite.parent / "calibration.json").write_text(
        json.dumps(calibration),
        encoding="utf-8",
    )
    (mutable_suite.parent / "judgment-attached.json").write_text(
        json.dumps(judgment),
        encoding="utf-8",
    )
    manifest = _manifest(mutable_suite)
    manifest["calibrations"] = [{"path": "calibration.json"}]
    manifest["judge_artifacts"] = [{"path": "judgment-attached.json"}]
    _write_manifest(mutable_suite, manifest)

    report = build_suite(mutable_suite)

    assert report.calibrations is not None
    assert report.calibrations[0]["gating_allowed"] is True
    assert report.calibrations[0]["calibration_digest"] == digest
    assert report.judge_artifacts is not None
    assert report.judge_artifacts[0]["calibration_digest"] == digest
    assert report.judge_artifacts[0]["gating_allowed"] is True


def test_forged_judgment_gate_bit_cannot_light_suite_badge(
    mutable_suite: Path,
) -> None:
    """Judgment gating_allowed alone is not source of truth; calibration.json is."""
    forged = {
        "schema_version": "0.1",
        "mode": "pointwise",
        "rubric_name": "helpfulness",
        "rubric_version": "1.0.0",
        "judge_model": {
            "provider": "mock",
            "model": "mock-judge",
            "resolved_version": "sha256:judge",
        },
        "gating_allowed": True,
        "calibration_digest": "sha256:not-a-suite-calibration",
        "gating_block_reason": None,
        "items": [],
    }
    (mutable_suite.parent / "forged-judgment.json").write_text(
        json.dumps(forged),
        encoding="utf-8",
    )
    manifest = _manifest(mutable_suite)
    manifest["judge_artifacts"] = [{"path": "forged-judgment.json"}]
    _write_manifest(mutable_suite, manifest)

    report = build_suite(mutable_suite)

    assert report.judge_artifacts is not None
    assert report.judge_artifacts[0]["gating_allowed"] is False
    assert report.judge_artifacts[0]["calibration_digest"] == "sha256:not-a-suite-calibration"


def test_judgment_without_calibration_digest_cannot_light_suite_badge(
    mutable_suite: Path,
) -> None:
    """Forged gating_allowed with null digest stays dark; badges need a passing digest."""
    forged = {
        "schema_version": "0.1",
        "mode": "pointwise",
        "rubric_name": "helpfulness",
        "rubric_version": "1.0.0",
        "judge_model": {
            "provider": "mock",
            "model": "mock-judge",
            "resolved_version": "sha256:judge",
        },
        "gating_allowed": True,
        "calibration_digest": None,
        "gating_block_reason": None,
        "items": [],
    }
    (mutable_suite.parent / "forged-no-digest.json").write_text(
        json.dumps(forged),
        encoding="utf-8",
    )
    manifest = _manifest(mutable_suite)
    manifest["judge_artifacts"] = [{"path": "forged-no-digest.json"}]
    _write_manifest(mutable_suite, manifest)

    report = build_suite(mutable_suite)

    assert report.judge_artifacts is not None
    assert report.judge_artifacts[0]["gating_allowed"] is False
    assert report.judge_artifacts[0]["calibration_digest"] is None


def test_suite_rejects_tampered_calibration_digest(mutable_suite: Path) -> None:
    calibration_body = {
        "schema_version": "0.1",
        "judgment_digest": "sha256:judgment",
        "rubric_name": "helpfulness",
        "rubric_version": "1.0.0",
        "holdout": {
            "label_set_id": "holdout-v1",
            "n": 150,
            "agreement_metric": "cohen_kappa",
            "agreement": 0.72,
            "agreement_ci": None,
        },
        "dev": None,
        "threshold": 0.60,
        "min_holdout_n": 150,
        "min_dev_n": 50,
        "family_separation_ok": True,
        "gating_allowed": True,
        "plain_language": "Passing holdout calibration.",
        "block_reasons": [],
    }
    calibration = {
        **calibration_body,
        "calibration_digest": "sha256:" + ("0" * 64),
    }
    (mutable_suite.parent / "tampered-calibration.json").write_text(
        json.dumps(calibration),
        encoding="utf-8",
    )
    manifest = _manifest(mutable_suite)
    manifest["calibrations"] = [{"path": "tampered-calibration.json"}]
    _write_manifest(mutable_suite, manifest)

    with pytest.raises(SuiteValidationError) as exc:
        build_suite(mutable_suite)

    assert exc.value.code == "INVALID_ARTIFACT"
    assert "calibration_digest" in str(exc.value)
