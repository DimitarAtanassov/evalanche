"""Suite contract, determinism, privacy, and isolation regression tests."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

import evalharness.compare.service as compare_service
import evalharness.wiring as wiring
from evalharness.charts import script_json
from evalharness.cli import app
from evalharness.judge import attach_calibration, run_judgment, validate_calibration
from evalharness.judge.models import JudgeMode
from evalharness.suite import (
    SuiteValidationError,
    build_suite,
    load_suite,
    suite_to_html,
    suite_to_json,
    write_suite_artifacts,
)
from evalharness.suite.models import CompareArtifact
from evalharness.suite.render import (
    LATENCY_CHART_DIV_ID,
    LEADERBOARD_CHART_DIV_ID,
    SLICE_CHART_DIV_ID,
)

ROOT = Path(__file__).parents[2]
GOLDEN = ROOT / "fixtures" / "suite" / "golden"
JUDGE_FIXTURES = ROOT / "fixtures" / "judge"
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


def _sha256_canonical(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


def _judgment_body_digest(judgment: dict[str, object]) -> str:
    """Independent restatement of the published judgment identity rule.

    Recomputed here rather than imported so the digest binding is asserted against
    the contract instead of against whatever the implementation happens to do.
    """
    gate_fields = {"gating_allowed", "gating_block_reason", "calibration_digest"}
    return _sha256_canonical(
        {key: value for key, value in judgment.items() if key not in gate_fields}
    )


def _judgment_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
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
        "items": [],
    }
    return {**payload, **overrides}


def _passing_calibration_payload(
    *,
    judgment_digest: str,
    gating_allowed: bool = True,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": "0.1",
        "judgment_digest": judgment_digest,
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
        "gating_allowed": gating_allowed,
        "plain_language": "Passing holdout calibration.",
        "block_reasons": [],
    }
    return {**body, "calibration_digest": _sha256_canonical(body)}


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


def test_comparison_row_is_fully_pinned() -> None:
    report = build_suite(GOLDEN / "suite.yaml")

    assert report.comparisons == [
        {
            "path": "qa-compare.json",
            "artifact_sha256": ("beabd8218d8500dd45462e1b0381e235d04f04b854e423d577d11271f0de3b30"),
            "baseline_run_id": "10000000-0000-4000-8000-000000000001",
            "candidate_run_id": "10000000-0000-4000-8000-000000000002",
            "excluded_flaky_cases": ["qa-5"],
            "metric": "exact_match",
            "n": 9,
            "baseline": 0.6666666667,
            "candidate": 0.7777777778,
            "absolute_delta": 0.1111111111,
            "relative_delta": 0.1666666667,
            "cohens_h": 0.2506583334,
            "ci_low": -0.1111111111,
            "ci_high": 0.3333333333,
            "p_value": 0.625,
            "significant_bh": False,
        }
    ]


def test_additive_json_formatting_does_not_change_canonical_suite_digest(
    mutable_suite: Path,
) -> None:
    before = build_suite(mutable_suite)
    comparison_path = mutable_suite.parent / "qa-compare.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=4) + "\n\n",
        encoding="utf-8",
    )

    after = build_suite(mutable_suite)

    assert after.suite_digest == before.suite_digest
    assert after.comparisons == before.comparisons


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


def test_runs_compare_artifact_validates_and_round_trips_through_suite_build(
    mutable_suite: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_run_id = uuid.UUID("10000000-0000-4000-8000-000000000001")
    candidate_run_id = uuid.UUID("10000000-0000-4000-8000-000000000002")
    run = SimpleNamespace(
        dataset_id=uuid.UUID(int=1),
        prompt_template_id=uuid.UUID(int=2),
        config_sha256="same-config",
        repeats=1,
    )

    class FakeRepository:
        def __init__(self, _session: object) -> None:
            pass

        async def get_run(self, run_id: uuid.UUID) -> SimpleNamespace | None:
            return run if run_id in {baseline_run_id, candidate_run_id} else None

    class FakeResult:
        def __init__(self, rows: list[tuple[str, int, bool]]) -> None:
            self._rows = rows

        def all(self) -> list[tuple[str, int, bool]]:
            return self._rows

    class FakeSession:
        def __init__(self) -> None:
            self._calls = 0

        async def execute(self, _statement: object) -> FakeResult:
            rows = (
                [("case-1", 0, True), ("case-2", 0, False), ("case-3", 0, False)]
                if self._calls == 0
                else [("case-1", 0, True), ("case-2", 0, True), ("case-3", 0, False)]
            )
            self._calls += 1
            return FakeResult(rows)

    @asynccontextmanager
    async def fake_session_scope() -> AsyncIterator[FakeSession]:
        yield FakeSession()

    # The CLI takes its store from the composition root, so that is where it is swapped.
    monkeypatch.setattr(wiring, "RunRepository", FakeRepository)
    monkeypatch.setattr(compare_service, "session_scope", fake_session_scope)
    # Suite artifacts must live under the manifest directory; see the containment test.
    compare_path = mutable_suite.parent / "produced-compare.json"

    result = CLI.invoke(
        app,
        [
            "runs",
            "compare",
            str(baseline_run_id),
            str(candidate_run_id),
            "--output",
            str(compare_path),
        ],
    )

    assert result.exit_code == 0
    produced = json.loads(compare_path.read_text(encoding="utf-8"))
    assert json.loads(result.stdout) == produced
    validated = CompareArtifact.model_validate(produced)
    assert validated.schema_version == "1.0"
    assert validated.result.model_dump(mode="json") == produced["result"]

    manifest = _manifest(mutable_suite)
    manifest["compares"] = [{"path": compare_path.name}]
    _write_manifest(mutable_suite, manifest)
    output = tmp_path / "suite-output"
    report = write_suite_artifacts(mutable_suite, output)
    built = json.loads((output / "suite.json").read_text(encoding="utf-8"))

    assert report.comparisons[0]["artifact_sha256"]
    assert built["comparisons"] == report.comparisons
    assert built["comparisons"][0]["baseline_run_id"] == str(baseline_run_id)
    assert built["comparisons"][0]["candidate_run_id"] == str(candidate_run_id)
    assert built["comparisons"][0]["metric"] == "exact_match"
    assert built["comparisons"][0]["n"] == 3


def test_runs_compare_reports_no_effect_when_both_arms_are_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run compared against its own outcomes must publish a null result.

    McNemar on zero discordant pairs has no evidence either way, so anything but
    a zero delta at p=1.0 would let noise read as a regression or a win.
    """
    baseline_run_id = uuid.UUID("10000000-0000-4000-8000-000000000001")
    candidate_run_id = uuid.UUID("10000000-0000-4000-8000-000000000002")
    run = SimpleNamespace(
        dataset_id=uuid.UUID(int=1),
        prompt_template_id=uuid.UUID(int=2),
        config_sha256="same-config",
        repeats=1,
    )
    rows = [("case-1", 0, True), ("case-2", 0, False), ("case-3", 0, True)]

    class FakeRepository:
        def __init__(self, _session: object) -> None:
            pass

        async def get_run(self, run_id: uuid.UUID) -> SimpleNamespace | None:
            return run if run_id in {baseline_run_id, candidate_run_id} else None

    class FakeResult:
        def all(self) -> list[tuple[str, int, bool]]:
            return list(rows)

    class FakeSession:
        async def execute(self, _statement: object) -> FakeResult:
            return FakeResult()

    @asynccontextmanager
    async def fake_session_scope() -> AsyncIterator[FakeSession]:
        yield FakeSession()

    monkeypatch.setattr(wiring, "RunRepository", FakeRepository)
    monkeypatch.setattr(compare_service, "session_scope", fake_session_scope)
    output = tmp_path / "identical-compare.json"

    result = CLI.invoke(
        app,
        [
            "runs",
            "compare",
            str(baseline_run_id),
            str(candidate_run_id),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.stdout)
    assert emitted == json.loads(output.read_text(encoding="utf-8"))
    assert emitted["excluded_flaky_cases"] == []
    assert emitted["result"] == {
        "metric": "exact_match",
        "n": 3,
        "baseline": pytest.approx(2 / 3),
        "candidate": pytest.approx(2 / 3),
        "absolute_delta": 0.0,
        "relative_delta": 0.0,
        "cohens_h": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0,
        "p_value": 1.0,
        "significant_bh": False,
    }
    assert CompareArtifact.model_validate(emitted).schema_version == "1.0"


def test_cli_suite_validate_happy_path() -> None:
    result = CLI.invoke(app, ["suite", "validate", str(GOLDEN / "suite.yaml")])

    assert result.exit_code == 0
    assert result.stdout.count("\n") == 1
    assert json.loads(result.stdout) == {
        "compares": 1,
        "members": 4,
        "name": "suite-golden",
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
    assert result.stdout.count("\n") == 1
    assert json.loads(result.stdout) == {
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


def test_optional_judge_rag_artifacts_render_bounded_separate_panels(
    mutable_suite: Path,
) -> None:
    calibration = _passing_calibration_payload(judgment_digest="sha256:another-judgment")
    judgment = _judgment_payload(items=[{"reasoning": "must not be copied into suite"}])
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


def _attach(judgment: dict[str, object], calibration: dict[str, object]) -> dict[str, object]:
    """Apply what ``judge attach-calibration`` writes onto a judgment."""
    return {
        **judgment,
        "gating_allowed": True,
        "calibration_digest": calibration["calibration_digest"],
        "gating_block_reason": calibration["plain_language"],
    }


def _publish_judge_artifacts(
    manifest_path: Path,
    *,
    calibration: dict[str, object] | None,
    judgment: dict[str, object],
) -> None:
    manifest = _manifest(manifest_path)
    if calibration is not None:
        (manifest_path.parent / "calibration.json").write_text(
            json.dumps(calibration), encoding="utf-8"
        )
        manifest["calibrations"] = [{"path": "calibration.json"}]
    (manifest_path.parent / "judgment.json").write_text(json.dumps(judgment), encoding="utf-8")
    manifest["judge_artifacts"] = [{"path": "judgment.json"}]
    _write_manifest(manifest_path, manifest)


def test_attached_judgment_lights_suite_badge_when_calibration_digest_matches(
    mutable_suite: Path,
) -> None:
    """Suite badge lights only when judgment digest matches a passing calibrations[] entry."""
    judgment = _judgment_payload()
    calibration = _passing_calibration_payload(judgment_digest=_judgment_body_digest(judgment))
    digest = calibration["calibration_digest"]

    _publish_judge_artifacts(
        mutable_suite,
        calibration=calibration,
        judgment=_attach(judgment, calibration),
    )
    report = build_suite(mutable_suite)

    assert report.calibrations is not None
    assert report.calibrations[0]["gating_allowed"] is True
    assert report.calibrations[0]["calibration_digest"] == digest
    assert report.judge_artifacts is not None
    assert report.judge_artifacts[0]["calibration_digest"] == digest
    assert report.judge_artifacts[0]["gating_allowed"] is True


def test_real_calibration_pipeline_lights_the_badge_only_for_the_attached_judgment(
    mutable_suite: Path,
    tmp_path: Path,
) -> None:
    """Drive the shipped judge pipeline into the suite instead of hand-built payloads.

    The neighbouring badge tests restate the digest rule in the fixture helpers,
    so they would still pass if ``validate_calibration`` and ``attach_calibration``
    emitted something the suite refuses. This one runs both for real and publishes
    the pre-attach judgment alongside the attached one.
    """
    judgment = tmp_path / "judgment.json"
    run_judgment(
        mode=JudgeMode.POINTWISE,
        rubric_path=JUDGE_FIXTURES / "rubric-pointwise.yaml",
        candidates_path=JUDGE_FIXTURES / "candidates-calibration.jsonl",
        pairs_path=None,
        provider="mock",
        model="mock-judge",
        judge_family="qwen",
        candidate_family="llama",
        responses_path=JUDGE_FIXTURES / "mock-judge-responses-calibration.jsonl",
        seed=1,
        output_path=judgment,
    )
    calibration_path = mutable_suite.parent / "calibration.json"
    calibration = validate_calibration(
        judgment_path=judgment,
        labels_dev_path=JUDGE_FIXTURES / "labels-dev.jsonl",
        labels_holdout_path=JUDGE_FIXTURES / "labels-holdout.jsonl",
        rubric_path=JUDGE_FIXTURES / "rubric-pointwise.yaml",
        output_path=calibration_path,
    )
    attach_calibration(
        judgment_path=judgment,
        calibration_path=calibration_path,
        output_path=mutable_suite.parent / "judgment-attached.json",
    )
    shutil.copyfile(judgment, mutable_suite.parent / "judgment-unattached.json")
    manifest = _manifest(mutable_suite)
    manifest["calibrations"] = [{"path": "calibration.json"}]
    manifest["judge_artifacts"] = [
        {"path": "judgment-attached.json"},
        {"path": "judgment-unattached.json"},
    ]
    _write_manifest(mutable_suite, manifest)

    report = build_suite(mutable_suite)

    assert calibration.gating_allowed is True
    assert report.calibrations is not None
    assert report.calibrations[0]["calibration_digest"] == calibration.calibration_digest
    assert report.judge_artifacts is not None
    badges = {str(row["path"]): row for row in report.judge_artifacts}
    assert badges["judgment-attached.json"]["gating_allowed"] is True
    assert badges["judgment-attached.json"]["calibration_digest"] == calibration.calibration_digest
    assert badges["judgment-unattached.json"]["gating_allowed"] is False
    assert badges["judgment-unattached.json"]["calibration_digest"] is None


def test_stolen_calibration_digest_cannot_light_suite_badge(mutable_suite: Path) -> None:
    """A judgment that quotes another judgment's passing digest stays dark."""
    calibrated = _judgment_payload()
    calibration = _passing_calibration_payload(judgment_digest=_judgment_body_digest(calibrated))
    thief = _attach(
        _judgment_payload(items=[{"case_id": "case-1", "score": 5, "reasoning": "ungraded"}]),
        calibration,
    )
    assert thief["calibration_digest"] == calibration["calibration_digest"]

    _publish_judge_artifacts(mutable_suite, calibration=calibration, judgment=thief)
    report = build_suite(mutable_suite)

    assert report.calibrations is not None
    assert report.calibrations[0]["gating_allowed"] is True
    assert report.judge_artifacts is not None
    assert report.judge_artifacts[0]["gating_allowed"] is False
    assert report.judge_artifacts[0]["calibration_digest"] == calibration["calibration_digest"]
    assert "CALIBRATION_JUDGMENT_MISMATCH" in str(report.judge_artifacts[0]["gating_block_reason"])


def test_judgment_bound_to_a_failing_calibration_cannot_light_suite_badge(
    mutable_suite: Path,
) -> None:
    """A correct digest binding is not enough; the calibration must also pass."""
    judgment = _judgment_payload()
    calibration = _passing_calibration_payload(
        judgment_digest=_judgment_body_digest(judgment),
        gating_allowed=False,
    )

    _publish_judge_artifacts(
        mutable_suite,
        calibration=calibration,
        judgment=_attach(judgment, calibration),
    )
    report = build_suite(mutable_suite)

    assert report.calibrations is not None
    assert report.calibrations[0]["gating_allowed"] is False
    assert report.judge_artifacts is not None
    assert report.judge_artifacts[0]["gating_allowed"] is False


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


def test_script_json_escapes_script_terminator_and_js_line_separators() -> None:
    payload = {"label": "</script><script>alert(1)</script>\u2028\u2029 & done"}

    encoded = script_json(payload)

    assert "</script" not in encoded
    assert "\u2028" not in encoded
    assert "\u2029" not in encoded
    assert json.loads(encoded) == payload


def test_hostile_member_label_cannot_break_out_of_the_chart_script(
    mutable_suite: Path,
) -> None:
    """Member labels reach the inline Vega spec, so they must not close the script."""
    manifest = _manifest(mutable_suite)
    member_runs = manifest["member_runs"]
    assert isinstance(member_runs, list)
    first = member_runs[0]
    assert isinstance(first, dict)
    first["label"] = "</script><script>alert(1)</script>"
    _write_manifest(mutable_suite, manifest)

    html = suite_to_html(build_suite(mutable_suite))

    assert "\\u003c/script\\u003e\\u003cscript\\u003e" in html
    assert "<script>alert(1)</script>" not in html


@pytest.mark.parametrize(
    "declared",
    ["../outside-run.json", "nested/../../outside-run.json"],
)
def test_suite_refuses_member_artifacts_outside_the_manifest_directory(
    mutable_suite: Path,
    declared: str,
) -> None:
    """A shared manifest must not be able to hash files from anywhere on the host."""
    outside = mutable_suite.parent.parent / "outside-run.json"
    shutil.copyfile(mutable_suite.parent / "qa-baseline.json", outside)
    manifest = _manifest(mutable_suite)
    member_runs = manifest["member_runs"]
    assert isinstance(member_runs, list)
    first = member_runs[0]
    assert isinstance(first, dict)
    first["path"] = declared
    _write_manifest(mutable_suite, manifest)

    with pytest.raises(SuiteValidationError) as exc:
        build_suite(mutable_suite)

    assert exc.value.code == "ARTIFACT_OUTSIDE_SUITE"


def test_suite_refuses_absolute_artifact_path_outside_the_manifest_directory(
    mutable_suite: Path,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside-run.json"
    shutil.copyfile(mutable_suite.parent / "qa-baseline.json", outside)
    manifest = _manifest(mutable_suite)
    member_runs = manifest["member_runs"]
    assert isinstance(member_runs, list)
    first = member_runs[0]
    assert isinstance(first, dict)
    first["path"] = str(outside)
    _write_manifest(mutable_suite, manifest)

    with pytest.raises(SuiteValidationError) as exc:
        build_suite(mutable_suite)

    assert exc.value.code == "ARTIFACT_OUTSIDE_SUITE"


def test_suite_rejects_calibration_without_judgment_digest(mutable_suite: Path) -> None:
    """A self-consistent calibration with no judgment binding must not publish a gate.

    ``attach_calibration`` refuses such an artifact, so ``calibrations[]`` showing it as
    passing would advertise a gate no judgment could ever legitimately claim.
    """
    bound = _passing_calibration_payload(judgment_digest="sha256:judgment")
    body = {
        key: value
        for key, value in bound.items()
        if key not in {"judgment_digest", "calibration_digest"}
    }
    unbound = {**body, "calibration_digest": _sha256_canonical(body)}
    assert unbound["gating_allowed"] is True
    (mutable_suite.parent / "unbound-calibration.json").write_text(
        json.dumps(unbound),
        encoding="utf-8",
    )
    manifest = _manifest(mutable_suite)
    manifest["calibrations"] = [{"path": "unbound-calibration.json"}]
    _write_manifest(mutable_suite, manifest)

    with pytest.raises(SuiteValidationError) as exc:
        build_suite(mutable_suite)

    assert exc.value.code == "INVALID_ARTIFACT"
    assert "judgment_digest" in str(exc.value)


def test_suite_rejects_tampered_calibration_digest(mutable_suite: Path) -> None:
    calibration = {
        **_passing_calibration_payload(judgment_digest="sha256:judgment"),
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
