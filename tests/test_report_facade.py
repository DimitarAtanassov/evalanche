"""Characterization: reporting public façade keeps stable export names."""

from __future__ import annotations

import evalharness.reporting as reporting_pkg
from evalharness.reporting import assemble, io, render
from evalharness.reporting import report as report_facade


def test_report_facade_exports_public_symbols() -> None:
    assert report_facade.PRIMARY_METRIC is assemble.PRIMARY_METRIC
    assert report_facade.RunReport is assemble.RunReport
    assert report_facade.assemble_run_report is assemble.assemble_run_report
    assert report_facade.build_report is io.build_report
    assert report_facade.write_report is io.write_report
    assert report_facade.report_to_html is render.report_to_html
    assert report_facade.report_to_json is render.report_to_json
    assert report_facade.report_to_junit is render.report_to_junit


def test_package_reexports_core_api() -> None:
    for name in (
        "RunReport",
        "assemble_run_report",
        "build_report",
        "write_report",
        "report_to_json",
        "report_to_html",
    ):
        assert getattr(reporting_pkg, name) is getattr(report_facade, name)
