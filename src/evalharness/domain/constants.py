"""Literals shared by scoring, reporting, and the suite artifacts."""

from __future__ import annotations

# Slice key for the un-sliced population. Real slice keys are "dimension=value",
# so the sentinel cannot collide with one.
OVERALL_SLICE = "__overall__"

# Default metric the report headlines when a caller names none.
PRIMARY_METRIC = "exact_match"

# Published artifact contracts. Changing one is a schema break for its readers.
REPORT_SCHEMA_VERSION = "2.2"
COMPARE_SCHEMA_VERSION = "1.0"
SUITE_SCHEMA_VERSION = "0.1"
MATRIX_SCHEMA_VERSION = "0.1"
BASELINE_SCHEMA_VERSION = "0.1"
GATES_SCHEMA_VERSION = "0.1"
# Schema shared by file-primary judge, calibration, and RAG evidence artifacts.
SUPPLEMENT_SCHEMA_VERSION = "0.1"
