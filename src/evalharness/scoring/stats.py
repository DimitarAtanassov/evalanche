"""Compatibility shim over ``evalharness.statistics``; import from there in new code."""

from __future__ import annotations

from evalharness.statistics import percentile, wilson_interval

__all__ = ["percentile", "wilson_interval"]
