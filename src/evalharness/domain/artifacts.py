"""Shared artifact contract primitives.

Versioned JSON inputs across suite, matrix, gates, judge, and calibration all
reject unknown fields. One base keeps that policy in a single place.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base model for versioned inputs that reject unknown fields."""

    model_config = ConfigDict(extra="forbid")
