"""Shim: re-exports the compare service. Prefer ``evalharness.services.compare``."""

from evalharness.services.compare import CompareService

__all__ = ["CompareService"]
