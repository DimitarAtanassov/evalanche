"""Persistence: one repository per table behind a session-scoped unit of work."""

from evalharness.repositories.uow import RunStoreUow

__all__ = ["RunStoreUow"]
