"""Provider registry via entry points."""

from __future__ import annotations

from importlib.metadata import entry_points

from evalharness.core.protocols import Provider


def load_provider(name: str, **kwargs: object) -> Provider:
    eps = entry_points(group="evalharness.providers")
    for ep in eps:
        if ep.name == name:
            provider_cls = ep.load()
            return provider_cls(**kwargs)  # type: ignore[no-any-return]
    available = [ep.name for ep in eps]
    raise ValueError(f"Unknown provider '{name}'. Available: {available}")
