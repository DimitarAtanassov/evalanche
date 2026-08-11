"""Application composition: settings and the container entry points build."""

from evalharness.app.bootstrap import build_container
from evalharness.app.container import AppContainer

__all__ = ["AppContainer", "build_container"]
