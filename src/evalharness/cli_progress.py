"""Rich adapter for transport-neutral pipeline progress events."""

from __future__ import annotations

from types import TracebackType

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from evalharness.observability import ProgressEvent


class PipelineProgress:
    """Render one evolving task while pipeline stages change.

    Core execution code only knows about ``ProgressEvent``. This adapter owns every
    Rich-specific decision and can be replaced by a web/socket adapter later.
    """

    def __init__(self, console: Console) -> None:
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[detail]}"),
            console=console,
            disable=not console.is_terminal,
            transient=False,
        )
        self._task: TaskID | None = None
        self._stage = ""

    def __enter__(self) -> PipelineProgress:
        self._progress.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._progress.stop()

    def __call__(self, event: ProgressEvent) -> None:
        description = event.stage.value.replace("_", " ").title()
        detail = self._detail(event)
        total = max(event.total, 1)
        completed = min(event.completed, total)
        if self._task is None:
            self._task = self._progress.add_task(
                description, total=total, completed=completed, detail=detail
            )
        elif self._stage != event.stage.value:
            self._progress.update(
                self._task,
                description=description,
                total=total,
                completed=completed,
                detail=detail,
            )
        else:
            self._progress.update(
                self._task,
                total=total,
                completed=completed,
                detail=detail,
            )
        self._stage = event.stage.value

    @staticmethod
    def _detail(event: ProgressEvent) -> str:
        counters = " ".join(f"{key}={value}" for key, value in event.counters.items())
        return " · ".join(part for part in (event.message, counters) if part)
