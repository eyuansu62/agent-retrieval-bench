from __future__ import annotations

import sys
from typing import TextIO


class ProgressReporter:
    def __init__(self, enabled: bool = False, stream: TextIO | None = None) -> None:
        self.enabled = enabled
        self.stream = stream or sys.stderr
        self.line_open = False

    def message(self, text: str) -> None:
        if not self.enabled:
            return
        if self.line_open:
            print(file=self.stream, flush=True)
            self.line_open = False
        print(f"[arb] {text}", file=self.stream, flush=True)

    def bar(self, label: str, total: int) -> "ProgressBar":
        return ProgressBar(label=label, total=total, reporter=self)


class ProgressBar:
    def __init__(self, label: str, total: int, reporter: ProgressReporter) -> None:
        self.label = label
        self.total = max(0, total)
        self.reporter = reporter
        self.current = 0
        self.rendered = False
        if self.reporter.enabled:
            self._render()

    def update(self, step: int = 1, suffix: str = "") -> None:
        if not self.reporter.enabled:
            return
        self.current = min(self.total, self.current + step)
        self._render(suffix=suffix)

    def finish(self, suffix: str = "") -> None:
        if not self.reporter.enabled or not self.rendered:
            return
        self.current = self.total
        self._render(suffix=suffix)
        print(file=self.reporter.stream, flush=True)
        self.reporter.line_open = False

    def _render(self, suffix: str = "") -> None:
        width = 28
        if self.total:
            filled = int(width * self.current / self.total)
            percent = int(100 * self.current / self.total)
        else:
            filled = width
            percent = 100
        bar = "#" * filled + "-" * (width - filled)
        suffix_text = f" {suffix}" if suffix else ""
        print(
            f"\r[arb] {self.label}: [{bar}] {self.current}/{self.total} {percent:3d}%{suffix_text}",
            end="",
            file=self.reporter.stream,
            flush=True,
        )
        self.reporter.line_open = True
        self.rendered = True
