from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path


class InspectionProgress:
    def __init__(self, output_dir: Path, *, total_phases: int = 8) -> None:
        self.output_dir = output_dir.expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "inspection.log"
        self.total_phases = total_phases
        self._handle = self.log_path.open("w", encoding="utf-8")

    def phase(self, number: int, message: str) -> None:
        self._emit(f"[{number}/{self.total_phases}] {message}")

    def detail(self, message: str) -> None:
        self._emit(f"      {message}")

    def complete(self, message: str) -> None:
        self._emit(f"[done] {message}")

    def failed(self, message: str) -> None:
        self._emit(f"[failed] {message}")

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def _emit(self, message: str) -> None:
        timestamp = datetime.now(UTC).isoformat(timespec="seconds")
        line = f"{timestamp} {message}"
        print(line, file=sys.stderr, flush=True)
        self._handle.write(line + "\n")
        self._handle.flush()

    def __enter__(self) -> InspectionProgress:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()
