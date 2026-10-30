from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import time
from pathlib import Path
from typing import Any


class Timer:
    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, *_args):
        self.elapsed_ms = (time.perf_counter() - self.started) * 1000.0


class JsonlMetrics:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()


UPDATE_TIMING = {
    'log': 'updates.jsonl',
    'update': 'completed optimizer update, 1-based',
    'update_seconds': 'wall seconds including accumulation, synchronization, development checks and checkpoint writes scheduled within this update',
    'elapsed_seconds': 'measured training-session wall seconds plus prior logged runtime on resume; excludes process downtime',
    'unix_time': 'update completion timestamp, Unix seconds',
    'timestamp_utc': 'same completion timestamp, ISO 8601 UTC',
}


class UpdateLogger:
    def __init__(self, path: str | Path, *, resume: bool = False) -> None:
        self.writer = JsonlMetrics(path)
        self.offset = 0.0
        if self.writer.path.exists() and self.writer.path.stat().st_size:
            if not resume:
                raise FileExistsError(f'update log already exists: {self.writer.path}')
            with self.writer.path.open(encoding='utf8') as stream:
                for line in stream:
                    if line.strip():
                        self.offset = float(json.loads(line)['elapsed_seconds'])
            if not math.isfinite(self.offset) or self.offset < 0:
                raise ValueError('invalid elapsed time in update log')
        self.session_start = None
        self.started = None
        self.update = None

    def start(self, update: int) -> None:
        if self.started is not None:
            raise RuntimeError('previous update has not finished')
        if type(update) is not int or update <= 0:
            raise ValueError('update must be a positive integer')
        self.started = time.perf_counter()
        if self.session_start is None:
            self.session_start = self.started
        self.update = update

    def finish(self, **metadata) -> dict[str, Any]:
        """Call after device synchronization and all scheduled update work."""
        if self.started is None:
            raise RuntimeError('update has not started')
        finished = time.perf_counter()
        timestamp = time.time()
        row = dict(metadata, update=self.update,
                   update_seconds=finished-self.started,
                   elapsed_seconds=self.offset+finished-self.session_start,
                   unix_time=timestamp,
                   timestamp_utc=datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z'))
        self.writer.write(row)
        self.started = None
        return row

