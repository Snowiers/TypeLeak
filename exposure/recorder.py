"""Session recorder: writes a live event stream to a replayable JSONL file.

The missing half of the replay workflow. `ReplaySource` can play a fixture back
deterministically, but until now the only fixture was synthetic. This turns any real
session into one, so a demo can be recorded from genuine classifier output and then
replayed identically for as many takes as it needs.

Output is the same JSONL that `ReplaySource` reads, so the round trip is:

    python -m exposure --record takes/session.jsonl        # capture a real run
    python -m exposure --source replay --fixture takes/session.jsonl

Two deliberate choices, both about not losing a good take:

Each line is flushed as it is written, so interrupting the process with Ctrl-C leaves
a complete, usable file rather than whatever happened to be in the buffer.

An existing path is never overwritten. A recorded session may be the only copy of a
take that is hard to reproduce, so the recorder picks the next free suffix instead of
destroying it.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import Any

MAX_SUFFIX_ATTEMPTS = 1000


def next_free_path(path: Path) -> Path:
    """Return `path`, or the first free `name-1`, `name-2`, ... variant.

    Recording over an earlier take would be an unrecoverable loss, so the caller
    never gets a path that already exists.
    """
    if not path.exists():
        return path
    for index in range(1, MAX_SUFFIX_ATTEMPTS):
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find a free filename near {path}")


class EventRecorder:
    """Appends assembled events to a JSONL file, one per line."""

    def __init__(self, path: str | Path) -> None:
        target = Path(path)
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        self.path = next_free_path(target)
        self._handle = self.path.open("w", encoding="utf-8")
        self._count = 0

    @property
    def count(self) -> int:
        """Events written so far. Worth logging when a session ends."""
        return self._count

    def write(self, payload: dict[str, Any]) -> None:
        """Record one event. Non-keystroke messages are ignored.

        State snapshots are server bookkeeping, not part of the event stream, and
        `ReplaySource` skips them on read — keeping them out of the file entirely
        makes a recording readable as exactly what it claims to be.
        """
        if payload.get("type") != "keystroke":
            return
        self._handle.write(json.dumps(payload) + "\n")
        self._handle.flush()
        self._count += 1

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> EventRecorder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
