"""Replay a recorded event stream at its original timing.

Demo insurance. `FakeSource` is random, so every run produces a different sequence of
alerts — fine for development, bad for rehearsal, where you want to practise talking
over a stream you have already seen. `ReplaySource` plays a JSONL file back at the
timing it was recorded with, so the same alert lands at the same moment every time.

It reads the *event* schema and reconstructs the upstream `Prediction`, rather than
replaying assembled events directly. That means replayed audio runs through the real
risk and alert code path, so a threshold change is visible in replay instead of being
frozen into the recording.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path

from exposure.event import Prediction

# Gaps longer than this are compressed. A recording with a two-minute pause in it
# should not stall a demo for two minutes.
MAX_GAP_SECONDS = 3.0

# Used when a recording carries no usable timing (identical or absent timestamps).
DEFAULT_GAP_SECONDS = 0.12


class ReplaySource:
    """Replays Predictions parsed from a JSONL event file."""

    def __init__(
        self,
        path: str | Path,
        *,
        loop: bool = True,
        speed: float = 1.0,
        max_gap: float = MAX_GAP_SECONDS,
        preserve_timestamps: bool = False,
    ) -> None:
        self._path = Path(path)
        self._loop = loop
        self._speed = speed if speed > 0 else 1.0
        self._max_gap = max_gap
        self._preserve_timestamps = preserve_timestamps
        self._predictions, self._gaps = self._load()

    @property
    def length(self) -> int:
        return len(self._predictions)

    def _load(self) -> tuple[list[Prediction], list[float]]:
        """Parse the file into Predictions plus the gap preceding each one."""
        predictions: list[Prediction] = []
        timestamps: list[float] = []

        with self._path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError as exc:
                    raise ValueError(
                        f"{self._path}:{line_number}: not valid JSON"
                    ) from exc
                if record.get("type") != "keystroke":
                    continue
                try:
                    key_topk = [(k, float(p)) for k, p in record["key_topk"]]
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{self._path}:{line_number}: keystroke record has no usable "
                        f"'key_topk' field"
                    ) from exc
                predictions.append(
                    Prediction(
                        key_topk=key_topk,
                        confidence=record.get("confidence"),
                        timestamp=record.get("timestamp"),
                        speech_present=bool(record.get("speech_present", False)),
                    )
                )
                timestamps.append(float(record.get("timestamp") or 0.0))

        if not predictions:
            raise ValueError(f"{self._path}: no keystroke events found")

        gaps = [0.0]
        for previous, current in zip(timestamps, timestamps[1:]):
            delta = current - previous
            if delta <= 0.0:
                delta = DEFAULT_GAP_SECONDS
            gaps.append(min(delta, self._max_gap))
        return predictions, gaps

    async def predictions(self) -> AsyncIterator[Prediction]:
        while True:
            for prediction, gap in zip(self._predictions, self._gaps):
                if gap:
                    await asyncio.sleep(gap / self._speed)
                yield self._stamp(prediction)
            if not self._loop:
                return
            # Looping restarts the burst rhythm; insert one pause so the seam between
            # the last and first event does not read as an unbroken run of typing.
            await asyncio.sleep(min(self._max_gap, 1.5) / self._speed)

    def _stamp(self, prediction: Prediction) -> Prediction:
        """Re-stamp to now unless the original recording time was asked for.

        The alert panel shows wall-clock times. A replayed stream showing yesterday's
        timestamps looks broken to anyone watching the demo.
        """
        if self._preserve_timestamps:
            return prediction
        return Prediction(
            key_topk=prediction.key_topk,
            confidence=prediction.confidence,
            timestamp=time.time(),
            speech_present=prediction.speech_present,
        )
