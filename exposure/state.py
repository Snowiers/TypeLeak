"""Alert log and the latched exposure indicator.

The latch exists because of a property of the pipeline that is easy to miss: events
are only produced when a keystroke is detected, so nothing is emitted when typing
stops. There is no "typing ended" signal to switch an indicator off. Rather than
inventing a timeout — which would silently hide exposure that is still true — the
indicator latches on and stays lit until manually cleared.

The clear itself is a frontend concern (Ctrl+Shift+X, window-focus only). This class
holds the authoritative state so a late-joining or reconnecting client can be told
what it missed.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from exposure.alert import Severity
from exposure.event import AudioEvent

DEFAULT_LOG_SIZE = 200

_SEVERITY_RANK: dict[Severity, int] = {"none": 0, "moderate": 1, "critical": 2}


class ExposureState:
    """Latched exposure flag plus a bounded log of recent alerts."""

    def __init__(self, log_size: int = DEFAULT_LOG_SIZE) -> None:
        self.log_size = log_size
        self._alerts: deque[AudioEvent] = deque(maxlen=log_size)
        self._latched = False
        self._peak: Severity = "none"
        self._total_events = 0
        self._total_alerts = 0

    def record(self, event: AudioEvent) -> None:
        """Fold one event into the running state."""
        self._total_events += 1
        if not event.alert:
            return
        self._total_alerts += 1
        self._alerts.append(event)
        self._latched = True
        if _SEVERITY_RANK[event.alert_severity] > _SEVERITY_RANK[self._peak]:
            self._peak = event.alert_severity

    def clear_latch(self) -> None:
        """Manual clear. Resets the indicator and peak severity, keeps the log."""
        self._latched = False
        self._peak = "none"

    @property
    def latched(self) -> bool:
        return self._latched

    @property
    def peak_severity(self) -> Severity:
        """Highest severity seen since the last clear — what the indicator shows."""
        return self._peak

    def recent_alerts(self, limit: int | None = None) -> list[AudioEvent]:
        """Most recent alerts, newest first."""
        alerts = list(reversed(self._alerts))
        return alerts if limit is None else alerts[:limit]

    def snapshot(self) -> dict[str, Any]:
        """State summary, sent to a client on connect so it starts in sync.

        This is a separate message type from the keystroke event — it is not the
        frozen schema and should not be confused with it.
        """
        return {
            "type": "state",
            "latched": self._latched,
            "peak_severity": self._peak,
            "total_events": self._total_events,
            "total_alerts": self._total_alerts,
            "recent_alerts": [e.to_dict() for e in self.recent_alerts()],
        }
