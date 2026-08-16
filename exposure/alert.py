"""Alert decision engine.

Implements the severity table from CLAUDE.md. Two thresholds on one axis — the risk
score — because this version has no context signal, so severity means "how
exploitable," not "how sensitive the target."

| Typing detected | Risk score                  | Alert          |
|-----------------|-----------------------------|----------------|
| No              | -                           | none           |
| Yes             | below RISK_THRESHOLD        | none, log only |
| Yes             | above RISK_THRESHOLD        | moderate       |
| Yes             | above CRITICAL_THRESHOLD    | critical       |

Speech presence is deliberately absent from this function's signature. It never
affects the decision — it is reported on the event for the alert panel to display,
because the point of the system is to tell the truth about exposure regardless of
what else is in the audio.
"""

from __future__ import annotations

from typing import Literal

Severity = Literal["none", "moderate", "critical"]

# PLACEHOLDERS. These are the only two tuning constants in the system and they need
# calibration against real recordings once the model side lands. Both live here so
# calibration is a two-line change.
RISK_THRESHOLD = 0.80
CRITICAL_THRESHOLD = 0.95


def decide_alert(
    risk: float,
    typing_detected: bool,
    *,
    risk_threshold: float = RISK_THRESHOLD,
    critical_threshold: float = CRITICAL_THRESHOLD,
) -> tuple[bool, Severity]:
    """Return `(alert, alert_severity)` for one keystroke event.

    Thresholds are exclusive lower bounds: a score exactly at the threshold does not
    fire. This keeps a default-valued score from tripping an alert by coincidence.
    """
    if not typing_detected:
        return False, "none"
    if risk > critical_threshold:
        return True, "critical"
    if risk > risk_threshold:
        return True, "moderate"
    return False, "none"
