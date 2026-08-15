"""Acoustic side-channel exposure monitor: risk, alert decision, and transport.

This package owns the middle of the pipeline. Upstream, the model side hands us a
per-keystroke Prediction. Downstream, we emit the frozen audio event schema over a
local websocket. Nothing here touches audio, models, or the DOM.

Observation only. Nothing in this package modifies the audio stream.
"""

from exposure.event import AudioEvent, Prediction, assemble_event
from exposure.risk import risk_score
from exposure.alert import CRITICAL_THRESHOLD, RISK_THRESHOLD, decide_alert

__all__ = [
    "AudioEvent",
    "Prediction",
    "assemble_event",
    "risk_score",
    "decide_alert",
    "RISK_THRESHOLD",
    "CRITICAL_THRESHOLD",
]
