"""The two seams: the upstream Prediction and the downstream audio event schema.

`Prediction` is what the model side hands us — one per detected keystroke. It is the
entire contract between their code and ours. They own everything that produces it;
we own everything that consumes it.

`AudioEvent` is the frozen schema from CLAUDE.md, serialized to the frontend.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

from exposure.alert import Severity, decide_alert
from exposure.risk import risk_score

# Retained in the schema but constant in this version: no context signal, no password
# detection. A future version needs the field, so it stays.
MODE: Literal["normal"] = "normal"

KeyTopK = list[tuple[str, float]]


@dataclass(slots=True)
class Prediction:
    """UPSTREAM SEAM. One per detected keystroke, produced by the model side.

    key_topk        Candidate keys with probabilities, highest first. Truncated to
                    top-k rather than the full vocabulary.
    confidence      Classifier top-1 probability. Should equal key_topk[0][1].
    timestamp       Unix seconds of the keystroke onset. Defaults to arrival time,
                    but the model side should stamp it at onset detection — arrival
                    time drifts by however long inference took.
    speech_present  From their VAD. False if they do not surface it; the alert panel
                    labels that case unknown rather than claiming silence.
    """

    key_topk: KeyTopK
    confidence: float | None = None
    timestamp: float | None = None
    speech_present: bool = False

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.confidence is None:
            self.confidence = self.key_topk[0][1] if self.key_topk else 0.0


@dataclass(slots=True)
class AudioEvent:
    """DOWNSTREAM SEAM. The frozen schema, one event per detected keystroke."""

    key_top1: str
    key_topk: KeyTopK
    confidence: float
    timestamp: float
    risk_score: float
    typing_detected: bool
    speech_present: bool
    alert: bool
    alert_severity: Severity
    type: str = "keystroke"
    mode: str = MODE

    def to_dict(self) -> dict[str, Any]:
        """Serialize in the schema's declared field order."""
        return {
            "type": self.type,
            "key_top1": self.key_top1,
            "key_topk": [[k, p] for k, p in self.key_topk],
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "risk_score": self.risk_score,
            "typing_detected": self.typing_detected,
            "speech_present": self.speech_present,
            "alert": self.alert,
            "alert_severity": self.alert_severity,
        }


def assemble_event(prediction: Prediction) -> AudioEvent:
    """Run one prediction through risk scoring and the alert decision.

    Note what crosses into `risk_score`: only the probability column. The key
    identities in `key_topk` continue on to the transcript panel and are never seen
    by the risk branch. That separation is the architectural rule and it is worth
    keeping visible right here, at the one place both branches are in scope.
    """
    probs = [p for _, p in prediction.key_topk]
    risk = risk_score(probs)

    # An event exists because an onset was detected, so typing_detected is True by
    # construction. It stays an explicit field because the frontend renders the
    # reasoning behind each alert, and because a future silence heartbeat would set
    # it False.
    typing_detected = bool(prediction.key_topk)

    alert, severity = decide_alert(risk, typing_detected)

    return AudioEvent(
        key_top1=prediction.key_topk[0][0] if prediction.key_topk else "",
        key_topk=prediction.key_topk,
        confidence=float(prediction.confidence or 0.0),
        timestamp=float(prediction.timestamp or time.time()),
        risk_score=risk,
        typing_detected=typing_detected,
        speech_present=prediction.speech_present,
        alert=alert,
        alert_severity=severity,
    )
