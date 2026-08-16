"""
Exposure scoring: turns a stream of per-keystroke classification events into
a single live "how acoustically exposed is this typing session right now"
score (0-100), plus a per-zone breakdown for the dashboard heatmap.

Also tracks a simple rolling ambient noise-floor estimate, used both for
the SNR feature (features.py) and as an input to the score itself — a
noisier room should genuinely lower the exposure score, since it makes the
attack harder in practice, not just in theory.
"""

from __future__ import annotations
from collections import deque, defaultdict
from dataclasses import dataclass, field
import time
import numpy as np

import config


@dataclass
class KeystrokeEvent:
    timestamp: float
    zone: str
    confidence: float
    snr_db: float


class NoiseFloorTracker:
    """Tracks a slowly-adapting estimate of ambient (non-keystroke) RMS level,
    used as the SNR baseline. Update with audio chunks that do NOT contain a
    detected onset.
    """

    def __init__(self, alpha: float = 0.02):
        self.alpha = alpha  # smoothing factor; smaller = slower adaptation
        self._rms = 1e-4

    def update(self, chunk: np.ndarray) -> None:
        chunk_rms = float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))
        self._rms = (1 - self.alpha) * self._rms + self.alpha * chunk_rms

    @property
    def rms(self) -> float:
        return self._rms


class ExposureScorer:
    def __init__(self,
                 window_events: int = config.EXPOSURE_WINDOW_EVENTS,
                 decay_seconds: float = config.EXPOSURE_DECAY_SECONDS):
        self.window_events = window_events
        self.decay_seconds = decay_seconds
        self._events: deque[KeystrokeEvent] = deque(maxlen=window_events)

    def add_event(self, zone: str, confidence: float, snr_db: float,
                  timestamp: float | None = None) -> None:
        self._events.append(KeystrokeEvent(
            timestamp=timestamp if timestamp is not None else time.time(),
            zone=zone, confidence=confidence, snr_db=snr_db,
        ))

    def current_score(self) -> dict:
        """Returns a dict with the overall 0-100 exposure score plus a
        per-zone confidence breakdown (for the live heatmap).
        """
        if not self._events:
            return {"exposure_score": 0.0, "zone_breakdown": {}, "n_events": 0}

        now = time.time()
        weighted_conf_sum = 0.0
        weight_sum = 0.0
        zone_weighted: dict[str, float] = defaultdict(float)
        zone_weight: dict[str, float] = defaultdict(float)

        for ev in self._events:
            age = now - ev.timestamp
            time_weight = max(0.0, 1.0 - age / self.decay_seconds)
            if time_weight <= 0:
                continue
            # SNR contributes multiplicatively: very noisy conditions (low/negative
            # SNR) suppress the effective confidence, reflecting real attack difficulty.
            snr_factor = float(np.clip((ev.snr_db + 5) / 20, 0.0, 1.0))
            w = time_weight * snr_factor
            weighted_conf_sum += w * ev.confidence
            weight_sum += w
            zone_weighted[ev.zone] += w * ev.confidence
            zone_weight[ev.zone] += w

        overall = 100.0 * (weighted_conf_sum / weight_sum) if weight_sum > 0 else 0.0
        breakdown = {
            zone: round(100.0 * zone_weighted[zone] / zone_weight[zone], 1)
            for zone in zone_weighted if zone_weight[zone] > 0
        }
        return {
            "exposure_score": round(overall, 1),
            "zone_breakdown": breakdown,
            "n_events": len(self._events),
        }
