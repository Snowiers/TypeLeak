"""
Streaming wrapper around detector.py's detect_onsets().

detect_onsets() was written as a BATCH function (run once over a whole
recorded session in process.py). To use it live, we periodically re-run it
over a recent rolling window of audio pulled from the RingBuffer, rather
than adapting it into a truly incremental algorithm -- this keeps us using
the EXACT SAME detection code as training (the whole point, per detector.py's
docstring), instead of a reimplementation that could drift from it.

Trade-off: onsets near the very end of the current analysis window may not
be confirmed yet (librosa's peak-picking needs a bit of trailing audio
context to confirm a peak is a true local max) -- they'll simply be found
on the NEXT re-analysis once more audio has arrived, which is why we
dedupe against previously-reported onsets rather than assuming each
analysis pass is authoritative on its own.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

import config
import detector  # the real, shared detect_onsets/extract_clip/clip_to_melspec


@dataclass
class OnsetEvent:
    sample_index: int   # global/monotonic sample index of the detected onset
    strength: float       # rough peak amplitude near the onset (informational only)


class StreamingOnsetDetector:
    def __init__(self,
                 sample_rate: int = config.SAMPLE_RATE,
                 cfg: dict = config.DETECTOR_CFG,
                 analysis_window_s: float = config.ONSET_ANALYSIS_WINDOW_S,
                 reanalysis_interval_s: float = config.ONSET_REANALYSIS_INTERVAL_S):
        self.sr = sample_rate
        self.cfg = cfg
        self.analysis_window_samples = int(analysis_window_s * sample_rate)
        self.reanalysis_interval_samples = int(reanalysis_interval_s * sample_rate)
        self.min_gap_samples = int(cfg["detection"]["min_gap_ms"] / 1000.0 * sample_rate)

        self._last_analysis_at = 0       # total_written value at last analysis pass
        self._last_reported_global = -10**9  # global sample index of last reported onset

    def maybe_process(self, ring_buffer) -> list[OnsetEvent]:
        """Call this after every chunk is appended to the ring buffer. Returns
        newly-confirmed onsets (usually empty -- only fires roughly every
        `reanalysis_interval_s`, and only for genuinely new onsets).
        """
        total = ring_buffer.total_written()
        if total - self._last_analysis_at < self.reanalysis_interval_samples:
            return []
        self._last_analysis_at = total

        window_start = max(0, total - self.analysis_window_samples)
        audio = ring_buffer.read_absolute_range(window_start, total)
        if audio is None or len(audio) < config.FRAME_HOP * 4:
            return []  # not enough audio yet for a meaningful analysis pass

        onset_samples_relative = detector.detect_onsets(audio, self.sr, self.cfg)

        events: list[OnsetEvent] = []
        for s_rel in onset_samples_relative:
            s_global = window_start + s_rel
            if s_global - self._last_reported_global < self.min_gap_samples:
                continue  # duplicate re-detection (overlapping window) or too close to previous
            w0 = max(0, s_rel - int(0.005 * self.sr))
            w1 = min(len(audio), s_rel + int(0.02 * self.sr))
            strength = float(np.max(np.abs(audio[w0:w1]))) if w1 > w0 else 0.0
            events.append(OnsetEvent(sample_index=s_global, strength=strength))
            self._last_reported_global = s_global

        return events
