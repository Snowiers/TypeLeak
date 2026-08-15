"""
Keystroke onset detection.

Keystrokes are short broadband transients (press click + release click).
Spectral flux — the frame-to-frame increase in spectral energy, summed
across frequency bins — is a standard, robust onset-detection signal for
exactly this kind of percussive transient (it's the same core idea used in
music onset detection for drum hits).

This module is deliberately stateful and incremental: it's designed to be
fed a continuous stream of small audio chunks (as they arrive from the mic)
rather than operating on a full pre-recorded file, so it can run live.
"""

from __future__ import annotations
import numpy as np
from collections import deque
from dataclasses import dataclass

import config


@dataclass
class OnsetEvent:
    sample_index: int   # global/monotonic sample index of the detected onset
    strength: float      # spectral flux value at detection (rough loudness proxy)


class OnsetDetector:
    def __init__(self,
                 sample_rate: int = config.SAMPLE_RATE,
                 frame_ms: float = config.ONSET_FRAME_MS,
                 hop_ms: float = config.ONSET_HOP_MS,
                 history_frames: int = config.ONSET_HISTORY_FRAMES,
                 threshold_k: float = config.ONSET_THRESHOLD_K,
                 refractory_ms: float = config.ONSET_REFRACTORY_MS):
        self.sample_rate = sample_rate
        self.frame_len = int(sample_rate * frame_ms / 1000)
        self.hop_len = int(sample_rate * hop_ms / 1000)
        self.refractory_samples = int(sample_rate * refractory_ms / 1000)

        self._window = np.hanning(self.frame_len).astype(np.float32)
        self._flux_history = deque(maxlen=history_frames)
        self._prev_spectrum: np.ndarray | None = None

        self._sample_carry = np.zeros(0, dtype=np.float32)  # leftover samples < 1 frame
        self._samples_consumed = 0   # global sample index of next frame start
        self._last_onset_sample = -10**9

        self.threshold_k = threshold_k

    def _spectral_flux(self, frame: np.ndarray) -> float:
        spec = np.abs(np.fft.rfft(frame * self._window))
        if self._prev_spectrum is None:
            self._prev_spectrum = spec
            return 0.0
        diff = spec - self._prev_spectrum
        flux = np.sum(diff[diff > 0])  # only count energy increases
        self._prev_spectrum = spec
        return float(flux)

    def process_chunk(self, chunk: np.ndarray, chunk_start_sample: int) -> list[OnsetEvent]:
        """Feed the next chunk of raw audio (in stream order).

        `chunk_start_sample` is the global sample index this chunk begins at
        (i.e. RingBuffer.total_written() *before* this chunk was appended) —
        needed so onset events carry a global index usable for later lookback.
        """
        events: list[OnsetEvent] = []
        data = np.concatenate([self._sample_carry, chunk])
        # base global index of `data[0]`
        base_index = chunk_start_sample - len(self._sample_carry)

        n_frames = 1 + (len(data) - self.frame_len) // self.hop_len if len(data) >= self.frame_len else 0
        for i in range(n_frames):
            start = i * self.hop_len
            frame = data[start:start + self.frame_len]
            frame_global_start = base_index + start
            flux = self._spectral_flux(frame)

            if len(self._flux_history) >= 4:  # need a little history before judging
                local_mean = float(np.mean(self._flux_history))
                local_std = float(np.std(self._flux_history))
                threshold = local_mean + self.threshold_k * local_std
                onset_sample = frame_global_start + self.frame_len // 2

                if (flux > threshold and flux > 1e-6 and
                        onset_sample - self._last_onset_sample >= self.refractory_samples):
                    events.append(OnsetEvent(sample_index=onset_sample, strength=flux))
                    self._last_onset_sample = onset_sample

            self._flux_history.append(flux)

        # keep leftover tail for next call
        consumed = n_frames * self.hop_len if n_frames > 0 else 0
        self._sample_carry = data[consumed:]
        return events
