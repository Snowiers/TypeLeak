"""
The streaming pipeline: wires audio input -> ring buffer -> onset detection
(detector.py's real algorithm, streamed -- see onset_detector.py) ->
feature extraction (detector.py's real extract_clip/clip_to_melspec) ->
classification (the real trained EfficientNetV2, or an untrained fallback)
-> exposure scoring, and exposes a callback hook for a dashboard/UI.

Usage (live mic, on the demo machine):

    from pipeline import AcousticGuardPipeline
    from audio_io import MicAudioSource

    def on_event(event: dict):
        print(event)

    pipeline = AcousticGuardPipeline(on_event=on_event)
    pipeline.run(MicAudioSource())

Usage (offline / semi-live demo fallback, replaying a wav array):

    from audio_io import ArrayAudioSource
    pipeline.run(ArrayAudioSource(samples, realtime=True))

See test_offline.py for a fully synthetic, no-hardware-needed test.
"""

from __future__ import annotations
import time
from typing import Callable, Optional

import numpy as np

import config
from audio_io import RingBuffer
from onset_detector import StreamingOnsetDetector, OnsetEvent
from features import MelFeatureExtractor
from model import KeyClassifier
from exposure import ExposureScorer, NoiseFloorTracker


class AcousticGuardPipeline:
    def __init__(self, on_event: Optional[Callable[[dict], None]] = None,
                 checkpoint_path: Optional[str] = config.MODEL_CHECKPOINT_PATH):
        self.ring_buffer = RingBuffer()
        self.onset_detector = StreamingOnsetDetector()
        self.feature_extractor = MelFeatureExtractor()
        self.classifier = KeyClassifier(checkpoint_path=checkpoint_path)
        self.exposure_scorer = ExposureScorer()
        self.noise_tracker = NoiseFloorTracker()
        self.on_event = on_event or (lambda event: None)

        clip_samples = int(config.SAMPLE_RATE * config.CLIP_MS / 1000)
        pre_samples = int(config.SAMPLE_RATE * config.PRE_ONSET_MS / 1000)
        self._post_samples = clip_samples - pre_samples  # audio needed AFTER onset
        self._pre_samples = pre_samples

        # onsets confirmed by the onset detector but whose full clip window
        # isn't available in the ring buffer yet
        self._pending: list[OnsetEvent] = []

    def _handle_chunk(self, chunk: np.ndarray) -> None:
        self.ring_buffer.append(chunk)

        new_onsets = self.onset_detector.maybe_process(self.ring_buffer)
        if new_onsets:
            self._pending.extend(new_onsets)
        else:
            # only update the ambient noise floor when nothing new was
            # detected this pass, so keystroke transients don't pollute
            # the "quiet room" estimate
            self.noise_tracker.update(chunk)

        self._process_pending()

    def _process_pending(self) -> None:
        still_pending = []
        total_written = self.ring_buffer.total_written()

        for onset in self._pending:
            window_end = onset.sample_index + self._post_samples
            if window_end > total_written:
                still_pending.append(onset)  # not enough trailing audio yet
                continue

            # read a bit more context than strictly needed on both sides so
            # detector.extract_clip() (which expects "audio" containing the
            # onset with room around it) never needs to zero-pad unless
            # we're genuinely near a real stream boundary
            read_start = max(0, onset.sample_index - self._pre_samples)
            read_end = min(total_written, window_end)
            audio_slice = self.ring_buffer.read_absolute_range(read_start, read_end)
            if audio_slice is None:
                continue  # fell out of the ring buffer -- shouldn't normally happen
            onset_relative = onset.sample_index - read_start

            self._classify_and_score(onset, audio_slice, onset_relative)

        self._pending = still_pending

    def _classify_and_score(self, onset: OnsetEvent, audio_slice: np.ndarray,
                            onset_relative: int) -> None:
        clip = self.feature_extractor.build_clip(audio_slice, onset_relative)
        image = self.feature_extractor.extract(clip)
        label, confidence, probs = self.classifier.predict(image)
        snr_db = self.feature_extractor.snr_estimate(clip, self.noise_tracker.rms)

        below_confidence = confidence < config.MIN_PREDICTION_CONFIDENCE
        # "junk" is a real trained class (release clicks / non-keystroke
        # sounds) -- exclude it from the exposure score.
        is_junk = label == "junk"

        if not below_confidence and not is_junk:
            self.exposure_scorer.add_event(zone=label, confidence=confidence, snr_db=snr_db)
        score_info = self.exposure_scorer.current_score()

        event = {
            "timestamp": time.time(),
            "sample_index": onset.sample_index,
            "onset_strength": onset.strength,
            "predicted_key": label,
            "confidence": round(confidence, 3),
            "below_confidence_threshold": below_confidence,
            "is_junk": is_junk,
            "snr_db": round(snr_db, 1),
            "exposure_score": score_info["exposure_score"],
            "zone_breakdown": score_info["zone_breakdown"],
            "model_trained": self.classifier.trained,
        }
        self.on_event(event)

    def run(self, audio_source) -> None:
        """Blocking call -- starts consuming from the given audio source
        (MicAudioSource or ArrayAudioSource, see audio_io.py).
        """
        audio_source.stream(self._handle_chunk)
