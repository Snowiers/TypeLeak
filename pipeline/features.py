"""
Feature extraction -- now a thin wrapper around detector.py's REAL
extract_clip() and clip_to_melspec(), the exact same functions process.py
used to build the training dataset. No more guessed normalization/resize
logic -- this replaces the earlier placeholder version.
"""

from __future__ import annotations
import numpy as np
import torch

import config
import detector


class MelFeatureExtractor:
    def __init__(self, sample_rate: int = config.SAMPLE_RATE, cfg: dict = config.DETECTOR_CFG):
        self.sample_rate = sample_rate
        self.cfg = cfg
        self.image_size = cfg["output"]["image_size"]

    def build_clip(self, audio_containing_onset: np.ndarray, onset_sample_relative: int) -> np.ndarray:
        """audio_containing_onset: a 1D array that CONTAINS the onset at the
        given relative sample index (with enough context before/after --
        detector.extract_clip() zero-pads automatically if it doesn't).
        Returns the fixed-length raw audio clip (same as training).
        """
        return detector.extract_clip(audio_containing_onset, onset_sample_relative,
                                      self.sample_rate, self.cfg)

    def extract(self, clip: np.ndarray) -> torch.Tensor:
        """clip: fixed-length raw audio clip from build_clip().
        Returns a [3, image_size, image_size] tensor ready for KeyClassifier.predict()
        -- mel-spectrogram via detector.clip_to_melspec() (IDENTICAL to training),
        then repeated to 3 channels (matches train.py's KeystrokeDataset exactly).
        """
        spec = detector.clip_to_melspec(clip, self.sample_rate, self.cfg)  # [image_size, image_size], float32 in [0,1]
        t = torch.from_numpy(spec).unsqueeze(0).repeat(3, 1, 1)  # [3, image_size, image_size]
        return t

    def snr_estimate(self, audio_window: np.ndarray, noise_floor_rms: float) -> float:
        """Rough signal-to-noise ratio estimate for this event, in dB.
        Independent of the model-input pipeline above -- used only for
        exposure scoring, not classification.
        """
        signal_rms = float(np.sqrt(np.mean(np.square(audio_window)) + 1e-12))
        noise_floor_rms = max(noise_floor_rms, 1e-6)
        return 20.0 * np.log10(signal_rms / noise_floor_rms)
