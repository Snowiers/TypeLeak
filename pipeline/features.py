"""
Feature extraction: turns a short raw-audio window around a detected onset
into a log-mel spectrogram, the standard input representation for the CNN
classifier (this matches the representation used in the published acoustic
keystroke research this project is based on).
"""

from __future__ import annotations
import numpy as np
import torch
import torchaudio

import config


class MelFeatureExtractor:
    def __init__(self,
                 sample_rate: int = config.SAMPLE_RATE,
                 n_mels: int = config.N_MELS,
                 n_fft: int = config.N_FFT,
                 hop_length: int = config.HOP_LENGTH):
        self.sample_rate = sample_rate
        self._transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self._to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def extract(self, audio_window: np.ndarray) -> torch.Tensor:
        """audio_window: 1D float32 numpy array -> returns [1, n_mels, T] tensor,
        normalized to roughly zero-mean/unit-std for stable training/inference.
        """
        wav = torch.from_numpy(audio_window).float().unsqueeze(0)  # [1, samples]
        mel = self._transform(wav)          # [1, n_mels, T]
        mel_db = self._to_db(mel)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        return mel_db

    def snr_estimate(self, audio_window: np.ndarray, noise_floor_rms: float) -> float:
        """Rough signal-to-noise ratio estimate for this event, in dB.
        `noise_floor_rms` should come from a rolling ambient-noise tracker
        (see exposure.py) so this reflects *current* room conditions.
        """
        signal_rms = float(np.sqrt(np.mean(np.square(audio_window)) + 1e-12))
        noise_floor_rms = max(noise_floor_rms, 1e-6)
        return 20.0 * np.log10(signal_rms / noise_floor_rms)
