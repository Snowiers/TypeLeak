#!/usr/bin/env python3
"""
detector.py  --  Unsupervised acoustic keystroke detection + clip -> mel-spectrogram.

This is the SHARED core used in two places:
  * process.py  -> detects onsets in recorded audio, which are then labelled by
                   matching to pynput timestamps (training-data collection).
  * (inference) -> the SAME detect_onsets() finds keystrokes live from the mic,
                   with NO labels, and the trained model predicts each clip.

Improvements over the paper's extractor:
  - proper onset-strength ENVELOPE (not a rectified-amplitude proxy),
  - adaptive PEAK-PICKING on a 0..1-normalized envelope (no magic absolute
    threshold, no "tune until you get exactly N keystrokes" loop that can hang),
  - a minimum inter-keystroke gap so one press isn't double-counted.
"""
import numpy as np
import librosa
from scipy.signal import find_peaks
from PIL import Image


def _ms_to_frames(ms, sr, hop):
    return max(1, int(ms / 1000.0 * sr / hop))


def onset_envelope(audio, sr, hop):
    """0..1-normalized onset-strength envelope (one value per frame of `hop` samples)."""
    env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop)
    m = float(env.max())
    if m > 0:
        env = env / m
    return env


def detect_onsets(audio, sr, cfg):
    """
    Return sample-indices of keystroke onsets, using an ABSOLUTE amplitude
    envelope (NOT normalized by the buffer's max). This makes detection
    buffer-INDEPENDENT: the same keystroke is found at the same position whether
    it sits in a 2s live buffer or the whole training file -> live extraction
    matches training extraction, and there's no per-buffer normalization to
    cause the session-to-session inconsistency. It's also much cheaper than
    FFT onset-strength, which keeps the live server real-time.
    """
    d = cfg["detection"]
    if len(audio) < 8:
        return []
    min_amp = float(d.get("min_peak_amplitude", 0.02))
    min_gap = max(1, int(d.get("min_gap_ms", 60) / 1000.0 * sr))
    # smoothed absolute-amplitude envelope (~4ms moving average)
    win = max(1, int(0.004 * sr))
    env = np.convolve(np.abs(audio), np.ones(win, dtype=np.float32) / win, mode="same")
    peaks, _ = find_peaks(env, height=min_amp, distance=min_gap)
    return [int(p) for p in peaks]   # sample indices of loud amplitude peaks


def extract_clip(audio, onset_sample, sr, cfg):
    """Fixed-length clip centered so the onset sits `pre_onset_ms` into it (zero-padded)."""
    clip_len = int(cfg["clip"]["clip_ms"] / 1000.0 * sr)
    pre = int(cfg["clip"]["pre_onset_ms"] / 1000.0 * sr)
    start = onset_sample - pre
    end = start + clip_len
    clip = np.zeros(clip_len, dtype=np.float32)
    src_lo = max(0, start)
    src_hi = min(len(audio), end)
    if src_hi > src_lo:
        clip[src_lo - start: src_lo - start + (src_hi - src_lo)] = audio[src_lo:src_hi]
    return clip


def clip_to_melspec(clip, sr, cfg):
    """Clip -> normalized square mel-spectrogram (float32 in [0,1])."""
    s = cfg["spectrogram"]
    mel = librosa.feature.melspectrogram(
        y=clip, sr=sr, n_fft=s["n_fft"], hop_length=s["hop_length"],
        n_mels=s["n_mels"], fmin=s["fmin"], fmax=s["fmax"],
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mn, mx = mel_db.min(), mel_db.max()
    mel_norm = (mel_db - mn) / (mx - mn + 1e-8)
    size = cfg["output"]["image_size"]
    img = Image.fromarray((mel_norm * 255).astype(np.uint8)).resize(
        (size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0
