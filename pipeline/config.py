"""
Central configuration for the Acoustic Side-Channel Guard streaming pipeline.

IMPORTANT: everything under DETECTOR_CFG (audio/spectrogram/clip/detection/
output sections) is loaded DIRECTLY from config.yaml -- the same file used
by process.py and record.py to build the training dataset. Do NOT hardcode
duplicate copies of these values elsewhere; always read from DETECTOR_CFG
(or the derived constants below, which are just convenience unpacking of
the same dict) so training and live inference can never silently drift
apart. If you change config.yaml, these update automatically.

NOTE ON FILE LOCATIONS: detector.py, config.yaml, process.py, and train.py
all live in ~/Documents/keylogging -- a DIFFERENT directory from this
pipeline (~/Documents/testCode/spark-hacks/pipeline). Those files are not
copied in here and are never modified by this codebase; we only read
config.yaml and import detector.py from their real location, pointed at
below via KEYLOGGING_DIR. Change KEYLOGGING_DIR if that path moves.
"""
import os
import sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

# Where detector.py / config.yaml / process.py / train.py actually live --
# NOT this directory. Adjust here if the keylogging repo ever moves.
KEYLOGGING_DIR = os.path.expanduser("~/Documents/keylogging")

if not os.path.isdir(KEYLOGGING_DIR):
    raise FileNotFoundError(
        f"KEYLOGGING_DIR does not exist: {KEYLOGGING_DIR}\n"
        f"detector.py and config.yaml are expected there -- update "
        f"KEYLOGGING_DIR at the top of config.py if the keylogging repo "
        f"has moved."
    )

# Make `import detector` (used by onset_detector.py and features.py) find
# the real detector.py in KEYLOGGING_DIR, without copying the file here.
if KEYLOGGING_DIR not in sys.path:
    sys.path.insert(0, KEYLOGGING_DIR)

with open(os.path.join(KEYLOGGING_DIR, "config.yaml")) as f:
    DETECTOR_CFG = yaml.safe_load(f)

# --- Convenience unpacking of DETECTOR_CFG (for readability elsewhere in
# the codebase). These are DERIVED, not independently set -- see above. ---
SAMPLE_RATE = DETECTOR_CFG["audio"]["sample_rate"]
CHANNELS = DETECTOR_CFG["audio"]["channels"]
BLOCK_SIZE = DETECTOR_CFG["audio"]["blocksize"]

N_MELS = DETECTOR_CFG["spectrogram"]["n_mels"]
N_FFT = DETECTOR_CFG["spectrogram"]["n_fft"]
HOP_LENGTH = DETECTOR_CFG["spectrogram"]["hop_length"]
FMIN = DETECTOR_CFG["spectrogram"]["fmin"]
FMAX = DETECTOR_CFG["spectrogram"]["fmax"]

CLIP_MS = DETECTOR_CFG["clip"]["clip_ms"]
PRE_ONSET_MS = DETECTOR_CFG["clip"]["pre_onset_ms"]

FRAME_HOP = DETECTOR_CFG["detection"]["frame_hop"]
MIN_GAP_MS = DETECTOR_CFG["detection"]["min_gap_ms"]
SENSITIVITY = DETECTOR_CFG["detection"]["sensitivity"]
MIN_PEAK_AMPLITUDE = DETECTOR_CFG["detection"]["min_peak_amplitude"]
MIN_PREDICTION_CONFIDENCE = DETECTOR_CFG["detection"]["min_confidence"]

IMAGE_SIZE = DETECTOR_CFG["output"]["image_size"]

# --- Ring buffer / streaming ---
RING_BUFFER_SECONDS = 2.0

# --- Streaming onset re-analysis (see onset_detector.py) ---
# detect_onsets() from detector.py is a BATCH function (designed to run once
# over a whole recorded session in process.py). To use it live, we
# periodically re-run it over a recent rolling window of audio rather than
# incrementally -- these control that trade-off between latency and
# redundant computation.
ONSET_ANALYSIS_WINDOW_S = 1.0     # how far back each re-analysis looks
ONSET_REANALYSIS_INTERVAL_S = 0.1  # how often we re-run analysis on new audio

# --- Classifier ---
MODEL_CHECKPOINT_PATH = os.path.expanduser("~/Documents/keylogging/dataset/runs/20260816_054252/best_model.pt")   # set to your best_model.pt path once ready

# Fallback settings used ONLY when no real checkpoint is loaded (keeps the
# pipeline testable without one). A real checkpoint overrides all of these
# with what it actually stores.
FALLBACK_MODEL_NAME = "tf_efficientnetv2_s.in1k"
FALLBACK_IMAGE_SIZE = IMAGE_SIZE
FALLBACK_LABELS = list(DETECTOR_CFG["recording"]["accepted_keys"]) + [
    DETECTOR_CFG["junk"]["label"]
]

# --- Exposure scoring ---
EXPOSURE_WINDOW_EVENTS = 30
EXPOSURE_DECAY_SECONDS = 15.0

# --- Network audio (capture device -> GN100) ---
NETWORK_PORT = 9999
