"""
Central configuration for the Acoustic Side-Channel Guard streaming pipeline.
Tune these values based on your actual mic/keyboard during testing.
"""

# --- Audio capture ---
SAMPLE_RATE = 16000          # Hz. 16k is plenty for keystroke transients (energy is
                              # concentrated below ~8kHz) and keeps compute light.
CHANNELS = 1
BLOCK_SIZE = 512              # samples per audio callback chunk (~32ms at 16kHz)

# Ring buffer length kept in memory for onset lookback/lookahead extraction
RING_BUFFER_SECONDS = 2.0

# --- Onset detection ---
ONSET_FRAME_MS = 10            # analysis frame size for energy/flux calc
ONSET_HOP_MS = 5                # hop between analysis frames
ONSET_HISTORY_FRAMES = 43       # ~ number of frames used for adaptive threshold (≈ 200ms at 5ms hop)
ONSET_THRESHOLD_K = 6.0         # threshold = local_mean + K * local_std
                                  # Tuned on synthetic clicks vs. quiet background noise
                                  # (see test_offline.py). RE-TUNE this against your actual
                                  # room/mic — real background noise (typing environment,
                                  # HVAC, etc.) has a different flux profile than synthetic
                                  # Gaussian noise and will likely need a different K.
ONSET_REFRACTORY_MS = 80        # minimum gap between two accepted onsets (avoids double-triggering
                                  # on a single key's press+release transient)

# --- Feature extraction window (relative to detected onset sample index) ---
PRE_ONSET_MS = 20     # capture a little audio before the detected onset
POST_ONSET_MS = 180   # and the decay/resonance tail after it
N_MELS = 40
N_FFT = 512
HOP_LENGTH = 128

# --- Classifier ---
# Coarse zone labels only (privacy-by-design choice — see project doc).
ZONE_LABELS = [
    "left_hand",
    "right_hand",
    "home_row",
    "space_enter_utility",
]
NUM_CLASSES = len(ZONE_LABELS)
MODEL_CHECKPOINT_PATH = None   # set to a .pt path once you've trained one

# --- Exposure scoring ---
EXPOSURE_WINDOW_EVENTS = 30     # rolling window size (number of recent keystroke events)
EXPOSURE_DECAY_SECONDS = 15.0   # events older than this contribute less (time-based decay)
