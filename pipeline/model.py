"""
Acoustic keystroke-zone classifier.

Small CNN over log-mel spectrograms — deliberately lightweight so it runs
fast enough for real-time inference. Predicts a coarse keyboard ZONE
(config.ZONE_LABELS), not individual keys — see project doc for why that's
a deliberate privacy-by-design choice, not a technical ceiling.

Works untrained (random weights) out of the box so the rest of the pipeline
is fully testable before you've trained anything. Call `load_checkpoint`
once you have real trained weights (see train.py, to be added once you have
labeled data).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class KeystrokeZoneCNN(nn.Module):
    def __init__(self, n_mels: int = config.N_MELS, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, n_mels, T]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.global_pool(x).flatten(1)
        return self.fc(x)  # logits, [B, num_classes]


class ZoneClassifier:
    """Inference-time wrapper: handles device placement, batching a single
    spectrogram, and turning logits into a labeled, confidence-scored result.
    """

    def __init__(self, checkpoint_path: str | None = config.MODEL_CHECKPOINT_PATH,
                 device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = KeystrokeZoneCNN().to(self.device)
        self.trained = False
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        self.model.eval()

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.trained = True

    @torch.no_grad()
    def predict(self, mel_spectrogram: torch.Tensor) -> tuple[str, float, torch.Tensor]:
        """mel_spectrogram: [1, n_mels, T] tensor (from MelFeatureExtractor).
        Returns (predicted_zone_label, confidence, full_probability_vector).
        """
        x = mel_spectrogram.unsqueeze(0).to(self.device)  # [1, 1, n_mels, T]
        logits = self.model(x)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu()
        idx = int(torch.argmax(probs).item())
        return config.ZONE_LABELS[idx], float(probs[idx]), probs
