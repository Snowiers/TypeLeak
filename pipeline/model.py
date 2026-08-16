"""
Keystroke key classifier.

IMPORTANT: this replaces the original hand-written KeystrokeZoneCNN. The
actual checkpoint produced by train.py uses a pretrained EfficientNetV2
(via `timm`), predicting individual keys (37 classes: 0-9, a-z, 'junk'),
not the 4 coarse zones originally planned. See project notes for the
resulting privacy-framing discussion -- this is a real capability change,
not just an implementation detail.

The checkpoint format (from train.py) is self-describing:
    {
        "model_state": <state_dict>,
        "model_name": "tf_efficientnetv2_s.in1k",   # timm model name
        "idx_to_label": {0: "0", 1: "1", ..., 20: "junk", ...},
        "image_size": 128,
        "val_acc": 0.8695652173913043,
    }
This wrapper reads model_name/idx_to_label/image_size directly from the
checkpoint, so it adapts automatically to any future retrain -- no
hardcoded architecture or label list here.
"""

from __future__ import annotations
import os
import torch
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None  # only required once you actually load a real checkpoint

import config


class KeyClassifier:
    """Inference-time wrapper around the trained EfficientNetV2 keystroke
    classifier. Handles device placement and turns logits into a labeled,
    confidence-scored result -- mirrors the old ZoneClassifier's interface
    (predict() returns (label, confidence, probs)) so pipeline.py needs
    minimal changes.
    """

    def __init__(self, checkpoint_path: str | None = config.MODEL_CHECKPOINT_PATH,
                 device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.trained = False
        self.idx_to_label: dict[int, str] = {}
        self.image_size: int = config.FALLBACK_IMAGE_SIZE
        self.model_name: str = config.FALLBACK_MODEL_NAME

        if checkpoint_path:
            self._load_real_checkpoint(checkpoint_path)
        else:
            self._build_untrained_fallback()

        self.model.eval()

    def _build_untrained_fallback(self) -> None:
        """No checkpoint given -- build an untrained model so the rest of
        the pipeline (onset detection, feature extraction, exposure
        scoring) is still fully testable without a trained model. Uses
        the same architecture family as real checkpoints, just random
        weights, and a placeholder label set from config.
        """
        if timm is None:
            raise RuntimeError(
                "timm is required (even for the untrained fallback, to match "
                "the real model's architecture). Install with: pip install timm"
            )
        self.idx_to_label = {i: lbl for i, lbl in enumerate(config.FALLBACK_LABELS)}
        self.model = timm.create_model(
            self.model_name, pretrained=False, num_classes=len(self.idx_to_label)
        ).to(self.device)
        self.trained = False

    def _load_real_checkpoint(self, path: str) -> None:
        if timm is None:
            raise RuntimeError("timm is required to load this checkpoint. Install with: pip install timm")

        path = os.path.expanduser(path)
        ckpt = torch.load(path, map_location=self.device)

        # be tolerant of either our custom dict format or a plain state_dict,
        # in case someone points this at a differently-shaped checkpoint later
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state = ckpt["model_state"]
            self.model_name = ckpt.get("model_name", self.model_name)
            raw_idx_to_label = ckpt.get("idx_to_label", None)
            self.image_size = ckpt.get("image_size", self.image_size)
            if raw_idx_to_label is None:
                raise ValueError(
                    f"Checkpoint at {path} has no 'idx_to_label' -- can't determine "
                    f"num_classes or label names. Inspect it with inspect_checkpoint.py."
                )
            # JSON/torch.load sometimes gives int keys, sometimes str keys -- normalize to int
            self.idx_to_label = {int(k): v for k, v in raw_idx_to_label.items()}
        else:
            raise ValueError(
                f"Checkpoint at {path} doesn't look like train.py's format "
                f"(expected a dict with a 'model_state' key). Got: "
                f"{type(ckpt).__name__}"
                + (f" with keys {list(ckpt.keys())}" if isinstance(ckpt, dict) else "")
                + ". Run inspect_checkpoint.py on it to see what's actually inside."
            )

        num_classes = len(self.idx_to_label)
        self.model = timm.create_model(
            self.model_name, pretrained=False, num_classes=num_classes
        ).to(self.device)
        self.model.load_state_dict(state)
        self.trained = True
        print(f"[KeyClassifier] loaded {self.model_name} checkpoint: "
              f"{num_classes} classes, image_size={self.image_size}, "
              f"val_acc={ckpt.get('val_acc', 'unknown')}")

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor) -> tuple[str, float, torch.Tensor]:
        """image_tensor: [3, H, W] tensor, H=W=self.image_size, already
        preprocessed to match training (see features.py -- this is the
        part still pending exact replication of process.py).
        Returns (predicted_label, confidence, full_probability_vector).
        """
        x = image_tensor.unsqueeze(0).to(self.device)  # [1, 3, H, W]
        logits = self.model(x)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu()
        idx = int(torch.argmax(probs).item())
        return self.idx_to_label[idx], float(probs[idx]), probs
