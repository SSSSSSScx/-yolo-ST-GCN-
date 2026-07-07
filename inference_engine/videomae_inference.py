"""VideoMAE action recognition inference — lab safety fine-tuned model.

Loads the fine-tuned VideoMAE from models/videomae_lab_safety/
5 classes: normal, eating, fall, fight, smoking.
Input: 16-frame person-crop clips at 224x224.
"""

import os
import numpy as np
import torch
from collections import deque
from loguru import logger

from .action_labels import ACTION_LABELS

# VideoMAE 5-class → our 10-class mapping
VIDEOMAE_TO_LAB = {
    "normal": 7,     # → other (will be refined by rules)
    "eating": 4,     # → 饮食动作
    "fall": 5,       # → 摔倒 (rules refine to fallen=6)
    "fight": 9,      # → 推搡嬉闹
    "smoking": 8,    # → 抽烟
}


class VideoMAEInference:
    """VideoMAE action recognizer for lab safety.

    Collects person-crop frames, builds 16-frame clips, runs VideoMAE.
    """

    def __init__(self, model_path: str = "models/videomae_lab_safety"):
        self._model_path = model_path
        self._model = None
        self._processor = None
        self._device = None
        self._loaded = False
        self._load_failed = False

        # Per-track frame buffer (person crops)
        self._frame_buffers: dict[int, deque] = {}
        # Per-track last prediction
        self._last_pred: dict[int, dict] = {}

    @property
    def available(self) -> bool:
        if not self._loaded and not self._load_failed:
            self._load()
        return self._loaded

    def _load(self):
        """Lazy-load VideoMAE model."""
        if not os.path.isdir(self._model_path):
            logger.warning(f"VideoMAE model not found: {self._model_path}")
            self._load_failed = True
            return

        try:
            from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._processor = VideoMAEImageProcessor.from_pretrained(self._model_path)
            self._model = VideoMAEForVideoClassification.from_pretrained(
                self._model_path,
                torch_dtype=torch.float16 if self._device.type == "cuda" else torch.float32
            ).to(self._device).eval()

            self._loaded = True
            logger.info(f"VideoMAE loaded: 5-class lab safety model ({self._device})")
        except Exception as e:
            logger.error(f"VideoMAE load failed: {e}")
            self._load_failed = True

    def add_frame(self, track_id: int, person_crop: np.ndarray):
        """Add a person-crop frame to the track's buffer.

        Args:
            track_id: person track ID
            person_crop: BGR image (H, W, 3) of the person
        """
        if track_id not in self._frame_buffers:
            self._frame_buffers[track_id] = deque(maxlen=20)

        # Resize to 224x224
        import cv2
        crop = cv2.resize(person_crop, (224, 224))
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        self._frame_buffers[track_id].append(crop_rgb)

    def predict(self, track_id: int) -> dict | None:
        """Run VideoMAE inference if enough frames are collected.

        Returns: {"action": str, "action_id": int, "confidence": float} or None
        """
        if not self.available:
            return None

        buf = self._frame_buffers.get(track_id, deque())
        if len(buf) < 16:
            return None

        try:
            # Take last 16 frames
            frames = list(buf)[-16:]
            # Convert list of (224,224,3) numpy → (16, 224, 224, 3) numpy
            video = np.stack(frames)

            inputs = self._processor(list(video), return_tensors="pt")
            inputs = {k: v.to(device=self._device, dtype=self._model.dtype) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits[0].float().cpu().numpy()

            # Softmax
            probs = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
            pred_id = int(np.argmax(probs))
            confidence = float(probs[pred_id])

            # Map id → label → our action
            id2label = self._model.config.id2label
            label = id2label.get(pred_id, "normal")
            action_id = VIDEOMAE_TO_LAB.get(label, 7)

            result = {
                "action": ACTION_LABELS.get(action_id, "其他操作"),
                "action_id": action_id,
                "confidence": confidence,
                "videomae_label": label,
            }

            self._last_pred[track_id] = result
            return result

        except Exception as e:
            logger.error(f"VideoMAE inference error [track {track_id}]: {e}")
            return None

    def reset_track(self, track_id: int) -> None:
        self._frame_buffers.pop(track_id, None)
        self._last_pred.pop(track_id, None)
