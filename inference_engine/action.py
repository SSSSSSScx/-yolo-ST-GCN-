"""Action recognition engine V3."""

import numpy as np
from loguru import logger
from .action_labels import ACTION_LABELS
from .action_v3 import ActionRecognizerV3


class ActionRecognizer:
    """V3 action recognition."""

    def __init__(self, window_size: int = 64, stride: int = 8,
                 model_path: str = "models/action_xgb.json"):
        self._engine = ActionRecognizerV3()
        logger.info("ActionRecognizer V3: ready (VideoMAE + 6 detectors)")

    def recognize(self, track_id: int, keypoints: np.ndarray,
                  bbox: list[float], fps: float = 30.0,
                  frame: np.ndarray = None) -> dict:
        if keypoints is None or keypoints.size == 0:
            return {"action": ACTION_LABELS[7], "action_id": 7, "confidence": 0.0}
        return self._engine.recognize(track_id, keypoints, fps,
                                       frame=frame, bbox=bbox)

    def reset_track(self, track_id: int) -> None:
        self._engine.reset_track(track_id)
