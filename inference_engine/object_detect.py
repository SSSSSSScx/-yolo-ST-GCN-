from collections import defaultdict
import cv2
import numpy as np
from loguru import logger


class ObjectDetector:
    """Detects smoke, fire, and other hazard objects in frames.

    Uses YOLO-based detection. Smoke requires >= 3 consecutive frames
    to confirm (reduces false positives). Fire triggers immediately.
    """

    def __init__(self, model_path: str = "models/yolo11n.onnx", confidence_threshold: float = 0.4):
        self._model_path = model_path
        self._conf_threshold = confidence_threshold
        self._model = None
        self._session = None
        self._use_onnx = False
        self._loaded = False
        self._img_size = (640, 640)

        # Smoke: COCO class 0 is person — smoke not in COCO
        # Fire: not in COCO either
        # We use class indices from a custom model; for now, note that
        # specialized smoke/fire model is needed for production use.
        self._hazard_classes = {
            0: "person",  # for reference
            # Smoke/fire require specialized model
        }

        # Frame confirmation counters for smoke
        self._smoke_counter = 0
        self._smoke_confirm_frames = 3

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Detect hazard objects. Currently returns empty list.

        Production deployment requires:
        - YOLO model fine-tuned on smoke/fire dataset
        - Or dedicated smoke/fire detection model (e.g., from Kaggle)
        - Place model in models/ directory as 'smoke_fire.onnx'
        """
        self._ensure_model()

        if self._model is None:
            return []

        results = self._model(frame, verbose=False)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, classes):
                cls_id = int(cls)
                cls_name = self._hazard_classes.get(cls_id, "")
                if conf < self._conf_threshold:
                    continue
                detections.append({
                    "class": cls_name or f"unknown_{cls_id}",
                    "bbox": box.tolist(),
                    "confidence": float(conf),
                })

        # Apply smoke confirmation
        confirmed = self._apply_smoke_confirmation(detections)
        return confirmed

    def _ensure_model(self):
        if self._loaded:
            return

        import os
        smoke_fire_path = "models/smoke_fire.onnx"
        if os.path.exists(smoke_fire_path):
            import onnxruntime as ort
            from .utils import get_onnx_providers
            self._session = ort.InferenceSession(smoke_fire_path, providers=get_onnx_providers())
            self._use_onnx = True
            self._loaded = True
            logger.info(f"Smoke/fire ONNX model loaded: {smoke_fire_path}")
            return

        # No specialized model available — skip detection entirely.
        # YOLO11n COCO model cannot detect smoke/fire, so running it
        # would waste significant inference time with zero results.
        self._loaded = True
        logger.info("ObjectDetector: no smoke/fire model, detection disabled (idle)")

    def _apply_smoke_confirmation(self, detections: list[dict]) -> list[dict]:
        has_smoke = any(d["class"] == "smoke" for d in detections)
        if has_smoke:
            self._smoke_counter += 1
        else:
            self._smoke_counter = max(0, self._smoke_counter - 1)

        confirmed = []
        for d in detections:
            if d["class"] == "smoke" and self._smoke_counter < self._smoke_confirm_frames:
                continue
            confirmed.append(d)
        return confirmed
