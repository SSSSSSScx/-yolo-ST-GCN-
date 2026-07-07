import os
import cv2
import numpy as np
from loguru import logger


class PersonDetector:
    """YOLO-based person detector.

    Model loading cascade:
    1. Load any .onnx file from models/ directory
    2. Try YOLOv12n, YOLO11n via ultralytics → export ONNX
    3. Use ultralytics PyTorch directly as final fallback
    """

    # Model cascade: try these in order for ultralytics auto-download
    _MODEL_CASCADE = ["yolov12n.pt", "yolo11n.pt"]

    def __init__(self, model_path: str = "models/yolov12n.onnx", confidence_threshold: float = 0.45):
        self._model_path = model_path
        self._conf_threshold = confidence_threshold
        self._low_conf_threshold = 0.3
        self._iou_threshold = 0.45
        self._session = None
        self._input_name = None
        self._use_onnx = False
        self._ultralytics_model = None
        self._img_size = (640, 640)

    def _find_onnx_model(self) -> str | None:
        if os.path.exists(self._model_path):
            return self._model_path
        models_dir = os.path.dirname(self._model_path) or "models"
        if os.path.isdir(models_dir):
            for f in sorted(os.listdir(models_dir)):
                if f.endswith(".onnx") and "pose" not in f:
                    return os.path.join(models_dir, f)
            for f in sorted(os.listdir(models_dir)):
                if f.endswith(".onnx"):
                    return os.path.join(models_dir, f)
        return None

    def _ensure_model(self) -> None:
        if self._session is not None or self._ultralytics_model is not None:
            return

        existing = self._find_onnx_model()
        if existing:
            self._model_path = existing
            self._load_onnx()
            return

        logger.info("No ONNX model found, attempting ultralytics download/export...")
        import ultralytics

        for pt_model in self._MODEL_CASCADE:
            try:
                logger.info(f"Trying {pt_model}...")
                model = ultralytics.YOLO(pt_model)
                self._ultralytics_model = model
                logger.info(f"Using {pt_model} (PyTorch) for inference")

                # Try exporting to ONNX for next run
                try:
                    os.makedirs(os.path.dirname(self._model_path) or "models", exist_ok=True)
                    onnx_path = self._model_path
                    model.export(format="onnx", imgsz=self._img_size[0], simplify=True)
                    exported = f"{pt_model[:-3]}.onnx"
                    if os.path.exists(exported):
                        import shutil
                        shutil.move(exported, onnx_path)
                        logger.info(f"ONNX model exported to {onnx_path}")
                except Exception as e:
                    logger.warning(f"ONNX export failed (non-fatal): {e}")
                return
            except Exception as e:
                logger.warning(f"Failed to load {pt_model}: {e}")
                continue

        raise RuntimeError(
            "Cannot load any YOLO model. Tried: " + ", ".join(self._MODEL_CASCADE) + ". "
            "Please place a YOLO .onnx file in models/ or ensure network access for ultralytics."
        )

    def _load_onnx(self) -> None:
        import onnxruntime as ort
        from .utils import get_onnx_providers
        self._session = ort.InferenceSession(self._model_path, providers=get_onnx_providers())
        self._input_name = self._session.get_inputs()[0].name
        self._input_shape = self._session.get_inputs()[0].shape
        self._use_onnx = True
        logger.info(f"ONNX model loaded: {self._model_path}")

    def detect(self, frame: np.ndarray) -> list[dict]:
        self._ensure_model()

        if self._use_onnx:
            return self._detect_onnx(frame)
        else:
            return self._detect_ultralytics(frame)

    def _detect_onnx(self, frame: np.ndarray) -> list[dict]:
        h, w = frame.shape[:2]
        input_tensor = self._preprocess(frame)
        outputs = self._session.run(None, {self._input_name: input_tensor})
        detections = outputs[0]
        return self._postprocess(detections, w, h)

    def _detect_ultralytics(self, frame: np.ndarray) -> list[dict]:
        results = self._ultralytics_model(frame, verbose=False)
        detections_list = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            for box, conf, cls in zip(boxes, confs, classes):
                if int(cls) != 0:
                    continue
                if conf >= self._low_conf_threshold:
                    detections_list.append({
                        "bbox": box.tolist(),
                        "confidence": float(conf),
                    })
        return self._nms(detections_list)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, self._img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def _postprocess(self, detections: np.ndarray, orig_w: int, orig_h: int) -> list[dict]:
        """Post-process ONNX output: (1, 84, 8400) → list of person detections.

        YOLO ONNX format: [cx, cy, w, h, class_0_prob, ..., class_79_prob]
        Convert cx,cy,w,h → x1,y1,x2,y2 and scale to original image size.
        """
        persons = []
        if detections.ndim == 3:
            detections = detections[0]  # (84, 8400)

        # Transpose to (8400, 84) for easier iteration
        if detections.shape[0] == 84:
            detections = detections.T  # (8400, 84)

        for det in detections:
            cx, cy, w, h = det[:4]
            class_probs = det[4:]
            if len(class_probs) == 0:
                continue
            max_conf = float(np.max(class_probs))
            class_id = int(np.argmax(class_probs))

            if class_id != 0:  # person class
                continue
            if max_conf < self._low_conf_threshold:
                continue

            # Convert center format to corner format
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            # Scale to original image dimensions
            scale_x = orig_w / self._img_size[0]
            scale_y = orig_h / self._img_size[1]
            x1 *= scale_x; y1 *= scale_y
            x2 *= scale_x; y2 *= scale_y

            persons.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(max_conf),
            })

        return self._nms(persons)

    def _nms(self, detections: list[dict]) -> list[dict]:
        if not detections:
            return []

        detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        kept = []
        suppressed = [False] * len(detections)

        def _iou(box_a, box_b):
            xa = max(box_a[0], box_b[0])
            ya = max(box_a[1], box_b[1])
            xb = min(box_a[2], box_b[2])
            yb = min(box_a[3], box_b[3])
            inter = max(0, xb - xa) * max(0, yb - ya)
            area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
            area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
            return inter / (area_a + area_b - inter + 1e-6)

        for i in range(len(detections)):
            if suppressed[i]:
                continue
            kept.append(detections[i])
            for j in range(i + 1, len(detections)):
                if suppressed[j]:
                    continue
                if _iou(detections[i]["bbox"], detections[j]["bbox"]) > self._iou_threshold:
                    suppressed[j] = True

        return kept
