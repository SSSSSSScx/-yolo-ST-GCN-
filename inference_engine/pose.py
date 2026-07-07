import cv2
import numpy as np
from loguru import logger


class PoseEstimator:
    """Pose estimator using ultralytics YOLO11-Pose.

    Falls back to heuristic-only mode if model unavailable.
    """

    def __init__(self, model_path: str = "models/yolo11n-pose.onnx"):
        self._model_path = model_path
        self._model = None
        self._use_onnx = False
        self._session = None
        self._input_name = None
        self._img_size = (640, 640)
        self._loaded = False

    def _ensure_model(self) -> None:
        if self._loaded:
            return

        import os
        if os.path.exists(self._model_path):
            self._load_onnx()
            return

        try:
            from ultralytics import YOLO
            logger.info("Loading YOLO11-Pose from ultralytics...")
            self._model = YOLO("yolo11n-pose.pt")
            self._loaded = True
            logger.info("YOLO11-Pose loaded (PyTorch)")

            # Try exporting to ONNX
            try:
                os.makedirs(os.path.dirname(self._model_path) or "models", exist_ok=True)
                self._model.export(format="onnx", imgsz=self._img_size[0], simplify=True)
                if os.path.exists("yolo11n-pose.onnx"):
                    import shutil
                    shutil.move("yolo11n-pose.onnx", self._model_path)
                    logger.info(f"Pose ONNX exported to {self._model_path}")
            except Exception as e:
                logger.warning(f"Pose ONNX export failed (non-fatal): {e}")
        except Exception as e:
            logger.warning(f"Cannot load pose model: {e}. Pose estimation disabled.")
            self._model = None
            self._loaded = True

    def _load_onnx(self) -> None:
        import onnxruntime as ort
        from .utils import get_onnx_providers
        self._session = ort.InferenceSession(self._model_path, providers=get_onnx_providers())
        self._input_name = self._session.get_inputs()[0].name
        self._use_onnx = True
        self._loaded = True
        logger.info(f"Pose ONNX model loaded: {self._model_path}")

    def estimate(self, frame: np.ndarray, person_bboxes: list[list]) -> list[np.ndarray]:
        """Return list of (17, 3) keypoint arrays, one per person bbox."""
        self._ensure_model()

        if self._model is None and self._session is None:
            return [np.zeros((17, 3)) for _ in person_bboxes]

        if self._use_onnx:
            return self._estimate_onnx(frame, person_bboxes)
        else:
            return self._estimate_ultralytics(frame, person_bboxes)

    def _estimate_ultralytics(self, frame, person_bboxes):
        results = self._model(frame, verbose=False)
        keypoints_list = []

        for result in results:
            if result.keypoints is None:
                continue
            kpts = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
            boxes = result.boxes
            if boxes is None:
                for k in kpts:
                    keypoints_list.append(k)
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            for bbox in person_bboxes:
                best_k = np.zeros((17, 3))
                best_iou = 0
                for k, box in zip(kpts, xyxy):
                    iou = self._box_iou(bbox, box.tolist())
                    if iou > best_iou:
                        best_iou = iou
                        best_k = k
                keypoints_list.append(best_k)

        while len(keypoints_list) < len(person_bboxes):
            keypoints_list.append(np.zeros((17, 3)))

        return keypoints_list[:len(person_bboxes)]

    def _estimate_onnx(self, frame, person_bboxes):
        h, w = frame.shape[:2]
        all_kpts = []
        for bbox in person_bboxes:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            bw, bh = x2 - x1, y2 - y1
            # Expand 20%
            x1 = max(0, int(x1 - 0.2 * bw))
            y1 = max(0, int(y1 - 0.2 * bh))
            x2 = min(w, int(x2 + 0.2 * bw))
            y2 = min(h, int(y2 + 0.2 * bh))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                all_kpts.append(np.zeros((17, 3)))
                continue

            img = cv2.resize(crop, self._img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)

            out = self._session.run(None, {self._input_name: img})
            # YOLO-pose ONNX output: (1, 56, 8400)
            # 56 = 4(bbox) + 1(conf) + 17*3(kpts)
            pred = out[0]  # (56, 8400)
            pred = pred.T   # (8400, 56)

            # Find best detection by confidence
            scores = pred[:, 4]
            best_idx = int(np.argmax(scores))

            # Keypoints at columns 5:56 (51 values = 17×3)
            kpts_flat = pred[best_idx, 5:56]
            kpts = kpts_flat.reshape(17, 3).copy()

            # Scale keypoints back to original crop coordinates
            scale_x = bw / self._img_size[0]
            scale_y = bh / self._img_size[1]
            kpts[:, 0] = kpts[:, 0] * scale_x + x1
            kpts[:, 1] = kpts[:, 1] * scale_y + y1
            all_kpts.append(kpts.astype(np.float32))

        return all_kpts

    @staticmethod
    def _box_iou(a, b):
        xa = max(a[0], b[0]); ya = max(a[1], b[1])
        xb = min(a[2], b[2]); yb = min(a[3], b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter + 1e-6)
