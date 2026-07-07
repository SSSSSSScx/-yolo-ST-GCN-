import time
from typing import Optional
import cv2
import numpy as np
from loguru import logger
from .base import CameraInterface


class FileCamera(CameraInterface):
    def __init__(self, path: str, loop: bool = True):
        self._path = path
        self._loop = loop
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 30.0
        self._frame_interval: float = 1.0 / 30.0
        self._last_frame_time: float = 0.0
        self._opened = False

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            logger.error(f"Failed to open video file: {self._path}")
            return False
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        if self._fps <= 0:
            self._fps = 30.0
        self._frame_interval = 1.0 / self._fps
        self._opened = True
        logger.info(f"File camera opened: path={self._path}, fps={self._fps:.1f}")
        return True

    def read_frame(self) -> Optional[np.ndarray]:
        if not self._opened or self._cap is None:
            return None
        now = time.time()
        elapsed = now - self._last_frame_time
        if elapsed < self._frame_interval:
            time.sleep(self._frame_interval - elapsed)
        ret, frame = self._cap.read()
        if not ret or frame is None:
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    return None
            else:
                return None
        self._last_frame_time = time.time()
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._opened = False
        logger.info(f"File camera closed: {self._path}")

    def get_info(self) -> dict:
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._cap and self._cap.isOpened() else 0
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._cap and self._cap.isOpened() else 0
        return {
            "id": f"file-{self._path}",
            "type": "file",
            "device": self._path,
            "resolution": [w, h],
            "fps": self._fps,
        }

    def is_healthy(self) -> bool:
        if not self._opened:
            return False
        return (time.time() - self._last_frame_time) < 3.0
