import time
from typing import Optional
import cv2
import numpy as np
from loguru import logger
from .base import CameraInterface


class USBCamera(CameraInterface):
    def __init__(self, device: str = "0", resolution: tuple[int, int] = (1280, 720), fps: int = 30):
        self._device = device
        self._resolution = resolution
        self._fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_frame_time: float = 0.0
        self._opened = False

    def open(self) -> bool:
        try:
            device_index = int(self._device)
            self._cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
        except ValueError:
            self._cap = cv2.VideoCapture(self._device)

        if not self._cap.isOpened():
            logger.error(f"Failed to open USB camera: {self._device}")
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._opened = True
        logger.info(f"USB camera opened: device={self._device}, resolution={self._resolution}, fps={self._fps}")
        return True

    def read_frame(self) -> Optional[np.ndarray]:
        if not self._opened or self._cap is None:
            return None
        ret, frame = self._cap.read()
        if ret and frame is not None:
            self._last_frame_time = time.time()
            return frame
        return None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._opened = False
        logger.info(f"USB camera closed: {self._device}")

    def get_info(self) -> dict:
        return {
            "id": f"usb-{self._device}",
            "type": "usb",
            "device": self._device,
            "resolution": list(self._resolution),
            "fps": self._fps,
        }

    def is_healthy(self) -> bool:
        if not self._opened:
            return False
        return (time.time() - self._last_frame_time) < 3.0
