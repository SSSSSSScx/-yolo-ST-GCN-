"""Stream service - frame store + stream state management (reference: StreamService)."""

import threading
import time
from loguru import logger


class FrameStore:
    """Thread-safe store for the latest JPEG frame per camera."""

    def __init__(self):
        self._frames: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def put(self, cam_id: str, jpeg_bytes: bytes) -> None:
        with self._lock:
            self._frames[cam_id] = jpeg_bytes

    def get(self, cam_id: str) -> bytes | None:
        with self._lock:
            return self._frames.get(cam_id)


class StreamService:
    """Manages frame storage and stream state per camera.

    Reference: NestJS StreamService — provides frame relay + stream status aggregation.
    """

    def __init__(self):
        self._frame_store = FrameStore()
        self._last_frame_time: dict[str, float] = {}
        self._stream_online: dict[str, bool] = {}

    @property
    def frame_store(self) -> FrameStore:
        return self._frame_store

    def record_frame(self, cam_id: str, jpeg_bytes: bytes):
        self._frame_store.put(cam_id, jpeg_bytes)
        self._last_frame_time[cam_id] = time.time()
        if not self._stream_online.get(cam_id):
            self._stream_online[cam_id] = True
            logger.info(f"Stream online: {cam_id}")

    def check_health(self, timeout: float = 10.0) -> dict[str, str]:
        """Return status per camera. Marks offline if no frame for >timeout seconds."""
        now = time.time()
        result = {}
        for cam_id in list(self._last_frame_time.keys()):
            if now - self._last_frame_time[cam_id] > timeout:
                self._stream_online[cam_id] = False
            result[cam_id] = "online" if self._stream_online.get(cam_id) else "offline"
        return result

    def get_status(self, cam_id: str) -> str:
        return "online" if self._stream_online.get(cam_id, False) else "offline"

    def get_all_status(self) -> list[dict]:
        return [
            {"camera_id": cid, "status": st, "last_frame_time": self._last_frame_time.get(cid, 0)}
            for cid, st in self.check_health().items()
        ]
