import threading
import time
from typing import Optional
import numpy as np
import yaml
from loguru import logger
from .cameras.base import CameraInterface


class CameraManager:
    def __init__(self):
        self._cameras: dict[str, CameraInterface] = {}
        self._frames: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._running = False
        self._health_thread: Optional[threading.Thread] = None

    def load_config(self, config_path: str) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        from .cameras.usb_camera import USBCamera
        from .cameras.file_camera import FileCamera

        for cam_cfg in config.get("cameras", []):
            cam_type = cam_cfg.get("type", "usb")
            cam_id = cam_cfg["id"]

            if cam_type == "usb":
                camera = USBCamera(
                    device=str(cam_cfg.get("device", "0")),
                    resolution=tuple(cam_cfg.get("resolution", [1280, 720])),
                    fps=cam_cfg.get("fps", 30),
                )
            elif cam_type == "file":
                camera = FileCamera(
                    path=cam_cfg["path"],
                    loop=cam_cfg.get("loop", True),
                )
            else:
                logger.warning(f"Unknown camera type: {cam_type}, skipping")
                continue

            self.add_camera(cam_id, camera)

    def add_camera(self, camera_id: str, camera: CameraInterface) -> None:
        self._cameras[camera_id] = camera
        logger.info(f"Camera registered: {camera_id}")

    def add_and_start(self, camera_id: str, camera: CameraInterface) -> bool:
        """Dynamically add a camera and start its capture thread (hot-reload)."""
        if camera_id in self._cameras:
            logger.warning(f"Camera already exists: {camera_id}")
            return False
        if not camera.open():
            logger.error(f"Failed to open camera: {camera_id}")
            return False
        self._cameras[camera_id] = camera
        thread = threading.Thread(
            target=self._capture_loop,
            args=(camera_id, camera),
            daemon=True,
            name=f"cam-{camera_id}",
        )
        thread.start()
        self._threads.append(thread)
        logger.info(f"Camera hot-started: {camera_id}")
        return True

    def remove_camera(self, camera_id: str) -> bool:
        """Stop and remove a camera (hot-remove)."""
        camera = self._cameras.pop(camera_id, None)
        if camera is None:
            return False
        camera.close()
        with self._lock:
            self._frames.pop(camera_id, None)
        logger.info(f"Camera removed: {camera_id}")
        return True

    def start_all(self) -> None:
        self._running = True
        for cam_id, camera in self._cameras.items():
            if not camera.open():
                logger.error(f"Failed to open camera: {cam_id}")
                continue
            thread = threading.Thread(
                target=self._capture_loop,
                args=(cam_id, camera),
                daemon=True,
                name=f"cam-{cam_id}",
            )
            thread.start()
            self._threads.append(thread)

        self._health_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True,
            name="health-check",
        )
        self._health_thread.start()

    def _capture_loop(self, cam_id: str, camera: CameraInterface) -> None:
        while self._running:
            frame = camera.read_frame()
            if frame is not None:
                with self._lock:
                    self._frames[cam_id] = frame
            else:
                time.sleep(0.01)

    def _health_check_loop(self) -> None:
        while self._running:
            time.sleep(5)
            for cam_id, camera in self._cameras.items():
                if not camera.is_healthy():
                    logger.error(f"Camera unhealthy: {cam_id}, attempting reconnect...")
                    camera.close()
                    if camera.open():
                        logger.info(f"Camera reconnected: {cam_id}")
                    else:
                        logger.error(f"Camera reconnect failed: {cam_id}")

    def get_frames(self) -> dict[str, np.ndarray]:
        with self._lock:
            return dict(self._frames)

    def get_camera(self, camera_id: str) -> Optional[CameraInterface]:
        return self._cameras.get(camera_id)

    def stop_all(self) -> None:
        self._running = False
        for thread in self._threads:
            thread.join(timeout=2.0)
        if self._health_thread:
            self._health_thread.join(timeout=2.0)
        for camera in self._cameras.values():
            camera.close()
        self._threads.clear()
        logger.info("All cameras stopped")
