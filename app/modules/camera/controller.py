"""Camera controller — list + config CRUD with hot-reload support."""

import yaml
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter(prefix="/cameras", tags=["cameras"])


class CameraCreateDto(BaseModel):
    id: str
    name: str = ""
    type: str = "usb"
    device: str = "0"
    resolution: list[int] = [1280, 720]
    fps: int = 30


def register(camera_ids: list[str], config_path: str, reload_callback=None):
    """Register camera routes.

    reload_callback: optional callable(camera_id, camera_cfg, action)
        action = "add" | "delete" — called after config is saved so the
        running system can hot-reload the camera without restart.
    """

    @router.get("")
    async def list_cameras():
        return JSONResponse([{"id": cid, "name": cid, "stream_url": f"/api/stream/{cid}"}
                             for cid in camera_ids])

    @router.get("/config")
    async def get_config():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return JSONResponse(yaml.safe_load(f) or {})
        except Exception:
            return JSONResponse({"cameras": []})

    @router.post("/config")
    async def add_camera(dto: CameraCreateDto):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {"cameras": []}
        cams = cfg.get("cameras", [])
        if any(c.get("id") == dto.id for c in cams):
            raise HTTPException(status_code=409, detail="Camera ID already exists")
        cam_cfg = {"id": dto.id, "name": dto.name or dto.id, "type": dto.type,
                    "device": dto.device, "resolution": dto.resolution, "fps": dto.fps}
        cams.append(cam_cfg)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump({"cameras": cams}, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        # Hot-reload: notify running system
        if reload_callback:
            reload_callback(dto.id, cam_cfg, "add")

        return JSONResponse({"status": "ok", "id": dto.id})

    @router.delete("/config/{cam_id}")
    async def delete_camera(cam_id: str):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {"cameras": []}
        cams = cfg.get("cameras", [])
        before = len(cams)
        cams = [c for c in cams if c.get("id") != cam_id]
        if len(cams) == before:
            raise HTTPException(status_code=404, detail="Camera not found")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump({"cameras": cams}, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        # Hot-reload: notify running system
        if reload_callback:
            reload_callback(cam_id, None, "delete")

        return JSONResponse({"status": "ok"})

    return router
