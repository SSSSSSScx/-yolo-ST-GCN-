"""Admin controller — rules, zones, config management."""

import os, yaml
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from storage.database import EventDatabase


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f: return yaml.safe_load(f) or {}


def _write_yaml(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def create_admin_router(db: EventDatabase, ws_manager=None, assessor=None,
                        rules_dir: str = "danger_assessor/rules",
                        camera_config_path: str = "camera_service/config.yaml",
                        camera_ids: list[str] = None,
                        stream_max_dim: int = 640):
    router = APIRouter(prefix="/admin", tags=["admin"])

    # ── Rules CRUD ──────────────────────────────────────────────
    @router.get("/action-rules")
    async def get_action_rules():
        return JSONResponse(_read_yaml(os.path.join(rules_dir, "action_rules.yaml")))

    @router.put("/action-rules")
    async def update_action_rules(data: dict = Body(...)):
        _write_yaml(os.path.join(rules_dir, "action_rules.yaml"), data)
        if assessor: assessor.reload_rules()
        return JSONResponse({"status": "ok"})

    @router.get("/object-rules")
    async def get_object_rules():
        return JSONResponse(_read_yaml(os.path.join(rules_dir, "object_rules.yaml")))

    @router.put("/object-rules")
    async def update_object_rules(data: dict = Body(...)):
        _write_yaml(os.path.join(rules_dir, "object_rules.yaml"), data)
        if assessor: assessor.reload_rules()
        return JSONResponse({"status": "ok"})

    @router.post("/reload")
    async def reload_configs():
        if assessor: assessor.reload_rules()
        return JSONResponse({"status": "ok"})

    @router.get("/ws-status")
    async def ws_status():
        return JSONResponse({"active_connections": ws_manager.active_count if ws_manager else 0})

    # ── Legacy global zones ─────────────────────────────────────
    @router.get("/zones")
    async def get_zones():
        return JSONResponse(_read_yaml(os.path.join(rules_dir, "zones.yaml")))

    @router.put("/zones")
    async def update_zones(data: dict = Body(...)):
        _write_yaml(os.path.join(rules_dir, "zones.yaml"), data)
        if assessor: assessor.reload_rules()
        return JSONResponse({"status": "ok"})

    # ── Per-camera zones ────────────────────────────────────────
    @router.get("/zones/{camera_id}")
    async def get_camera_zones(camera_id: str):
        if assessor:
            # Return stream_polygon (original stream coords) for frontend display
            zones = []
            for z in assessor._zone_manager.get_zones(camera_id):
                zones.append({
                    "id": z["id"], "name": z.get("name", z["id"]),
                    "type": z.get("type", "restricted_zone"),
                    "polygon": z.get("stream_polygon", z.get("polygon", [])),
                })
            return JSONResponse({"zones": zones})
        return JSONResponse({"zones": []})

    @router.post("/zones/{camera_id}")
    async def add_camera_zone(camera_id: str, data: dict = Body(...)):
        if not assessor:
            return JSONResponse({"status": "error", "message": "Assessor not available"})
        zone_id = data.get("id", "").strip()
        if not zone_id:
            return JSONResponse({"status": "error", "message": "Zone id required"})

        stream_polygon = data.get("polygon", [])
        # Convert stream polygon → original frame polygon
        # Stream is resized from original; scale = original_dim / stream_dim
        frame_polygon = _to_frame_coords(stream_polygon, stream_max_dim)

        assessor._zone_manager.add_zone(camera_id, {
            "id": zone_id, "name": data.get("name", zone_id),
            "type": data.get("type", "restricted_zone"),
            "polygon": frame_polygon,          # original frame coords (for alert checking)
            "stream_polygon": stream_polygon,   # stream coords (for frontend display)
            "description": data.get("description", ""),
        })
        return JSONResponse({"status": "ok"})

    @router.delete("/zones/{camera_id}/{zone_id}")
    async def delete_camera_zone(camera_id: str, zone_id: str):
        if not assessor:
            return JSONResponse({"status": "error"})
        ok = assessor._zone_manager.delete_zone(camera_id, zone_id)
        return JSONResponse({"status": "ok" if ok else "not_found"})

    return router


def _to_frame_coords(polygon: list, stream_max_dim: int) -> list:
    """Scale polygon from stream coordinates to original frame coordinates.

    Stream dimensions = resized from 1280x720 → max_dim on longer side.
    For 1280x720 with max_dim=480: scale_x = 1280/480 ≈ 2.67, scale_y = 720/480 ≈ 1.5
    But the actual scale depends on the resize ratio.
    We use a fixed original resolution of 1280x720 as default.
    """
    orig_w, orig_h = 1280, 720  # typical camera resolution
    if orig_w >= orig_h:
        scale = orig_w / stream_max_dim
    else:
        scale = orig_h / stream_max_dim

    result = []
    for pt in polygon:
        if len(pt) >= 2:
            result.append([int(pt[0] * scale), int(pt[1] * scale)])
        else:
            result.append(pt)
    return result
