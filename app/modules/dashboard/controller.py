"""Dashboard controller — aggregated data."""

import time
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from storage.database import EventDatabase
from web_api.ws_manager import WebSocketManager

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def register(db: EventDatabase, ws_manager: WebSocketManager = None, camera_ids: list[str] = None):
    _camera_ids = camera_ids or ["cam-main"]

    @router.get("")
    async def get_dashboard():
        stats = db.stats()
        events = db.query(limit=10, offset=0)
        cam_info = [{"id": cid, "name": cid, "stream_url": f"/api/stream/{cid}",
                      "subscribers": ws_manager.subscriber_count(cid) if ws_manager else 0}
                    for cid in _camera_ids]
        return JSONResponse({
            "stats": stats, "recent_alerts": events, "cameras": cam_info,
            "system": {"total_cameras": len(_camera_ids), "total_alerts": stats.get("total", 0),
                       "timestamp": int(time.time())},
        })

    return router
