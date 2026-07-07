"""Alert controller — events CRUD + stats (reference: AlertController)."""

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from storage.database import EventDatabase
from .schemas import BatchDeleteDto

router = APIRouter(prefix="/events", tags=["alerts"])


def register(db: EventDatabase):
    @router.delete("")
    async def clear_all_events():
        count = db.clear_all()
        return JSONResponse({"status": "ok", "deleted": count})

    @router.get("/stats")
    async def get_stats():
        """Must be before /{event_id} to avoid path capture."""
        return JSONResponse(db.stats())

    @router.get("")
    async def list_events(limit: int = Query(50, ge=1, le=500),
                          offset: int = Query(0, ge=0),
                          min_level: int = Query(0, ge=0, le=3)):
        return JSONResponse(db.query(limit=limit, offset=offset, min_level=min_level))

    @router.get("/{event_id}")
    async def get_event(event_id: int):
        evt = db.get_by_id(event_id)
        if evt is None: raise HTTPException(status_code=404, detail="Event not found")
        return JSONResponse(evt)

    @router.delete("/{event_id}")
    async def delete_event(event_id: int):
        if not db.delete_event(event_id):
            raise HTTPException(status_code=404, detail="Event not found")
        return JSONResponse({"status": "ok"})

    @router.post("/batch-delete")
    async def batch_delete(dto: BatchDeleteDto):
        count = db.delete_events(dto.ids)
        return JSONResponse({"status": "ok", "deleted": count})

    return router
