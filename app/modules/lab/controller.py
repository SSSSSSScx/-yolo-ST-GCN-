"""Lab controller — CRUD (reference: LabController)."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from storage.database import EventDatabase

router = APIRouter(prefix="/labs", tags=["labs"])


class LabCreateDto(BaseModel):
    name: str; description: str = ""; cameras: list[str] = []


def register(db: EventDatabase):
    @router.get("")
    async def list_labs():
        return JSONResponse(db.lab_list())

    @router.post("")
    async def add_lab(dto: LabCreateDto):
        if not dto.name.strip(): raise HTTPException(status_code=400, detail="Name required")
        lid = db.lab_add(dto.name.strip(), dto.description, str(dto.cameras))
        return JSONResponse({"status": "ok", "id": lid})

    @router.delete("/{lab_id}")
    async def delete_lab(lab_id: int):
        if not db.lab_delete(lab_id): raise HTTPException(status_code=404, detail="Lab not found")
        return JSONResponse({"status": "ok"})

    return router
