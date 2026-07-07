"""Auth controller — login + user management (reference: AuthController)."""

import hashlib
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from storage.database import EventDatabase

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginDto(BaseModel): username: str; password: str


class AddUserDto(BaseModel):
    username: str; password: str; role: str = "admin"


def _hash(pw: str) -> str: return hashlib.sha256(pw.encode()).hexdigest()


def register(db: EventDatabase):
    @router.post("/login")
    async def login(dto: LoginDto):
        user = db.user_auth(dto.username, _hash(dto.password))
        if user is None: raise HTTPException(status_code=401, detail="Invalid credentials")
        return JSONResponse({"status": "ok", "user": user})

    @router.get("/users")
    async def list_users():
        return JSONResponse(db.user_list())

    @router.post("/users")
    async def add_user(dto: AddUserDto):
        if not dto.username.strip() or not dto.password.strip():
            raise HTTPException(status_code=400, detail="Username and password required")
        uid = db.user_add(dto.username.strip(), _hash(dto.password.strip()), dto.role)
        return JSONResponse({"status": "ok", "id": uid})

    @router.delete("/users/{user_id}")
    async def delete_user(user_id: int):
        if not db.user_delete(user_id):
            raise HTTPException(status_code=404, detail="User not found")
        return JSONResponse({"status": "ok"})

    return router
