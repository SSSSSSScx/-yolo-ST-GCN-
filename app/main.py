"""App bootstrap — NestJS-style module assembly (reference: main.ts + AppModule).

Assembles all modules, mounts controllers, starts uvicorn.
"""

import threading
import asyncio
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from loguru import logger

from .config import config
from .modules.stream.service import StreamService
from .modules.stream.controller import register as register_stream
from .modules.alert.controller import register as register_alert
from .modules.camera.controller import register as register_camera
# Auth module disabled — login not required
# from .modules.auth.controller import register as register_auth
from .modules.lab.controller import register as register_lab
from .modules.dashboard.controller import register as register_dashboard
from storage.database import EventDatabase
from web_api.ws_manager import WebSocketManager


def create_app(db: EventDatabase, ws_manager: WebSocketManager = None,
               assessor=None, records_dir: str = "records",
               frame_store=None, camera_ids: list[str] = None,
               stream_service: StreamService = None,
               camera_reload_callback=None) -> FastAPI:
    """Create and configure the FastAPI application with all modules."""
    _camera_ids = camera_ids or ["cam-main"]

    app = FastAPI(title="Lab Warning System API", version="1.0.0")

    # CORS
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"], allow_credentials=True)

    # === Stream Module ===
    if stream_service and frame_store:
        stream_router = register_stream(stream_service, _camera_ids, frame_store)
        app.include_router(stream_router, prefix="/api")

    # === Alert Module (events CRUD) ===
    alert_router = register_alert(db)
    app.include_router(alert_router, prefix="/api")

    # === Camera Module ===
    camera_router = register_camera(_camera_ids, config.camera_config_path,
                                     reload_callback=camera_reload_callback)
    app.include_router(camera_router, prefix="/api")

    # === Auth Module (disabled — login not required) ===
    # auth_router = register_auth(db)
    # app.include_router(auth_router, prefix="/api")

    # === Lab Module ===
    lab_router = register_lab(db)
    app.include_router(lab_router, prefix="/api")

    # === Dashboard Module ===
    dash_router = register_dashboard(db, ws_manager, _camera_ids)
    app.include_router(dash_router, prefix="/api")

    # === Admin module (rules CRUD) ===
    from .modules.admin_controller import create_admin_router
    admin_router = create_admin_router(db, ws_manager, assessor, config.rules_dir, config.camera_config_path, _camera_ids, stream_max_dim=480)
    app.include_router(admin_router, prefix="/api")

    # === WebSocket endpoint ===
    @app.websocket("/ws")
    async def ws_endpoint(ws):
        from fastapi import WebSocket, WebSocketDisconnect
        if ws_manager is None:
            await ws.close()
            return
        client_id = await ws_manager.connect(ws)
        try:
            await ws_manager.handle_client_messages(client_id, ws)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            ws_manager.disconnect(client_id)

    # === Health check ===
    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "lab-warning-backend", "version": "1.0.0"}

    # === Static files ===
    import os
    from fastapi.responses import Response
    static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web_api", "static")
    if os.path.isdir(static_dir):
        def _serve_index():
            return FileResponse(
                os.path.join(static_dir, "index.html"),
                headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                         "Pragma": "no-cache", "Expires": "0"})

        @app.get("/")
        async def index():
            return _serve_index()
        @app.get("/monitor")
        async def monitor():
            return _serve_index()
        @app.get("/admin")
        async def admin():
            return _serve_index()
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # === Video serving (from database BLOB — JPEG frame pack) ===
    @app.get("/api/video/{event_id}")
    async def serve_video(event_id: int):
        from fastapi.responses import Response
        from fastapi import HTTPException
        video_data = db.get_video(event_id)
        if not video_data:
            raise HTTPException(status_code=404, detail="该告警暂无录像")
        return Response(
            content=video_data,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"inline; filename=event_{event_id}.bin",
                     "Accept-Ranges": "bytes"})

    return app


class WebServer:
    """Background web server (reference: NestJS bootstrap in main.ts)."""

    def __init__(self, db, ws_manager=None, assessor=None,
                 records_dir="records", frame_store=None,
                 camera_ids=None, host="0.0.0.0", port=8080,
                 camera_reload_callback=None):
        self._db = db
        self._ws_manager = ws_manager
        self._assessor = assessor
        self._records_dir = records_dir
        self._frame_store = frame_store
        self._camera_ids = camera_ids or ["cam-main"]
        self._host = host
        self._port = port
        self._thread = None
        self._stream_service = StreamService()
        self._camera_reload_callback = camera_reload_callback

    def start(self):
        app = create_app(self._db, ws_manager=self._ws_manager,
                         assessor=self._assessor, records_dir=self._records_dir,
                         frame_store=self._frame_store, camera_ids=self._camera_ids,
                         stream_service=self._stream_service,
                         camera_reload_callback=self._camera_reload_callback)
        self._thread = threading.Thread(target=self._run, args=(app,), daemon=True)
        self._thread.start()
        logger.info(f"============================================")
        logger.info(f"  HTTP:     http://{self._host}:{self._port}")
        logger.info(f"  Stream:   http://localhost:{self._port}/api/stream/<id>")
        logger.info(f"  Monitor:  http://localhost:{self._port}/monitor")
        logger.info(f"  Health:   http://localhost:{self._port}/health")
        logger.info(f"============================================")

    def _run(self, app):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        if self._ws_manager:
            self._ws_manager.set_event_loop(loop)
        srv = uvicorn.Server(uvicorn.Config(app, host=self._host, port=self._port, log_level="warning"))
        loop.run_until_complete(srv.serve())
