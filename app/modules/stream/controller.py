"""Stream controller - MJPEG endpoint + status (reference: StreamController)."""

import asyncio
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, StreamingResponse
from .service import StreamService, FrameStore

router = APIRouter(prefix="/stream", tags=["stream"])


def register(service: StreamService, camera_ids: list[str], frame_store: FrameStore):
    @router.get("/mjpeg")
    async def mjpeg_stream():
        """GET /api/stream/mjpeg — MJPEG video stream proxy. Single-camera fallback."""
        async def generate():
            last_jpeg = None
            while True:
                for cid in camera_ids:
                    jpeg = frame_store.get(cid)
                    if jpeg and jpeg != last_jpeg:
                        last_jpeg = jpeg
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                               b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" +
                               jpeg + b"\r\n")
                        break
                await asyncio.sleep(0.016)
        return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

    @router.get("/status")
    async def stream_status():
        """GET /api/stream/status — all camera stream states."""
        return JSONResponse(service.get_all_status())

    @router.get("/{cam_id}")
    async def stream_camera(cam_id: str):
        """GET /api/stream/{cam_id} — MJPEG stream for a specific camera."""
        async def generate():
            last_jpeg = None
            while True:
                jpeg = frame_store.get(cam_id)
                if jpeg and jpeg != last_jpeg:
                    last_jpeg = jpeg
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" +
                           jpeg + b"\r\n")
                await asyncio.sleep(0.016)
        return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

    return router
