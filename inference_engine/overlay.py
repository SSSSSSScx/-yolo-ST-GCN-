import cv2
import numpy as np
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from .action_labels import ACTION_LABELS

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simsun.ttc",
]

# ── color palettes (BGR for OpenCV, RGB for PIL) ──
_ACTION_BGR = {
    0: (80, 200, 60),   1: (80, 200, 60),   2: (80, 200, 60),   # posture — green
    3: (0, 180, 220),   4: (0, 140, 240),   5: (60, 60, 255),   # run/eat/fall
    6: (60, 60, 255),   7: (140, 140, 160), 8: (0, 140, 240),   # fallen/other/smoke
    9: (0, 180, 220),                                              # pushing — yellow
}
_LEVEL_BGR = {0: (80, 200, 60), 1: (0, 180, 220), 2: (0, 100, 240), 3: (60, 60, 255)}
_LEVEL_BG_BGR = {0: (20, 60, 20), 1: (20, 50, 80), 2: (20, 30, 100), 3: (30, 0, 50)}


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """Convert '#RRGGBB' to BGR tuple for OpenCV."""
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return (b, g, r)


def _action_id_from_name(name: str) -> int:
    for k, v in ACTION_LABELS.items():
        if v == name:
            return k
    return 7


class OverlayRenderer:
    """Modern detection-box overlay with rounded corners and pill labels.

    Draws directly with OpenCV (no PIL dependency at runtime).
    Supports Chinese text via PIL if available, otherwise falls back to pinyin.
    """

    def __init__(self, use_pil: bool = True):
        self._use_pil = use_pil
        self._pil_ok = use_pil
        self._font_cache: dict[int, ImageFont.FreeTypeFont] = {}

    # ── public API ──

    def draw(self, frame: np.ndarray, pipeline_output: dict) -> np.ndarray:
        """Render bounding boxes + labels onto the frame."""
        persons = pipeline_output.get("persons", [])
        alerts = pipeline_output.get("alerts", [])

        # Resolve per-track danger level
        track_levels: dict[int, int] = {}
        for a in alerts:
            tid = a.get("track_id", -1)
            lv = a.get("level", 0)
            if tid not in track_levels or lv > track_levels[tid]:
                track_levels[tid] = lv

        result = frame.copy()

        if self._pil_ok:
            try:
                result = self._draw_pil(result, persons, track_levels)
            except Exception as e:
                logger.warning(f"PIL overlay failed ({e}), using OpenCV")
                self._pil_ok = False
                result = self._draw_opencv(frame.copy(), persons, track_levels)
        else:
            result = self._draw_opencv(result, persons, track_levels)

        return result

    # ── PIL path ──

    def _draw_pil(self, frame: np.ndarray, persons: list[dict],
                  track_levels: dict[int, int]) -> np.ndarray:
        h, w = frame.shape[:2]
        font_size = max(24, h // 22)
        font = self._font_cache.get(font_size)
        if font is None:
            font = _get_font(font_size)
            self._font_cache[font_size] = font

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)

        for p in persons:
            self._draw_person_pil(draw, p, track_levels, font, h)

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def _draw_person_pil(self, draw, p: dict, track_levels: dict,
                         font, frame_h: int) -> None:
        bbox = p["bbox"]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 10:
            return

        dl = track_levels.get(p["track_id"], 0)
        action = p.get("action", "")
        tid = p.get("track_id", -1)
        action_id = _action_id_from_name(action)

        # Color
        if dl > 0:
            fill_rgb = {1: (50, 140, 40), 2: (30, 90, 160), 3: (60, 20, 40)}.get(dl, (30, 70, 30))
            line_rgb = {1: (0, 170, 200), 2: (0, 90, 230), 3: (50, 40, 230)}.get(dl, (80, 200, 60))
        else:
            bgr = _ACTION_BGR.get(action_id, (80, 200, 60))
            line_rgb = (bgr[2], bgr[1], bgr[0])
            fill_rgb = tuple(max(0, c // 5) for c in line_rgb)

        # Main outline (double-line for depth) — no fill, keep person visible
        draw.rectangle([x1 - 1, y1 - 1, x2 + 1, y2 + 1], outline=fill_rgb, width=1)
        draw.rectangle([x1, y1, x2, y2], outline=line_rgb, width=2)

        # Corner accents (L-shapes) — brighter, thicker
        corner_len = min(25, bw // 4, bh // 4)
        accent = tuple(min(255, c + 80) for c in line_rgb)
        for cx, cy, dx, dy in [
            (x1, y1, 1, 1), (x2, y1, -1, 1),
            (x1, y2, 1, -1), (x2, y2, -1, -1),
        ]:
            draw.line([(cx, cy), (cx + dx * corner_len, cy)], fill=accent, width=3)
            draw.line([(cx, cy), (cx, cy + dy * corner_len)], fill=accent, width=3)

        # Label pill — top-left corner
        label = f"ID:{tid}"
        if action:
            label += f"  {action}"

        # Measure text
        tbbox = draw.textbbox((0, 0), label, font=font)
        tw, th = tbbox[2] - tbbox[0] + 12, tbbox[3] - tbbox[1] + 8
        lx, ly = x1, max(0, y1 - th - 6)

        # Pill background with dark border for contrast
        draw.rectangle([lx - 2, ly - 2, lx + tw + 2, ly + th + 2], fill=(0, 0, 0))
        draw.rectangle([lx, ly, lx + tw, ly + th], fill=line_rgb)
        # Text with shadow
        text_x, text_y = lx + 6, ly + 4
        draw.text((text_x + 1, text_y + 1), label, font=font, fill=(0, 0, 0))
        text_color = (255, 255, 255) if sum(line_rgb) < 400 else (0, 0, 0)
        draw.text((text_x, text_y), label, font=font, fill=text_color)

    # ── OpenCV fallback ──

    def _draw_opencv(self, frame: np.ndarray, persons: list[dict],
                     track_levels: dict[int, int]) -> np.ndarray:
        for p in persons:
            self._draw_person_opencv(frame, p, track_levels)
        return frame

    def _draw_person_opencv(self, frame, p: dict, track_levels: dict) -> None:
        bbox = p["bbox"]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 10:
            return

        dl = track_levels.get(p["track_id"], 0)
        action = p.get("action", "")
        tid = p.get("track_id", -1)
        action_id = _action_id_from_name(action)

        color = _LEVEL_BGR[dl] if dl > 0 else _ACTION_BGR.get(action_id, (80, 200, 60))

        # Main outline (no fill — keep person visible)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Corner accents
        cl = min(25, bw // 4, bh // 4)
        bright = tuple(min(255, c + 80) for c in color)
        for cx, cy, dx, dy in [
            (x1, y1, 1, 1), (x2, y1, -1, 1),
            (x1, y2, 1, -1), (x2, y2, -1, -1),
        ]:
            cv2.line(frame, (cx, cy), (cx + dx * cl, cy), bright, 3)
            cv2.line(frame, (cx, cy), (cx, cy + dy * cl), bright, 3)

        # Label pill — top-left
        label = f"ID:{tid}"
        if action:
            label += f"  {action}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
        lx, ly = x1, max(th + 8, y1 - th - 8)
        # Dark shadow behind pill for contrast
        cv2.rectangle(frame, (lx - 5, ly - th - 7), (lx + tw + 7, ly + 5),
                      (0, 0, 0), -1)
        cv2.rectangle(frame, (lx - 4, ly - th - 6), (lx + tw + 6, ly + 4),
                      color, -1)
        cv2.putText(frame, label, (lx, ly),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
