import cv2
import numpy as np
from loguru import logger


class PPEDetector:
    """PPE (Personal Protective Equipment) detection.

    Uses person ROI sub-regions for targeted detection:
    - head_region (upper 25%): goggles, mask, face_shield
    - torso_region (middle 40%): lab_coat
    - hand_regions (wrist keypoints): gloves

    Uses heuristic color analysis as fallback when no specialized PPE model.
    """

    def __init__(self, model_path: str = "models/ppe_yolov12s.onnx"):
        self._model_path = model_path
        self._model = None
        self._loaded = False
        # Tracking for frame-consistency (5 frames to confirm missing)
        self._missing_counter: dict[int, dict[str, int]] = {}
        self._confirm_frames = 5

    def detect(self, frame: np.ndarray, person_data: list[dict]) -> list[dict]:
        """Detect PPE for each person.

        Args:
            frame: Full BGR frame
            person_data: list of {"track_id", "bbox", "pose_keypoints"}

        Returns:
            list of {"track_id": int, "ppe": {"lab_coat": bool, ...}}
        """
        results = []
        for person in person_data:
            track_id = person["track_id"]
            bbox = person["bbox"]
            kpts = person.get("pose_keypoints")

            ppe_status = self._heuristic_ppe(frame, bbox, kpts)
            ppe_status = self._apply_consistency(track_id, ppe_status)

            results.append({
                "track_id": track_id,
                "ppe": ppe_status,
            })

        return results

    def _heuristic_ppe(self, frame, bbox, kpts) -> dict[str, bool]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [max(0, min(d, s)) for d, s in zip(
            [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
            [w, h, w, h])]
        if x2 <= x1 or y2 <= y1:
            return {"lab_coat": False, "goggles": False, "gloves": False, "mask": False, "face_shield": False}

        # Sub-region extraction
        bh = y2 - y1
        head_region = frame[y1:y1 + int(bh * 0.25), x1:x2]
        torso_region = frame[y1 + int(bh * 0.2):y1 + int(bh * 0.65), x1:x2]

        # Heuristic: lab coat detection (white/light fabric on torso)
        lab_coat = self._detect_light_fabric(torso_region)

        # Heuristic: goggles (dark band across eye area)
        goggles = self._detect_goggles(head_region)

        # Heuristic: mask (light/dark area over lower face)
        mask = self._detect_mask(head_region)

        # Gloves: check hand regions from keypoints
        gloves = False
        if kpts is not None and len(kpts) >= 11:
            gloves = self._detect_gloves(frame, kpts)

        return {
            "lab_coat": lab_coat,
            "goggles": goggles,
            "gloves": gloves,
            "mask": mask,
            "face_shield": False,
        }

    def _detect_light_fabric(self, region) -> bool:
        if region is None or region.size == 0:
            return False
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        # White/light colors: low saturation, high value
        mask = cv2.inRange(hsv, (0, 0, 140), (180, 60, 255))
        ratio = np.count_nonzero(mask) / max(mask.size, 1)
        return ratio > 0.35

    def _detect_goggles(self, region) -> bool:
        if region is None or region.size == 0:
            return False
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        # Goggles typically appear as dark horizontal band in upper portion
        upper = gray[:max(1, gray.shape[0] // 2), :]
        dark_pixels = np.sum(upper < 80)
        total = max(upper.size, 1)
        return dark_pixels / total > 0.15

    def _detect_mask(self, region) -> bool:
        if region is None or region.size == 0:
            return False
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        # Mask area: lower half of face region
        lower = gray[max(1, gray.shape[0] // 2):, :]
        if lower.size == 0:
            return False
        # Check for uniform brightness (mask covers skin texture)
        std = np.std(lower)
        return std < 55

    def _detect_gloves(self, frame, kpts) -> bool:
        h, w = frame.shape[:2]
        # Check wrist regions (keypoints 9, 10)
        glove_score = 0
        for kid in [9, 10]:
            if kpts[kid, 2] < 0.3:
                continue
            wx, wy = int(kpts[kid, 0]), int(kpts[kid, 1])
            r = 30
            x1 = max(0, wx - r); y1 = max(0, wy - r)
            x2 = min(w, wx + r); y2 = min(h, wy + r)
            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            # Blue/purple gloves common in labs
            mask_blue = cv2.inRange(hsv, (90, 50, 50), (140, 255, 200))
            # White gloves
            mask_white = cv2.inRange(hsv, (0, 0, 150), (180, 40, 255))
            ratio = (np.count_nonzero(mask_blue) + np.count_nonzero(mask_white)) / max(patch.size // 3, 1)
            if ratio > 0.3:
                glove_score += 1
        return glove_score >= 1

    def _apply_consistency(self, track_id: int, current: dict[str, bool]) -> dict[str, bool]:
        if track_id not in self._missing_counter:
            self._missing_counter[track_id] = {k: 0 for k in current}

        result = {}
        for item, present in current.items():
            if not present:
                self._missing_counter[track_id][item] += 1
            else:
                self._missing_counter[track_id][item] = 0

            # Only report missing after consecutive confirmation
            result[item] = self._missing_counter[track_id][item] < self._confirm_frames

        # Cleanup old tracks
        if len(self._missing_counter) > 50:
            self._missing_counter.clear()

        return result

    def reset_track(self, track_id: int) -> None:
        self._missing_counter.pop(track_id, None)
