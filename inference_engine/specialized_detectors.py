"""6 special-purpose action detectors. V2: eating/smoking use YOLO object detection."""

import numpy as np
from collections import deque

# MediaPipe indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_ANKLE, R_ANKLE = 27, 28


def _get_xy(lm, idx):
    if hasattr(lm, 'shape'): return lm[idx, 0], lm[idx, 1]
    return lm[idx].x, lm[idx].y


def _midpoint(lm, a, b):
    ax, ay = _get_xy(lm, a); bx, by = _get_xy(lm, b)
    return (ax + bx) / 2, (ay + by) / 2


def _shoulder_width(lm):
    return np.sqrt((_get_xy(lm, L_SHOULDER)[0] - _get_xy(lm, R_SHOULDER)[0])**2 +
                   (_get_xy(lm, L_SHOULDER)[1] - _get_xy(lm, R_SHOULDER)[1])**2)


# COCO classes for eating/drinking objects
EAT_OBJECTS = {39, 40, 41, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55}


def _hand_has_object(landmarks, yolo_dets):
    """Check if a YOLO detection overlaps with hand region."""
    if not yolo_dets: return False, None
    lwx, lwy = _get_xy(landmarks, L_WRIST)
    rwx, rwy = _get_xy(landmarks, R_WRIST)
    sw = _shoulder_width(landmarks)
    radius = sw * 0.3
    for det in yolo_dets:
        bbox = det.get("bbox", [])
        if len(bbox) < 4: continue
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        if min(np.sqrt((cx - lwx)**2 + (cy - lwy)**2),
               np.sqrt((cx - rwx)**2 + (cy - rwy)**2)) < radius:
            return True, det.get("class_id", det.get("class", -1))
    return False, None


# ============================================================
class FallDetector:
    """Detect falling/falen via nose descent + body proximity to ground."""
    def __init__(self, history_len=15):
        self.history_len = history_len
        self._nose_hist = {}; self._hip_hist = {}

    def detect(self, track_id, landmarks, img_h=1080):
        if track_id not in self._nose_hist:
            self._nose_hist[track_id] = deque(maxlen=self.history_len)
            self._hip_hist[track_id] = deque(maxlen=self.history_len)
        nx, ny = _get_xy(landmarks, NOSE)
        hx, hy = _midpoint(landmarks, L_HIP, R_HIP)
        self._nose_hist[track_id].append(ny); self._hip_hist[track_id].append(hy)
        nh, hh = list(self._nose_hist[track_id]), list(self._hip_hist[track_id])
        if len(nh) < self.history_len: return (None, 0)
        avg_h = _shoulder_width(landmarks) * 1.5
        head_dropping = (nh[-1] - nh[0]) > 0.3 * max(avg_h, 1)
        body_ground = hy < 0.5 * max(np.mean(hh), 1)
        if head_dropping and body_ground: return (6, 0.9)
        if head_dropping: return (5, 0.7)
        return (None, 0)


class FightDetector:
    """Detect pushing via wrist speed + VideoMAE."""
    def __init__(self):
        self._wrist_hist = {}

    def detect(self, track_id, landmarks, vm_fight=0.0, img_h=1080):
        lwx, lwy = _get_xy(landmarks, L_WRIST); rwx, rwy = _get_xy(landmarks, R_WRIST)
        if track_id not in self._wrist_hist:
            self._wrist_hist[track_id] = deque(maxlen=2)
        prev = self._wrist_hist[track_id]
        self._wrist_hist[track_id].append(((lwx, lwy), (rwx, rwy)))
        arm_fast = False
        if len(prev) >= 1:
            plw, prw = prev[-1]
            lsp = np.sqrt((lwx-plw[0])**2+(lwy-plw[1])**2)
            rsp = np.sqrt((rwx-prw[0])**2+(rwy-prw[1])**2)
            arm_fast = max(lsp, rsp) > 0.02 * max(_shoulder_width(landmarks)*1.5, 1)
        vm_fight = vm_fight if isinstance(vm_fight, (int, float)) else 0.0
        if vm_fight > 0.6 and arm_fast: return (9, 0.85)
        if vm_fight > 0.5: return (9, vm_fight * 0.7)
        return (None, 0)


class RunDetector:
    """Detect running via hip speed + airborne ratio."""
    def __init__(self, history_len=10):
        self.history_len = history_len
        self._hip_hist = {}; self._ankle_hist = {}

    def detect(self, track_id, landmarks, fps=30):
        hx, hy = _midpoint(landmarks, L_HIP, R_HIP)
        lay, ray = _get_xy(landmarks, L_ANKLE)[1], _get_xy(landmarks, R_ANKLE)[1]
        if track_id not in self._hip_hist:
            self._hip_hist[track_id] = deque(maxlen=self.history_len)
            self._ankle_hist[track_id] = deque(maxlen=self.history_len)
        self._hip_hist[track_id].append((hx, hy))
        self._ankle_hist[track_id].append(min(lay, ray))
        hh, ah = list(self._hip_hist[track_id]), list(self._ankle_hist[track_id])
        if len(hh) < self.history_len: return (None, 0)
        dx = hh[-1][0] - hh[0][0]
        speed = abs(dx) / self.history_len
        SPEED_THRESHOLD = 15.0
        if speed < SPEED_THRESHOLD: return (None, 0)
        base_ankle = max(ah)
        airborne = sum(1 for a in ah if a > 0.95 * base_ankle) / len(ah)
        if airborne > 0.3: return (3, 0.85)
        return (3, 0.6)


class EatingDetector:
    """Eating: state-machine based hand-to-mouth reciprocation detector.

    States: AWAY, NEAR, TRANSITION
    Confirmation: 3+ complete cycles (AWAY→NEAR→AWAY) in 15-second window,
    with NEAR stay duration 0.3-2.0 seconds per visit.
    """
    def __init__(self, window_frames=450):  # 15 sec @ 30fps
        self.window = window_frames
        self._dist_history = {}
        self._state = {}      # per-track: current state
        self._near_start = {} # per-track: frame when NEAR started
        self._cycles = {}     # per-track: list of cycle (near_start_frame, near_end_frame)
        self._near_wrist_pos = {}  # per-track: wrist position at start of NEAR

    def detect(self, track_id, landmarks, vm_eat=0.0, yolo_dets=None):
        nx, ny = _get_xy(landmarks, NOSE)
        lwx, lwy = _get_xy(landmarks, L_WRIST)
        rwx, rwy = _get_xy(landmarks, R_WRIST)
        sw = _shoulder_width(landmarks)
        if sw < 1: sw = 1

        # Normalized wrist-nose distance
        lw_d = np.sqrt((lwx-nx)**2+(lwy-ny)**2) / sw
        rw_d = np.sqrt((rwx-nx)**2+(rwy-ny)**2) / sw
        min_dist = min(lw_d, rw_d)
        # Which wrist is closer
        wrist_x, wrist_y = (lwx, lwy) if lw_d < rw_d else (rwx, rwy)

        # Init track
        if track_id not in self._state:
            self._state[track_id] = "AWAY"
            self._dist_history[track_id] = deque(maxlen=self.window)
            self._cycles[track_id] = []
            self._near_start[track_id] = 0
            self._near_wrist_pos[track_id] = (0.0, 0.0)

        self._dist_history[track_id].append(min_dist)
        frame_idx = len(self._dist_history[track_id])

        # ---- State Machine ----
        prev_state = self._state[track_id]
        NEAR_THRESH = 0.40
        AWAY_THRESH = 0.55

        if min_dist < NEAR_THRESH:
            new_state = "NEAR"
        elif min_dist > AWAY_THRESH:
            new_state = "AWAY"
        else:
            new_state = "TRANSITION"

        # ---- Handle state transitions ----
        if prev_state == "AWAY" and new_state == "NEAR":
            # Hand moved to mouth — start timing
            self._near_start[track_id] = frame_idx
            self._near_wrist_pos[track_id] = (wrist_x, wrist_y)

        elif prev_state == "NEAR" and new_state == "AWAY":
            # Hand left mouth — complete a cycle
            near_end = frame_idx
            near_start = self._near_start[track_id]
            duration = near_end - near_start  # frames

            # Check NEAR stay quality:
            # 1. Duration: 10-60 frames (0.3-2.0 sec at 30fps)
            duration_ok = 10 <= duration <= 60

            # 2. Wrist displacement during NEAR stay < 0.05 (normalized)
            wx, wy = self._near_wrist_pos[track_id]
            displacement = np.sqrt((wrist_x-wx)**2 + (wrist_y-wy)**2) / sw
            displacement_ok = displacement < 0.05

            if duration_ok and displacement_ok:
                self._cycles[track_id].append((near_start, near_end))

        self._state[track_id] = new_state

        # ---- Evaluate eating ----
        # Clean old cycles (> window frames ago)
        self._cycles[track_id] = [(s, e) for s, e in self._cycles[track_id]
                                   if frame_idx - s < self.window]

        num_cycles = len(self._cycles[track_id])
        vm = vm_eat if isinstance(vm_eat, (int, float)) else 0.0

        # Object in hand bonus
        has_obj, obj_cls = _hand_has_object(landmarks, yolo_dets)
        is_eat_obj = obj_cls in EAT_OBJECTS if obj_cls else False

        # Scoring
        if num_cycles >= 3 and num_cycles <= 10:
            if is_eat_obj: return (4, 0.95)
            if vm > 0.4: return (4, 0.88)
            return (4, 0.78)

        if num_cycles >= 2 and is_eat_obj:
            return (4, 0.75)

        return (None, 0)


class SmokingDetector:
    """Smoking: sustained hand-near-face WITHOUT eating reciprocation.

    Key distinction from eating:
    - Smoking: continuous NEAR state >60 frames (2+ seconds), no cycles
    - Eating: multiple AWAY→NEAR→AWAY cycles
    """
    def __init__(self, window_frames=90):  # 3 sec @ 30fps
        self.window = window_frames
        self._dist_hist = {}

    def detect(self, track_id, landmarks, vm_smoke=0.0, yolo_dets=None):
        nx, ny = _get_xy(landmarks, NOSE)
        lwx, lwy = _get_xy(landmarks, L_WRIST)
        rwx, rwy = _get_xy(landmarks, R_WRIST)
        sw = _shoulder_width(landmarks)
        if sw < 1: sw = 1

        min_d = min(np.sqrt((lwx-nx)**2+(lwy-ny)**2), np.sqrt((rwx-nx)**2+(rwy-ny)**2)) / sw

        if track_id not in self._dist_hist:
            self._dist_hist[track_id] = deque(maxlen=self.window)
        self._dist_hist[track_id].append(min_d)
        dh = list(self._dist_hist[track_id])

        if len(dh) < self.window:
            return (None, 0)

        # Continuous NEAR: >65% of frames below 0.40 threshold
        near_frames = sum(1 for d in dh if d < 0.40)
        near_ratio = near_frames / len(dh)

        # Check for eating cycles (exclude smoking if eating-like pattern detected)
        cycles = 0
        state = None
        for d in dh:
            if d < 0.40:
                if state == 'away': cycles += 1
                state = 'near'
            elif d > 0.55:
                if state == 'near': state = 'away'

        is_eating_like = cycles >= 2  # If looks like eating, don't classify as smoking

        has_obj, _ = _hand_has_object(landmarks, yolo_dets)
        vm = vm_smoke if isinstance(vm_smoke, (int, float)) else 0.0

        # Smoking: sustained near-face AND NOT eating-like pattern
        if near_ratio > 0.60 and not is_eating_like:
            if has_obj and vm > 0.4: return (8, 0.90)
            if vm > 0.5: return (8, 0.82)
            return (8, 0.70)

        return (None, 0)


class PostureDetector:
    """Sitting / walking / standing with robust speed measurement."""
    def __init__(self):
        self._hip_hist = {}

    def detect(self, track_id, landmarks, fps=30):
        sx, sy = _midpoint(landmarks, L_SHOULDER, R_SHOULDER)
        hx, hy = _midpoint(landmarks, L_HIP, R_HIP)
        tilt = np.degrees(np.arctan2(abs(sx-hx), max(abs(sy-hy), 1)))
        if track_id not in self._hip_hist:
            self._hip_hist[track_id] = deque(maxlen=15)
        self._hip_hist[track_id].append((hx, hy))
        hh = list(self._hip_hist[track_id])
        # Sustained speed over 15 frames (0.5 sec) to avoid jitter
        speed = 0.0
        if len(hh) >= 10:
            dx = hh[-1][0] - hh[-10][0]
            dy = hh[-1][1] - hh[-10][1]
            speed = np.sqrt(dx**2 + dy**2) / 10  # avg per frame
        # Sitting: bent posture (tilt > 40°) overrides everything
        if tilt > 40:
            return (2, 0.9)
        # Walking: sustained speed > 6 px/frame over 15-frame window
        if speed > 6 and tilt < 30:
            return (1, 0.85)
        # Standing: default
        return (0, 0.9)
