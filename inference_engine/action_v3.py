"""Action Recognition V3 — complete pipeline using visibility routing + specialized detectors.

Flow:
  MediaPipe landmarks → VisibilityRouter → full/limited mode
  → 6 specialized detectors → dict of {class_id: confidence}
  → PriorityArbiter → (class_id, confidence)
  → TemporalSmoother → final output
"""

import numpy as np
from collections import deque
from loguru import logger
from .action_labels import ACTION_LABELS

# NTU60 → 10-class mapping
_NTU_MAP = {0:4,1:4,7:2,8:0,41:5,42:5,49:9,50:9,51:9,53:9,58:1,59:1}
from .visibility_router import VisibilityRouter
from .specialized_detectors import (
    FallDetector, FightDetector, RunDetector,
    SmokingDetector, EatingDetector, PostureDetector,
)
from .priority_arbiter import PriorityArbiter
from .temporal_smoother import TemporalSmoother
from .videomae_inference import VideoMAEInference
from .ctrgc_onnx import CTRGCNInference
from .coco_to_ntu import coco_to_ntu
from .temporal_resampler import prepare_ctrgcn_input


class ActionRecognizerV3:
    """Complete V3 action recognition pipeline."""

    def __init__(self):
        self.visibility = VisibilityRouter()
        self.fall_det = FallDetector()
        self.fight_det = FightDetector()
        self.run_det = RunDetector()
        self.smoke_det = SmokingDetector()
        self.eat_det = EatingDetector()
        self.posture_det = PostureDetector()
        self.arbiter = PriorityArbiter()
        self.smoother = TemporalSmoother()

        # VideoMAE (lazy-loaded)
        self._videomae = None
        self._vm_buffer = deque(maxlen=25)
        # ST-GCN for posture (lazy-loaded)
        self._stgcn = None
        self._coco_buffer: dict[int, deque] = {}
        self._stgcn_last: dict[int, tuple] = {}
        self._vm_last = None
        self._vm_counter = 0

        self._mode = "limited"
        self._vm_results: dict = None
        self._yolo_dets: list = []
        self._frame_count: dict[int, int] = {}

        logger.info("ActionRecognizer V3: VideoMAE + 6 detectors + arbitration + smoothing")

    def set_vm_result(self, result: dict):
        """Accept VideoMAE prediction from external pipeline."""
        self._vm_results = result

    def _coco17_to_mp33(self, coco):
        """Convert COCO-17 (17,3) to MediaPipe-33-like (33,3) format."""
        mp = np.zeros((33, 3), dtype=np.float32)
        if coco is None or not hasattr(coco, 'shape'):
            return mp
        if coco.shape == (33, 3):
            return coco
        # COCO → MediaPipe mapping (approximate)
        coco_to_mp = {
            0: 0,    # nose
            1: 1, 2: 2, 3: 3, 4: 4,  # eyes/ears → approximate
            5: 11,   # L shoulder
            6: 12,   # R shoulder
            7: 13,   # L elbow
            8: 14,   # R elbow
            9: 15,   # L wrist
            10: 16,  # R wrist
            11: 23,  # L hip
            12: 24,  # R hip
            13: 25,  # L knee
            14: 26,  # R knee
            15: 27,  # L ankle
            16: 28,  # R ankle
        }
        for ci, mi in coco_to_mp.items():
            if ci < coco.shape[0]:
                mp[mi] = coco[ci]
        return mp

    def set_yolo_detections(self, detections: list):
        """Accept YOLO object detections for eating/smoking analysis.
        detections: list of {class_id, bbox, confidence}
        """
        self._yolo_dets = detections or []

    def recognize(self, track_id: int, landmarks, fps=30, img_h=1080,
                  frame=None, bbox=None) -> dict:
        """Process one frame."""
        if landmarks is None:
            return {"action": ACTION_LABELS[7], "action_id": 7, "confidence": 0.0}

        # Convert COCO-17 → MediaPipe-33 if needed
        if hasattr(landmarks, 'shape') and landmarks.shape[0] == 17:
            landmarks = self._coco17_to_mp33(landmarks)

        # ---- VideoMAE: collect person crops ----
        if frame is not None and bbox and len(bbox) >= 4:
            self._vm_counter += 1
            try:
                x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        import cv2
                        crop = cv2.resize(crop, (224, 224))
                        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        self._vm_buffer.append(crop)
            except Exception:
                pass

            # Run VideoMAE every 16 frames
            if len(self._vm_buffer) >= 16 and self._vm_counter % 8 == 0:
                if self._videomae is None:
                    self._videomae = VideoMAEInference("models/videomae_lab_safety")
                if self._videomae.available:
                    try:
                        import torch
                        frames = list(self._vm_buffer)[-16:]
                        video = np.stack(frames)
                        from transformers import VideoMAEImageProcessor
                        processor = VideoMAEImageProcessor.from_pretrained("models/videomae_lab_safety")
                        inputs = processor(list(video), return_tensors="pt")
                        inputs = {k: v.to(self._videomae._device, dtype=self._videomae._model.dtype)
                                  for k, v in inputs.items()}
                        with torch.no_grad():
                            outputs = self._videomae._model(**inputs)
                            logits = outputs.logits[0].float().cpu().numpy()
                        probs = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
                        pred_id = int(np.argmax(probs))
                        label = self._videomae._model.config.id2label.get(pred_id, "normal")
                        self._vm_results = {"videomae_label": label, "confidence": float(probs[pred_id])}
                        if self._vm_counter % 120 == 0:
                            logger.info(f"[VM] → {label} ({self._vm_results['confidence']:.2f})")
                    except Exception as e:
                        if self._vm_counter % 120 == 0:
                            logger.warning(f"[VM] inference failed: {e}")

        # ---- Step 1: Visibility assessment ----
        assessment = self.visibility.assess(landmarks)
        self._mode = self.visibility.should_switch(self._mode, assessment)

        fc = self._frame_count.get(track_id, 0) + 1
        self._frame_count[track_id] = fc

        # ---- ST-GCN for posture classes (standing/walking/sitting) ----
        if hasattr(landmarks, 'shape') and landmarks.shape[0] == 17:
            # Save original COCO keypoints
            if track_id not in self._coco_buffer:
                self._coco_buffer[track_id] = deque(maxlen=64)
            self._coco_buffer[track_id].append(landmarks.copy())

        # Run ST-GCN every 8 frames
        if fc >= 30 and fc % 8 == 0:
            if self._stgcn is None:
                self._stgcn = CTRGCNInference("models/stgcn.onnx")
            if self._stgcn.available and track_id in self._coco_buffer:
                try:
                    coco_frames = list(self._coco_buffer[track_id])
                    if len(coco_frames) >= 30:
                        ntu_frames = [coco_to_ntu(f) for f in coco_frames]
                        ntu_seq = np.stack(ntu_frames, axis=0)
                        inp = prepare_ctrgcn_input(ntu_seq)
                        probs = self._stgcn.infer(inp)
                        if probs is not None:
                            lab_probs = np.zeros(10)
                            for ntu_cls in range(min(len(probs), 60)):
                                lab_probs[_NTU_MAP.get(ntu_cls, 7)] += probs[ntu_cls]
                            best = int(np.argmax(lab_probs))
                            self._stgcn_last[track_id] = (best, float(lab_probs[best]))
                except Exception as e:
                    pass

        # ---- Step 2: Run all detectors ----
        detections = {}

        # Fall detector
        cid, conf = self.fall_det.detect(track_id, landmarks, img_h)
        if cid is not None:
            detections[cid] = conf

        # Fight detector (with VideoMAE)
        vm_fight = 0.0
        vm_eat = 0.0
        vm_smoke = 0.0
        if self._vm_results:
            vm_label = self._vm_results.get("videomae_label", "normal")
            vm_conf = self._vm_results.get("confidence", 0)
            if vm_label == "fight": vm_fight = vm_conf
            elif vm_label == "eating": vm_eat = vm_conf
            elif vm_label == "smoking": vm_smoke = vm_conf

        cid, conf = self.fight_det.detect(track_id, landmarks, vm_fight, img_h)
        if cid is not None:
            detections[cid] = conf

        # Run detector
        cid, conf = self.run_det.detect(track_id, landmarks, fps)
        if cid is not None:
            detections[cid] = conf

        # Smoke detector (with YOLO)
        yolo = getattr(self, '_yolo_dets', None)
        cid, conf = self.smoke_det.detect(track_id, landmarks, vm_smoke, yolo)
        if cid is not None:
            detections[cid] = conf

        # Eat detector (with YOLO)
        cid, conf = self.eat_det.detect(track_id, landmarks, vm_eat, yolo)
        if cid is not None:
            detections[cid] = conf

        # Posture detector (always returns a result)
        cid, conf = self.posture_det.detect(track_id, landmarks, fps)
        # Override with ST-GCN for posture classes if available
        if track_id in self._stgcn_last:
            sg_cid, sg_conf = self._stgcn_last[track_id]
            if sg_cid in (0, 1, 2) and sg_conf > 0.40:
                detections[sg_cid] = sg_conf
            else:
                detections[cid] = conf
        else:
            detections[cid] = conf

        # ---- Shared geometry ----
        lm = landmarks
        sw = max(np.sqrt((lm[11,0]-lm[12,0])**2 + (lm[11,1]-lm[12,1])**2), 1)

        # ---- Fight/Pushing: speed burst detection (distinguishes from walking) ----
        if not hasattr(self, '_prev_wrist'): self._prev_wrist = {}
        if not hasattr(self, '_spd_baseline'): self._spd_baseline = {}
        if not hasattr(self, '_burst_win'): self._burst_win = {}
        if not hasattr(self, '_burst_hold'): self._burst_hold = {}

        cur_spd = 0.0
        if track_id not in self._prev_wrist:
            self._prev_wrist[track_id] = ((lm[15,0],lm[15,1]),(lm[16,0],lm[16,1]))
            self._burst_win[track_id] = deque(maxlen=12)
        else:
            plw = self._prev_wrist[track_id]
            lw_spd = np.sqrt((lm[15,0]-plw[0][0])**2+(lm[15,1]-plw[0][1])**2)
            rw_spd = np.sqrt((lm[16,0]-plw[1][0])**2+(lm[16,1]-plw[1][1])**2)
            self._prev_wrist[track_id] = ((lm[15,0],lm[15,1]),(lm[16,0],lm[16,1]))
            cur_spd = max(lw_spd, rw_spd)

            # Sliding window: 4+ burst frames in last 12 → trigger hold
            # Speed > 50 px/f (~0.5× shoulder at 2m) rules out eating/drinking hand movements
            baseline = self._spd_baseline.get(track_id, cur_spd)
            is_burst = cur_spd > 50 and cur_spd > baseline * 4.0
            self._burst_win[track_id].append(1 if is_burst else 0)
            burst_hits = sum(self._burst_win[track_id])

            if burst_hits >= 4:
                self._burst_hold[track_id] = 5

            # Only update baseline when burst density is low (avoid contamination)
            if burst_hits <= 1:
                self._spd_baseline[track_id] = 0.9 * baseline + 0.1 * cur_spd

        # Running cooldown: prevent class 9 for N frames after running detected
        if not hasattr(self, '_run_cooldown'): self._run_cooldown = {}
        if track_id not in self._run_cooldown:
            self._run_cooldown[track_id] = 0
        if detections.get(2, 0) > 0.30:
            self._run_cooldown[track_id] = 15  # ~0.5s cooldown
        rc = self._run_cooldown[track_id]
        if rc > 0:
            self._run_cooldown[track_id] = rc - 1

        # Emit class 9 while hold is active (suppressed when running or in cooldown)
        hold = self._burst_hold.get(track_id, 0)
        if hold > 0:
            self._burst_hold[track_id] = hold - 1
            if detections.get(2, 0) < 0.30 and rc <= 0:
                detections[9] = max(detections.get(9, 0), min(0.90, 0.50 + max(cur_spd, 30) / 55))

        # ---- Geometric override: elbow + wrist for eating/smoking (with persistence) ----
        wn = min(np.sqrt((lm[15,0]-lm[0,0])**2+(lm[15,1]-lm[0,1])**2),
                 np.sqrt((lm[16,0]-lm[0,0])**2+(lm[16,1]-lm[0,1])**2))
        wn_norm = wn / sw
        def _elbow_angle(sh, el, wr):
            v1 = np.array([lm[sh,0]-lm[el,0], lm[sh,1]-lm[el,1]])
            v2 = np.array([lm[wr,0]-lm[el,0], lm[wr,1]-lm[el,1]])
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1 or n2 < 1: return 180.0
            return float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n1*n2), -1, 1))))
        min_elbow = min(_elbow_angle(11,13,15), _elbow_angle(12,14,16))

        # Persistence counters per track (separate for eating/smoking)
        if not hasattr(self, '_eat_persist'): self._eat_persist = {}
        if not hasattr(self, '_smoke_persist'): self._smoke_persist = {}
        if track_id not in self._eat_persist:
            self._eat_persist[track_id] = 0
            self._smoke_persist[track_id] = 0

        # Eating: elbow < 85°, wrist < 0.70×shoulder, wrist not above nose, sustained 15+ frames (~0.5s)
        lw_d = np.sqrt((lm[15,0]-lm[0,0])**2+(lm[15,1]-lm[0,1])**2)
        rw_d = np.sqrt((lm[16,0]-lm[0,0])**2+(lm[16,1]-lm[0,1])**2)
        near_wrist_y = lm[15,1] if lw_d <= rw_d else lm[16,1]
        wrist_not_above_nose = (near_wrist_y >= lm[0,1] - 0.12 * sw)
        is_eat_pose = (min_elbow < 85 and wn_norm < 0.70 and wrist_not_above_nose)
        if is_eat_pose:
            self._eat_persist[track_id] += 1
        else:
            self._eat_persist[track_id] = max(0, self._eat_persist[track_id] - 2)

        # Smoking: elbow < 80°, wrist < 0.50×shoulder, wrist not above nose, sustained 15+ frames
        is_smoke_pose = (min_elbow < 80 and wn_norm < 0.50 and wrist_not_above_nose)
        if is_smoke_pose:
            self._smoke_persist[track_id] += 1
        else:
            self._smoke_persist[track_id] = max(0, self._smoke_persist[track_id] - 2)

        p_eat = self._eat_persist[track_id]
        p_smoke = self._smoke_persist[track_id]
        if is_smoke_pose and p_smoke >= 15:
            detections[8] = max(detections.get(8, 0), 0.80)
        if is_eat_pose and p_eat >= 15:
            detections[4] = max(detections.get(4, 0), 0.78)

        # ---- Step 3: Priority arbitration ----
        cid, conf = self.arbiter.arbitrate(detections)

        # ---- Step 4: Temporal smoothing ----
        cid, conf = self.smoother.smooth(cid, conf, track_id)

        # Diagnostic log: every 30 frames or on action change
        if not hasattr(self, '_last_log_cid'): self._last_log_cid = {}
        prev_cid = self._last_log_cid.get(track_id, -1)
        if fc % 30 == 0 or cid != prev_cid:
            self._last_log_cid[track_id] = cid
            burst_info = ""
            if hasattr(self, '_burst_win') and track_id in self._burst_win:
                bl = self._spd_baseline.get(track_id, 0)
                bw = list(self._burst_win[track_id])
                bh = sum(bw)
                ho = self._burst_hold.get(track_id, 0)
                burst_info = f"spd={cur_spd:.0f} baseline={bl:.1f} hits={bh}/12 hold={ho}"
            wrist_above = "↑" if near_wrist_y < lm[0,1] - 0.12 * sw else "↓"
            eat_info = f"elbow={min_elbow:.0f}° wn_norm={wn_norm:.2f} wrist={wrist_above} p_eat={p_eat} p_smoke={p_smoke}"
            det_info = " ".join([f"{ACTION_LABELS.get(k,'?')}={v:.2f}" for k, v in
                                 sorted(detections.items(), key=lambda x: -x[1])[:3]])
            logger.info(f"[V3] t={track_id} f={fc} | {burst_info} | {eat_info} | det=[{det_info}] → {ACTION_LABELS.get(cid)}")

        return {
            "action": ACTION_LABELS.get(cid, "其他操作"),
            "action_id": cid,
            "confidence": conf,
        }

    def reset_track(self, track_id):
        self.smoother.reset(track_id)
        self._frame_count.pop(track_id, None)
        self.fall_det._nose_hist.pop(track_id, None)
        self.fall_det._hip_hist.pop(track_id, None)
        self.fight_det._wrist_hist.pop(track_id, None)
        self.run_det._hip_hist.pop(track_id, None)
        self.run_det._ankle_hist.pop(track_id, None)
        self.smoke_det._dist_hist.pop(track_id, None)
        self.eat_det._dist_history.pop(track_id, None)
        self.posture_det._hip_hist.pop(track_id, None)
        self._coco_buffer.pop(track_id, None)
        self._stgcn_last.pop(track_id, None)
        if hasattr(self, '_spd_baseline'):
            self._spd_baseline.pop(track_id, None)
        if hasattr(self, '_burst_win'):
            self._burst_win.pop(track_id, None)
        if hasattr(self, '_burst_hold'):
            self._burst_hold.pop(track_id, None)
        if hasattr(self, '_prev_wrist'):
            self._prev_wrist.pop(track_id, None)
        if hasattr(self, '_eat_persist'):
            self._eat_persist.pop(track_id, None)
        if hasattr(self, '_smoke_persist'):
            self._smoke_persist.pop(track_id, None)
        if hasattr(self, '_last_log_cid'):
            self._last_log_cid.pop(track_id, None)
        if hasattr(self, '_run_cooldown'):
            self._run_cooldown.pop(track_id, None)

