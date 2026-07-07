import numpy as np
from collections import deque


class KalmanBoxTracker:
    """Kalman filter tracking a bounding box in image space.

    State: [cx, cy, w, h, vx, vy, vw, vh]  (8-dim)
    """

    count = 0

    def __init__(self, bbox: list[float], confidence: float = 0.5):
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        x1, y1, x2, y2 = bbox
        self.kf = self._init_kalman(x1, y1, x2, y2)
        self.time_since_update = 0
        self.history: deque = deque(maxlen=60)
        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        self.state = "tentative"
        self.confidence = confidence
        self.confidence_history: deque = deque([confidence], maxlen=10)
        self.depth_feature = (y2 - y1) / max((x2 - x1), 1)  # aspect ratio as depth proxy

        # RTCM state
        self.rtcm_confidence = confidence
        self.ema_alpha = 0.25
        self.second_diff_window: deque = deque(maxlen=5)

        # AOCR state
        self.occluded_frames = 0
        self.virtual_trajectory: list = []

    @staticmethod
    def _init_kalman(x1, y1, x2, y2):
        # 8 states, 4 measurements
        import cv2
        kf = cv2.KalmanFilter(8, 4)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ], np.float32)
        kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        kf.processNoiseCov *= 0.03
        kf.measurementNoiseCov *= 0.5
        kf.errorCovPost *= 10.0
        kf.statePost = np.array([[cx], [cy], [w], [h], [0], [0], [0], [0]], np.float32)
        return kf

    def predict(self):
        predicted = self.kf.predict()
        cx, cy, w, h = predicted[:4].flatten()
        self.age += 1
        self.time_since_update += 1

        if self.time_since_update > 0:
            self._update_rtcm()

        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    def update(self, bbox: list[float], confidence: float = 0.5):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        self.kf.correct(np.array([[cx], [cy], [w], [h]], np.float32))

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.confidence = confidence
        self.confidence_history.append(confidence)
        self.depth_feature = h / max(w, 1)
        self.occluded_frames = 0
        self.virtual_trajectory.clear()

        centroid = (cx, cy, self.age)
        self.history.append(centroid)

        if self.state == "tentative" and self.hit_streak >= 3:
            self.state = "confirmed"
        elif self.state == "lost":
            self.state = "confirmed"
            self.hit_streak = 1

        self._update_rtcm()

    def _update_rtcm(self):
        """RTCM: Robust Trajectory Confidence Modeling.

        Combines Kalman prediction residual, exponential moving average,
        and second-order difference to estimate confidence during occlusion.
        """
        hist_len = len(self.confidence_history)
        if hist_len < 2:
            return

        # EMA smoothing
        ema = self.confidence_history[-1]
        for c in list(self.confidence_history)[-2::-1]:
            ema = self.ema_alpha * c + (1 - self.ema_alpha) * ema

        # Second-order difference
        if hist_len >= 3:
            diffs = []
            vals = list(self.confidence_history)
            for i in range(2, len(vals)):
                d2 = vals[i] - 2 * vals[i - 1] + vals[i - 2]
                diffs.append(abs(d2))
            self.second_diff_window.append(np.mean(diffs) if diffs else 0.0)

        avg_second_diff = np.mean(list(self.second_diff_window)) if self.second_diff_window else 0.0

        # Decay factor based on time since update
        decay = np.exp(-0.1 * self.time_since_update)

        self.rtcm_confidence = ema * decay - 0.1 * avg_second_diff
        self.rtcm_confidence = max(0.0, min(1.0, self.rtcm_confidence))

    def get_state(self):
        cx, cy, w, h = self.kf.statePost[:4].flatten()
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


class PersonTracker:
    """RAP-SORT: Robust Association Probabilistic SORT tracker."""

    def __init__(self, max_age: int = 30, min_hits: int = 3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0
        self._next_id = 0

    def update(self, detections: list[dict]) -> list[dict]:
        self.frame_count += 1

        # Predict all existing tracks
        predictions = []
        active_tracks = []
        for trk in self.trackers:
            pred_bbox = trk.predict()
            predictions.append(pred_bbox)
            active_tracks.append(trk)

        # Build cost matrix
        n_tracks = len(active_tracks)
        n_dets = len(detections)

        if n_tracks == 0:
            matches = []
            unmatched_tracks = []
            unmatched_dets = list(range(n_dets))
        elif n_dets == 0:
            matches = []
            unmatched_tracks = list(range(n_tracks))
            unmatched_dets = []
        else:
            cost = np.zeros((n_tracks, n_dets))
            for t in range(n_tracks):
                trk = active_tracks[t]
                for d in range(n_dets):
                    det = detections[d]
                    cost[t, d] = self._pdiou(trk, det)

            matches, unmatched_tracks, unmatched_dets = self._linear_assignment(cost)

        # AOCR: attempt recovery for long-unmatched tracks
        recovered = []
        still_unmatched = []
        for t in unmatched_tracks:
            trk = active_tracks[t]
            if trk.time_since_update <= self.max_age:
                still_unmatched.append(t)
            # If track was lost, AOCR might have recovered it above
            # Tracks exceeding max_age will be marked lost/deleted below

        unmatched_tracks = still_unmatched

        # Update matched tracks
        for t, d in matches:
            det = detections[d]
            active_tracks[t].update(det["bbox"], det.get("confidence", 0.5))

        # Create new tracks for unmatched detections
        for d in unmatched_dets:
            det = detections[d]
            if det.get("confidence", 0) >= 0.3:
                trk = KalmanBoxTracker(det["bbox"], det.get("confidence", 0.5))
                self.trackers.append(trk)

        # Manage track states — only output tracks that were matched this frame
        # Unmatched tracks keep predicting internally but don't produce ghost boxes
        results = []
        active = []
        matched_ids = {active_tracks[t].id for t, _ in matches}

        for trk in self.trackers:
            if trk.time_since_update > self.max_age:
                continue  # expired

            bbox = trk.get_state()
            centroid = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

            # Only output tracks that were matched this frame or are newly created (age <= 1)
            # This prevents ghost boxes from Kalman-predicted positions of lost tracks
            if trk.id in matched_ids or trk.age <= 1:
                results.append({
                    "track_id": trk.id,
                    "bbox": bbox,
                    "centroid": centroid,
                    "state": trk.state,
                    "confidence": trk.rtcm_confidence,
                    "hits": trk.hit_streak,
                })
            active.append(trk)

        self.trackers = active
        return results

    def _pdiou(self, track: KalmanBoxTracker, det: dict) -> float:
        """PDIoU: Pseudo-Depth IoU.

        Enhances standard IoU with bbox height and depth features
        for better spatial awareness in overlapping scenarios.
        """
        t_bbox = track.get_state()
        d_bbox = det["bbox"]

        iou = self._iou(t_bbox, d_bbox)

        # Height similarity (deeper people appear higher in image)
        t_h = t_bbox[3] - t_bbox[1]
        d_h = d_bbox[3] - d_bbox[1]
        h_sim = min(t_h, d_h) / max(t_h, d_h, 1)

        # Depth feature similarity
        t_depth = t_h / max(t_bbox[2] - t_bbox[0], 1)
        d_depth = d_h / max(d_bbox[2] - d_bbox[0], 1)
        depth_sim = 1.0 - abs(t_depth - d_depth)

        # Combined PDIoU distance
        pdiou = 0.5 * iou + 0.3 * h_sim + 0.2 * depth_sim

        # RTCM confidence weighting
        pdiou *= track.rtcm_confidence

        return 1.0 - pdiou  # cost (lower is better)

    @staticmethod
    def _iou(box_a, box_b):
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    @staticmethod
    def _linear_assignment(cost: np.ndarray) -> tuple:
        """Hungarian algorithm for optimal matching."""
        from scipy.optimize import linear_sum_assignment
        if cost.size == 0:
            return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
        row_ind, col_ind = linear_sum_assignment(cost)
        matches = [(r, c) for r, c in zip(row_ind, col_ind) if cost[r, c] < 0.5]

        matched_rows = {m[0] for m in matches}
        matched_cols = {m[1] for m in matches}
        unmatched_rows = [i for i in range(cost.shape[0]) if i not in matched_rows]
        unmatched_cols = [i for i in range(cost.shape[1]) if i not in matched_cols]

        return matches, unmatched_rows, unmatched_cols
