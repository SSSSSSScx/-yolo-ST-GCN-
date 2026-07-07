import time
import numpy as np
from .detection import PersonDetector
from .tracking import PersonTracker
from .pose import PoseEstimator
from .action import ActionRecognizer
from .ppe import PPEDetector
from .object_detect import ObjectDetector


class InferencePipeline:
    def __init__(self, detector: PersonDetector, tracker: PersonTracker,
                 pose_estimator: PoseEstimator = None, action_recognizer: ActionRecognizer = None,
                 ppe_detector: PPEDetector = None, object_detector: ObjectDetector = None):
        self._detector = detector
        self._tracker = tracker
        self._pose = pose_estimator
        self._action = action_recognizer
        self._ppe = ppe_detector
        self._object = object_detector
        self._frame_id = 0
        self._fps = 30.0
        self._last_time = time.time()

        # Keyframe scheduling: run expensive ops every N frames
        self._pose_interval = 3       # Pose estimation every 3 frames
        self._ppe_interval = 5        # PPE detection every 5 frames
        self._object_interval = 10    # Object detection every 10 frames

        # Cached results between keyframes
        self._cached_keypoints: list = []
        self._cached_ppe: list = []
        self._cached_objects: list = []

    def process(self, frame) -> dict:
        self._frame_id += 1
        t0 = time.time()

        # Update FPS estimate
        dt = t0 - self._last_time
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(dt, 0.001))
        self._last_time = t0

        # Detection + Tracking (run every frame)
        detections = self._detector.detect(frame)
        tracked = self._tracker.update(detections)

        bboxes = [t["bbox"] for t in tracked]
        n_persons = len(tracked)

        # Pose estimation: keyframe or when person count changes
        if self._pose is not None and bboxes:
            if self._frame_id % self._pose_interval == 0 or n_persons != len(self._cached_keypoints):
                self._cached_keypoints = self._pose.estimate(frame, bboxes)
            keypoints_list = self._cached_keypoints
        else:
            keypoints_list = []

        # Build person data
        persons = []
        person_data = []
        for i, trk in enumerate(tracked):
            track_id = trk["track_id"]
            kpts = keypoints_list[i] if i < len(keypoints_list) else None
            bbox = trk["bbox"]
            kpts_list = kpts.tolist() if kpts is not None and hasattr(kpts, 'tolist') else None

            action_result = {"action": "站立", "action_id": 0, "confidence": 0.0, "pose_keypoints": None}
            if self._action is not None:
                # Always call recognize — even with bad keypoints, VideoMAE can use frame
                if kpts is None or not hasattr(kpts, 'shape') or kpts.shape != (17, 3):
                    kpts = np.zeros((17, 3), dtype=np.float32)
                action_result = self._action.recognize(track_id, kpts, bbox, self._fps, frame=frame)

            person_data.append({
                "track_id": track_id,
                "bbox": bbox,
                "pose_keypoints": kpts,
            })

            persons.append({
                "track_id": track_id,
                "bbox": bbox,
                "centroid": trk["centroid"],
                "state": trk["state"],
                "confidence": trk.get("confidence", 0.0),
                "action": action_result["action"],
                "action_id": action_result.get("action_id", 7),
                "action_confidence": action_result["confidence"],
                "pose_keypoints": kpts_list,
            })

        # PPE detection: keyframe or when person count changes
        if self._ppe is not None and person_data:
            if self._frame_id % self._ppe_interval == 0 or n_persons != len(self._cached_ppe):
                self._cached_ppe = self._ppe.detect(frame, person_data)
            ppe_results = self._cached_ppe
        else:
            ppe_results = []

        ppe_by_track = {p["track_id"]: p["ppe"] for p in ppe_results}
        for person in persons:
            person["ppe"] = ppe_by_track.get(person["track_id"], {})

        # Object detection: sparse keyframe only (heavy operation)
        if self._object is not None:
            if self._frame_id % self._object_interval == 0:
                self._cached_objects = self._object.detect(frame)
            object_detections = self._cached_objects
        else:
            object_detections = []

        return {
            "frame_id": self._frame_id,
            "timestamp": t0,
            "persons": persons,
            "object_detections": object_detections,
        }
