"""Visibility assessment — determines if the person is fully visible.

Uses MediaPipe 33-landmark indices directly.
"""

import numpy as np

# MediaPipe landmark indices
MP_NOSE = 0
MP_L_SHOULDER = 11
MP_R_SHOULDER = 12
MP_L_WRIST = 15
MP_R_WRIST = 16
MP_L_HIP = 23
MP_R_HIP = 24
MP_L_ANKLE = 27
MP_R_ANKLE = 28

CORE_POINTS = [
    MP_NOSE,
    MP_L_SHOULDER, MP_R_SHOULDER,
    MP_L_WRIST, MP_R_WRIST,
    MP_L_HIP, MP_R_HIP,
    MP_L_ANKLE, MP_R_ANKLE,
]


class VisibilityRouter:
    """Routes between 'full' and 'limited' modes based on keypoint quality."""

    def __init__(self, enter_threshold_frames=8, exit_threshold_frames=15,
                 torso_conf_threshold=0.7, min_core_points=6):
        self.enter_threshold = enter_threshold_frames
        self.exit_threshold = exit_threshold_frames
        self.torso_conf = torso_conf_threshold
        self.min_core = min_core_points
        self._enter_counter = 0
        self._exit_counter = 0
        self._current_mode = "limited"

    def assess(self, landmarks) -> str:
        """Assess visibility from MediaPipe landmarks (33, 3 or list of NormalizedLandmark).

        Returns 'full' or 'limited'.
        """
        if landmarks is None:
            return "limited"

        # Get confidences: if landmarks is (33,3) numpy, col 2 is confidence
        # If it's a list of MediaPipe landmarks, use .visibility
        if hasattr(landmarks, 'shape') and landmarks.shape == (33, 3):
            confs = landmarks[:, 2]
        else:
            confs = np.array([getattr(lm, 'visibility', 0.9) for lm in landmarks])

        # Core points with confidence > 0.5
        core_conf = confs[list(CORE_POINTS)]
        core_visible = (core_conf > 0.5).sum()

        # Torso: both shoulders and both hips > torso_conf_threshold
        shoulders_ok = (confs[MP_L_SHOULDER] > self.torso_conf and
                       confs[MP_R_SHOULDER] > self.torso_conf)
        hips_ok = (confs[MP_L_HIP] > self.torso_conf and
                  confs[MP_R_HIP] > self.torso_conf)
        torso_ok = shoulders_ok and hips_ok

        # Nose visible
        nose_ok = confs[MP_NOSE] > 0.5

        # At least one ankle visible
        ankle_ok = (confs[MP_L_ANKLE] > 0.5 or confs[MP_R_ANKLE] > 0.5)

        if core_visible >= self.min_core and torso_ok and nose_ok and ankle_ok:
            return "full"
        return "limited"

    def should_switch(self, current_mode, assessment):
        """Maintain hysteresis counters to avoid mode oscillation."""
        if current_mode == "limited":
            if assessment == "full":
                self._enter_counter += 1
                if self._enter_counter >= self.enter_threshold:
                    self._exit_counter = 0
                    return "full"
            else:
                self._enter_counter = 0
            return "limited"
        else:  # full
            if assessment == "limited":
                self._exit_counter += 1
                if self._exit_counter >= self.exit_threshold:
                    self._enter_counter = 0
                    return "limited"
            else:
                self._exit_counter = 0
            return "full"
