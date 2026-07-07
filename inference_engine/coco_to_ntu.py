"""COCO 17-keypoint to NTU 25-joint format converter.

Converts per-frame COCO pose estimates (from YOLO11n-pose) to NTU RGB+D
25-joint format for use with CTR-GCN pre-trained models.

COCO 17 keypoints:
  0: nose            1: left_eye        2: right_eye       3: left_ear
  4: right_ear       5: left_shoulder   6: right_shoulder  7: left_elbow
  8: right_elbow     9: left_wrist     10: right_wrist    11: left_hip
 12: right_hip      13: left_knee      14: right_knee     15: left_ankle
 16: right_ankle

NTU 25 joints:
  0: base_of_spine       1: middle_of_spine     2: neck              3: head
  4: left_shoulder       5: left_elbow          6: left_wrist        7: left_hand_tip
  8: right_shoulder      9: right_elbow        10: right_wrist      11: right_hand_tip
 12: left_hip           13: left_knee          14: left_ankle       15: left_big_toe
 16: right_hip          17: right_knee         18: right_ankle      19: right_big_toe
 20: left_small_toe     21: right_small_toe    22: left_thumb       23: right_thumb
 24: nose
"""

import numpy as np


# COCO index constants
_COCO_NOSE = 0
_COCO_LEFT_EYE = 1
_COCO_RIGHT_EYE = 2
_COCO_LEFT_EAR = 3
_COCO_RIGHT_EAR = 4
_COCO_LEFT_SHOULDER = 5
_COCO_RIGHT_SHOULDER = 6
_COCO_LEFT_ELBOW = 7
_COCO_RIGHT_ELBOW = 8
_COCO_LEFT_WRIST = 9
_COCO_RIGHT_WRIST = 10
_COCO_LEFT_HIP = 11
_COCO_RIGHT_HIP = 12
_COCO_LEFT_KNEE = 13
_COCO_RIGHT_KNEE = 14
_COCO_LEFT_ANKLE = 15
_COCO_RIGHT_ANKLE = 16


def coco_to_ntu(coco_kpts: np.ndarray) -> np.ndarray:
    """Convert COCO 17 (17,3) [x, y, conf] to NTU 25 (25,3) [x, y, conf].

    Mapping strategy:
    - 13 direct mappings (identical body parts)
    - 2 near-direct (head ≈ nose, neck = midpoint of shoulders)
    - 6 interpolated/copied joints (spine chain, hand tips, thumbs)
    - 4 fabricated joints (toe approximations from ankle positions)

    Args:
        coco_kpts: np.ndarray of shape (17, 3) with [x, y, confidence].

    Returns:
        np.ndarray of shape (25, 3) with [x, y, confidence].
    """
    ntu = np.zeros((25, 3), dtype=np.float32)

    # ---- Direct mappings (13 joints) ----
    ntu[4]  = coco_kpts[_COCO_LEFT_SHOULDER]    # left_shoulder
    ntu[8]  = coco_kpts[_COCO_RIGHT_SHOULDER]   # right_shoulder
    ntu[5]  = coco_kpts[_COCO_LEFT_ELBOW]       # left_elbow
    ntu[9]  = coco_kpts[_COCO_RIGHT_ELBOW]      # right_elbow
    ntu[6]  = coco_kpts[_COCO_LEFT_WRIST]       # left_wrist
    ntu[10] = coco_kpts[_COCO_RIGHT_WRIST]      # right_wrist
    ntu[12] = coco_kpts[_COCO_LEFT_HIP]         # left_hip
    ntu[16] = coco_kpts[_COCO_RIGHT_HIP]        # right_hip
    ntu[13] = coco_kpts[_COCO_LEFT_KNEE]        # left_knee
    ntu[17] = coco_kpts[_COCO_RIGHT_KNEE]       # right_knee
    ntu[14] = coco_kpts[_COCO_LEFT_ANKLE]       # left_ankle
    ntu[18] = coco_kpts[_COCO_RIGHT_ANKLE]      # right_ankle
    ntu[24] = coco_kpts[_COCO_NOSE]             # nose

    # ---- Near-direct (head ≈ nose) ----
    # NTU "head" (joint 3) is the head center; approximate with nose position
    ntu[3] = coco_kpts[_COCO_NOSE].copy()
    ntu[3, 1] -= 5  # nudge slightly upward (nose is below head center)

    # ---- Interpolated joints ----
    # Neck (joint 2): midpoint of left/right shoulder
    l_shoulder = coco_kpts[_COCO_LEFT_SHOULDER]
    r_shoulder = coco_kpts[_COCO_RIGHT_SHOULDER]
    ntu[2, 0] = 0.5 * (l_shoulder[0] + r_shoulder[0])
    ntu[2, 1] = 0.5 * (l_shoulder[1] + r_shoulder[1])
    ntu[2, 2] = min(l_shoulder[2], r_shoulder[2])

    # Base of spine (joint 0): midpoint of left/right hip
    l_hip = coco_kpts[_COCO_LEFT_HIP]
    r_hip = coco_kpts[_COCO_RIGHT_HIP]
    ntu[0, 0] = 0.5 * (l_hip[0] + r_hip[0])
    ntu[0, 1] = 0.5 * (l_hip[1] + r_hip[1])
    ntu[0, 2] = min(l_hip[2], r_hip[2])

    # Middle of spine (joint 1): midpoint of neck and base_of_spine
    ntu[1, 0] = 0.5 * (ntu[2, 0] + ntu[0, 0])
    ntu[1, 1] = 0.5 * (ntu[2, 1] + ntu[0, 1])
    ntu[1, 2] = min(ntu[2, 2], ntu[0, 2])

    # ---- Copied joints (hand tips = wrist position) ----
    # Left hand tip (joint 7): copy left wrist
    ntu[7] = coco_kpts[_COCO_LEFT_WRIST].copy()
    # Right hand tip (joint 11): copy right wrist
    ntu[11] = coco_kpts[_COCO_RIGHT_WRIST].copy()
    # Left thumb (joint 22): copy left wrist
    ntu[22] = coco_kpts[_COCO_LEFT_WRIST].copy()
    # Right thumb (joint 23): copy right wrist
    ntu[23] = coco_kpts[_COCO_RIGHT_WRIST].copy()

    # ---- Fabricated joints (toes from ankle approximation) ----
    # Estimate leg length for toe offset
    l_leg_len = np.sqrt(
        (coco_kpts[_COCO_LEFT_HIP][0] - coco_kpts[_COCO_LEFT_ANKLE][0]) ** 2 +
        (coco_kpts[_COCO_LEFT_HIP][1] - coco_kpts[_COCO_LEFT_ANKLE][1]) ** 2
    ) if coco_kpts[_COCO_LEFT_HIP][2] > 0 and coco_kpts[_COCO_LEFT_ANKLE][2] > 0 else 50.0

    r_leg_len = np.sqrt(
        (coco_kpts[_COCO_RIGHT_HIP][0] - coco_kpts[_COCO_RIGHT_ANKLE][0]) ** 2 +
        (coco_kpts[_COCO_RIGHT_HIP][1] - coco_kpts[_COCO_RIGHT_ANKLE][1]) ** 2
    ) if coco_kpts[_COCO_RIGHT_HIP][2] > 0 and coco_kpts[_COCO_RIGHT_ANKLE][2] > 0 else 50.0

    l_toe_offset = l_leg_len * 0.05  # 5% of leg length
    r_toe_offset = r_leg_len * 0.05

    l_ankle = coco_kpts[_COCO_LEFT_ANKLE]
    r_ankle = coco_kpts[_COCO_RIGHT_ANKLE]

    # Left big toe (joint 15): ankle shifted down
    ntu[15, 0] = l_ankle[0]
    ntu[15, 1] = l_ankle[1] + l_toe_offset
    ntu[15, 2] = l_ankle[2] * 0.5  # lower confidence for fabricated joint

    # Right big toe (joint 19): ankle shifted down
    ntu[19, 0] = r_ankle[0]
    ntu[19, 1] = r_ankle[1] + r_toe_offset
    ntu[19, 2] = r_ankle[2] * 0.5

    # Left small toe (joint 20): ankle shifted slightly right (outward)
    ntu[20, 0] = l_ankle[0] + l_toe_offset * 0.5
    ntu[20, 1] = l_ankle[1] + l_toe_offset * 0.8
    ntu[20, 2] = l_ankle[2] * 0.3

    # Right small toe (joint 21): ankle shifted slightly left (outward)
    ntu[21, 0] = r_ankle[0] - r_toe_offset * 0.5
    ntu[21, 1] = r_ankle[1] + r_toe_offset * 0.8
    ntu[21, 2] = r_ankle[2] * 0.3

    return ntu


def coco_to_ntu_batch(coco_sequence: np.ndarray) -> np.ndarray:
    """Convert a sequence of COCO frames to NTU format.

    Args:
        coco_sequence: np.ndarray of shape (T, 17, 3).

    Returns:
        np.ndarray of shape (T, 25, 3).
    """
    T = coco_sequence.shape[0]
    ntu_sequence = np.zeros((T, 25, 3), dtype=np.float32)
    for t in range(T):
        ntu_sequence[t] = coco_to_ntu(coco_sequence[t])
    return ntu_sequence
