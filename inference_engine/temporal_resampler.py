"""Temporal resampling for skeleton sequences.

ST-GCN requires exactly T=64 frames with NTU-25 joints. This module handles:
- Padding short sequences (<64 frames) by repeating the last frame
- Subsampling long sequences (>64 frames) with uniform spacing
- Passing through exact-length sequences unchanged
- Per-frame body-relative normalization (center on spine, scale by torso)

Padding with repeated frames is preferred over zero-padding because
zero skeletons create artifacts in graph convolution layers.
"""

import json
import os
import numpy as np


DEFAULT_T = 64  # ST-GCN default temporal window

# Training normalization stats (global min-max from NTU60 training set)
# Loaded lazily on first use
_NORM_STATS = None


def _load_norm_stats():
    """Load global normalization statistics from models/stgcn_norm_stats.json."""
    global _NORM_STATS
    if _NORM_STATS is not None:
        return _NORM_STATS
    stats_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "models", "stgcn_norm_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            _NORM_STATS = json.load(f)
    else:
        # Fallback: approximate NTU60 stats
        _NORM_STATS = {"x_min": -264.25, "x_max": 2274.0, "y_min": -134.75, "y_max": 1372.0}
    return _NORM_STATS


def prepare_temporal_input(skeleton_sequence: np.ndarray, T: int = DEFAULT_T) -> np.ndarray:
    """Resample a variable-length skeleton sequence to exactly T frames.

    Args:
        skeleton_sequence: np.ndarray of shape (T_in, V, C) where
            T_in = actual number of frames (variable)
            V = number of joints (25 for NTU)
            C = channels per joint (3: x, y, confidence)
        T: target number of frames (default 64 for ST-GCN)

    Returns:
        np.ndarray of shape (T, V, C), dtype float32.
    """
    if skeleton_sequence.ndim != 3:
        raise ValueError(f"Expected (T, V, C) array, got shape {skeleton_sequence.shape}")

    T_in, V, C = skeleton_sequence.shape

    if T_in == T:
        # Exact match, no resampling needed
        return skeleton_sequence.astype(np.float32)

    if T_in > T:
        # Subsample: uniform spacing
        indices = np.linspace(0, T_in - 1, T, dtype=int)
        return skeleton_sequence[indices].astype(np.float32)

    # T_in < T: pad by repeating last frame
    frames = list(skeleton_sequence)
    last_frame = skeleton_sequence[-1] if T_in > 0 else np.zeros((V, C), dtype=np.float32)

    while len(frames) < T:
        frames.append(last_frame)

    return np.stack(frames, axis=0).astype(np.float32)


def normalize_skeleton(ntu_sequence: np.ndarray) -> np.ndarray:
    """Normalize NTU skeleton coordinates for ST-GCN inference.

    Normalization strategy (matching training preprocessing):
    Global min-max normalization to [-1, 1] range using pre-computed
    statistics from the NTU60 training set.

    Args:
        ntu_sequence: np.ndarray of shape (T, 25, 3) with [x, y, conf].

    Returns:
        np.ndarray of shape (T, 25, 3), normalized. dtype float32.
    """
    if ntu_sequence.ndim != 3:
        raise ValueError(f"Expected (T, V, C) array, got shape {ntu_sequence.shape}")

    result = ntu_sequence.copy().astype(np.float32)
    stats = _load_norm_stats()

    # Apply global min-max normalization to x, y channels
    # x channel: 2 * (x - x_min) / (x_max - x_min) - 1
    x_range = stats["x_max"] - stats["x_min"]
    if x_range > 1e-6:
        result[:, :, 0] = 2 * (result[:, :, 0] - stats["x_min"]) / x_range - 1

    # y channel: 2 * (y - y_min) / (y_max - y_min) - 1
    y_range = stats["y_max"] - stats["y_min"]
    if y_range > 1e-6:
        result[:, :, 1] = 2 * (result[:, :, 1] - stats["y_min"]) / y_range - 1

    # Confidence channel unchanged
    return result


def prepare_ctrgcn_input(ntu_sequence: np.ndarray) -> np.ndarray:
    """Full pipeline: resample → normalize → transpose to ST-GCN input format.

    Combines temporal resampling, normalization, and dimension transposition
    into a single call for convenience.

    Args:
        ntu_sequence: np.ndarray of shape (T_in, 25, 3) raw NTU keypoints.

    Returns:
        np.ndarray of shape (1, 3, 64, 25, 2), ready for ST-GCN ONNX model.
    """
    # Step 1: Resample to T=64
    resampled = prepare_temporal_input(ntu_sequence, T=DEFAULT_T)

    # Step 2: Normalize
    normalized = normalize_skeleton(resampled)

    # Step 3: Transpose (T, V, C) → (C, T, V)
    data = np.transpose(normalized, (2, 0, 1))  # (3, 64, 25)

    # Step 4: Add batch and person dimensions → (1, C, T, V, M=2)
    data = data[np.newaxis, :, :, :, np.newaxis]  # (1, 3, 64, 25, 1)
    padding = np.zeros_like(data)
    data = np.concatenate([data, padding], axis=4)  # (1, 3, 64, 25, 2)

    return data.astype(np.float32)
