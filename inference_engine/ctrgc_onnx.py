"""ONNX Runtime inference wrapper for ST-GCN action recognition.

Loads an ST-GCN ONNX model and runs inference on pre-processed
NTU 25-joint skeleton sequences. Handles lazy loading, error recovery,
and input tensor preparation.

The expected ONNX model input shape is (N, C, T, V, M) = (1, 3, 64, 25, 2).
The output is (N, num_classes) logits before softmax.
"""

import os
import numpy as np
from loguru import logger

from .utils import get_onnx_providers


class CTRGCNInference:
    """ONNX Runtime inference engine for ST-GCN skeleton action recognition.

    Supports lazy loading — the model is only loaded on first inference call.
    If the model file is missing, `available` returns False and all infer()
    calls return None gracefully.
    """

    def __init__(self, model_path: str = "models/stgcn.onnx"):
        self._model_path = model_path
        self._session = None
        self._input_name: str = ""
        self._num_classes: int = 60  # NTU60 default
        self._loaded = False
        self._load_failed = False

    @property
    def available(self) -> bool:
        """True if the ONNX model is loaded and ready for inference."""
        if not self._loaded and not self._load_failed:
            self._ensure_loaded()
        return self._session is not None

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def _ensure_loaded(self) -> bool:
        """Attempt to load the ONNX model. Returns True on success."""
        if self._loaded:
            return self._session is not None
        if self._load_failed:
            return False

        if not os.path.exists(self._model_path):
            logger.warning(
                f"ST-GCN model not found: {self._model_path} — "
                "falling back to heuristic action recognition"
            )
            self._load_failed = True
            return False

        try:
            import onnxruntime as ort
            providers = get_onnx_providers()
            self._session = ort.InferenceSession(
                self._model_path, providers=providers
            )
            self._input_name = self._session.get_inputs()[0].name

            # Detect number of classes from output shape
            out_shape = self._session.get_outputs()[0].shape
            if len(out_shape) >= 2 and isinstance(out_shape[1], int):
                self._num_classes = out_shape[1]

            self._loaded = True
            logger.info(
                f"ST-GCN model loaded: {self._model_path} "
                f"({self._num_classes} classes, providers={providers})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load ST-GCN model: {e}")
            self._load_failed = True
            return False

    def infer(self, input_tensor: np.ndarray) -> np.ndarray | None:
        """Run ST-GCN inference on a pre-processed input tensor.

        Args:
            input_tensor: np.ndarray of shape (1, 3, 64, 25, 2), float32.
                Use `temporal_resampler.prepare_ctrgcn_input()` to create this.

        Returns:
            np.ndarray of shape (num_classes,) with softmax probabilities,
            or None if inference fails.
        """
        if not self.available:
            return None

        try:
            # Ensure correct shape
            if input_tensor.ndim == 4:
                # (C, T, V, M) → add batch dim
                input_tensor = input_tensor[np.newaxis]
            if input_tensor.ndim != 5:
                logger.error(f"ST-GCN input must be 5D, got {input_tensor.ndim}D")
                return None

            outputs = self._session.run(None, {self._input_name: input_tensor})

            # Output is (1, num_classes) logits
            logits = outputs[0]
            if logits.ndim == 2:
                logits = logits[0]  # remove batch dim

            # Apply softmax
            probs = self._softmax(logits)
            return probs

        except Exception as e:
            logger.error(f"ST-GCN inference error: {e}")
            return None

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e_x = np.exp(x - np.max(x))
        return e_x / (e_x.sum() + 1e-8)

    def reset(self) -> None:
        """Reset the model state (unload and reload on next use)."""
        self._session = None
        self._loaded = False
        self._load_failed = False
