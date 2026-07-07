"""Shared ONNX utilities."""

import onnxruntime as ort
from loguru import logger


def get_onnx_providers() -> list:
    """Return best available ONNX execution providers.

    Priority: CUDA > TensorRT > DirectML > CPU
    DirectML provides GPU acceleration via Windows DirectX on any GPU.
    Falls back to CPU when no accelerator is available.
    """
    available = ort.get_available_providers()

    preferred = [
        "CUDAExecutionProvider",
        "TensorrtExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]

    providers = [p for p in preferred if p in available]
    if not providers:
        providers = ["CPUExecutionProvider"]

    if providers[0] != "CPUExecutionProvider":
        logger.info(f"GPU acceleration enabled: {providers[0]}")
    else:
        logger.info("No GPU provider available, using CPU")
    return providers
