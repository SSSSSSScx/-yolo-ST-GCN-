from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class CameraInterface(ABC):
    @abstractmethod
    def open(self) -> bool:
        ...

    @abstractmethod
    def read_frame(self) -> Optional[np.ndarray]:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @abstractmethod
    def get_info(self) -> dict:
        ...

    @abstractmethod
    def is_healthy(self) -> bool:
        ...
