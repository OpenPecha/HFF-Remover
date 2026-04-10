"""Abstract base class for HFF detectors."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod

import numpy as np


class BaseHFFDetector(ABC):
    """Abstract base class for HFF detectors."""

    @abstractmethod
    def _normalize_detection_label(self, raw_label: Union[int, str]) -> Optional[str]:
        """Normalise a model-specific label to a standard HFF label.

        Each detector maps its own raw label (an ``int`` class ID or a
        ``str`` class name, depending on the model) to one of the
        canonical labels: ``"header"``, ``"footer"``, ``"footnote"``,
        ``"text-area"``, or returns ``None`` to discard the detection.

        Args:
            raw_label: The raw class identifier produced by the model.

        Returns:
            A normalised label string, or ``None`` if the class is not
            HFF-relevant.
        """

    @abstractmethod
    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        image_size: int = 1024,
    ) -> List[Dict[str, Any]]:
        """Detect headers, footers, and footnotes in an image.

        Args:
            image: Path to image file or numpy array.
            image_size: Input size for the model.

        Returns:
            List of detection dictionaries with keys:
                - bbox: Bounding box coordinates.  By default this is an
                  axis-aligned ``[x1, y1, x2, y2]`` list.  Subclasses
                  that accept a ``normalize_bbox`` parameter may return
                  polygon corners ``[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]``
                  when ``normalize_bbox=False``.
                - class_id: Class ID of the detection
                - class_name: Human-readable class name
                - confidence: Confidence score
        """

    @abstractmethod
    def detect_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        image_size: int = 1024,
        batch_size: int = 8,
    ) -> List[List[Dict[str, Any]]]:
        """Detect HFF in multiple images."""
