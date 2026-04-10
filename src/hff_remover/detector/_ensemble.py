"""Ensemble detector that combines results from multiple detectors."""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import numpy as np

from hff_remover.detector._base import BaseHFFDetector

_CANONICAL_LABELS = {"header", "footer", "footnote", "text-area"}


def _to_xyxy(bbox: list) -> list:
    """Convert a bbox in either format to axis-aligned ``[x1, y1, x2, y2]``."""
    if bbox and isinstance(bbox[0], (list, tuple)):
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
    return [float(v) for v in bbox]


class EnsembleDetector(BaseHFFDetector):
    """Ensemble detector that combines results from multiple detectors."""

    def __init__(
        self,
        detectors: List[BaseHFFDetector],
        merge_strategy: str = "union",
        iou_threshold: float = 0.5,
    ):
        """
        Initialize ensemble detector.

        Args:
            detectors: List of detector instances to use.
            merge_strategy: How to combine results ('union', 'intersection', 'cascade').
            iou_threshold: IoU threshold for merging overlapping boxes.
        """
        self.detectors = detectors
        self.merge_strategy = merge_strategy
        self.iou_threshold = iou_threshold

    def _normalize_detection_label(self, raw_label: Union[int, str]) -> Optional[str]:
        """Validate an already-normalised label from a child detector.

        Child detectors normalise labels in their own ``detect()`` calls,
        so the ensemble only needs to confirm the label is one of the
        canonical HFF values.

        Args:
            raw_label: The label string (expected to be pre-normalised).

        Returns:
            The label unchanged if canonical, otherwise ``None``.
        """
        label = str(raw_label)
        if label in _CANONICAL_LABELS:
            return label
        return None

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        image_size: int = 1024,
    ) -> List[Dict[str, Any]]:
        """
        Detect using all detectors and merge results.

        Args:
            image: Path to image file or numpy array.
            image_size: Input size for the model.

        Returns:
            Merged list of detections.
        """
        all_detections = []

        if self.merge_strategy == "cascade":
            # Use first detector, only use second if first returns nothing or fails
            for detector in self.detectors:
                try:
                    detections = detector.detect(image, image_size)
                    if detections:
                        return detections
                except Exception:
                    # This detector failed, try the next one
                    continue
            return []

        # Collect all detections (robust: continue if one detector fails)
        for detector in self.detectors:
            try:
                detections = detector.detect(image, image_size)
                all_detections.extend(detections)
            except Exception:
                # This detector failed, continue with others
                continue

        if self.merge_strategy == "union":
            # Merge overlapping boxes using NMS
            return self._non_max_suppression(all_detections)
        if self.merge_strategy == "intersection":
            # Only keep boxes detected by all detectors
            # (simplified: treat as union for now)
            return self._non_max_suppression(all_detections)

        return all_detections

    def detect_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        image_size: int = 1024,
        batch_size: int = 8,
    ) -> List[List[Dict[str, Any]]]:
        """Detect in multiple images."""
        return [self.detect(img, image_size) for img in images]

    def _non_max_suppression(
        self,
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Apply non-maximum suppression to merge overlapping boxes."""
        if not detections:
            return []

        # Sort by confidence
        sorted_dets = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        keep = []

        while sorted_dets:
            best = sorted_dets.pop(0)
            keep.append(best)

            # Remove boxes with high IoU with the best box
            sorted_dets = [
                det for det in sorted_dets
                if self._compute_iou(best["bbox"], det["bbox"]) < self.iou_threshold
            ]

        return keep

    def _compute_iou(self, box1: list, box2: list) -> float:
        """Compute IoU between two boxes (accepts polygon or xyxy format)."""
        b1 = _to_xyxy(box1)
        b2 = _to_xyxy(box2)
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        box1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
        box2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])

        union_area = box1_area + box2_area - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area
