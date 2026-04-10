"""Surya-based document layout HFF detector.

Uses the Surya LayoutPredictor to detect headers, footers, and footnotes.
Unlike YOLO-based detectors, Surya produces string labels (e.g. "page-header",
"page-footer", "footnote") that are mapped to normalised HFF categories.

Reference: https://github.com/VikParuchuri/surya
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from hff_remover.detector._base import BaseHFFDetector

# Surya layout detector (our labels: 0=text, 1=footer, 2=header, 3=footnote)
SURYA_HFF_CLASS_IDS: Dict[int, str] = {
    0: "text",
    1: "footer",
    2: "header",
    3: "footnote",
}

SURYA_LABEL_TO_OUR_CLASS: Dict[str, tuple] = {
    "Text": (0, "text-area"),
    "Caption": (0, "text-area"),
    "ListItem": (0, "text-area"),
    "Formula": (0, "text-area"),
    "TableOfContents": (0, "text-area"),
    "From": (0, "text-area"),
    "Table": (0, "text-area"),
    "Picture": (0, "text-area"),
    "Figure": (0, "text-area"),
    "SectionHeader": (0, "text-area"),
    "Code": (0, "text-area"),
    "PageHeader": (2, "header"),
    "PageFooter": (1, "footer"),
    "Footnote": (3, "footnote"),
}


def _ndarray_to_pil(image: np.ndarray) -> Image.Image:
    """Convert a numpy array (BGR or grayscale) to an RGB PIL Image.

    Args:
        image: Numpy array representing an image.

    Returns:
        PIL Image in RGB mode.
    """
    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")
    if image.shape[2] == 3:
        # Assume BGR (OpenCV convention) → convert to RGB
        return Image.fromarray(image[:, :, ::-1])
    return Image.fromarray(image).convert("RGB")


class SuryaLayoutDetector(BaseHFFDetector):
    """Detector for headers, footers, and footnotes using Surya layout analysis."""

    def __init__(
        self,
        confidence_threshold: float = 0.5,
    ) -> None:
        """Initialise the Surya layout detector.

        Args:
            confidence_threshold: Minimum confidence score for detections.
        """
        from surya.foundation import FoundationPredictor
        from surya.layout import LayoutPredictor
        from surya.settings import settings

        self.confidence_threshold = confidence_threshold
        self.layout_predictor = LayoutPredictor(
            FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_detection_label(
        self,
        raw_label: Union[int, str],
        default_to_text: bool = False,
    ) -> Optional[tuple]:
        """Normalise a Surya label to a standard HFF (class_id, class_name) pair.

        Args:
            raw_label: Raw label string from Surya's layout predictor.
            default_to_text: If ``True``, unmapped labels fall back to
                ``(0, "text")`` instead of ``None``.

        Returns:
            A ``(class_id, class_name)`` tuple, or ``None`` when the label
            is not HFF-relevant and *default_to_text* is ``False``.
        """
        
        normalised_label = SURYA_LABEL_TO_OUR_CLASS.get(raw_label)
        if normalised_label is not None:
            return normalised_label
        return (0, "text") if default_to_text else None

    # ------------------------------------------------------------------
    # Public API (BaseHFFDetector)
    # ------------------------------------------------------------------

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        image_size: int = 1024,
        filter_to_hff_only: bool = True,
        normalize_bbox: bool = True,
    ) -> List[Dict[str, Any]]:
        """Detect headers, footers, and footnotes in an image.

        Args:
            image: Numpy array (BGR format) of the image.
            image_size: Kept for API compatibility (unused by Surya).
            filter_to_hff_only: When ``True`` only HFF-mapped classes are
                kept; when ``False`` unmapped labels default to ``"text"``.
            normalize_bbox: When ``True`` (default), ``bbox`` is an
                axis-aligned ``[x1, y1, x2, y2]`` list.  When ``False``,
                ``bbox`` contains the raw polygon corners
                ``[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]`` as returned by
                Surya, which may represent a rotated quadrilateral.

        Returns:
            List of detection dicts with ``bbox``, ``class_id``,
            ``class_name``, and ``confidence`` keys.
        """
        _ = image_size
        pil_image = _ndarray_to_pil(image)
        layout_predictions = self.layout_predictor([pil_image])
        if not layout_predictions:
            return []

        layout_pred = layout_predictions[0]

        # --- extract bboxes from prediction ---
        if hasattr(layout_pred, "bboxes"):
            bboxes_data = layout_pred.bboxes or []
        elif isinstance(layout_pred, dict):
            bboxes_data = (
                layout_pred.get("bboxes")
                or layout_pred.get("boxes")
                or []
            )
        else:
            bboxes_data = []

        detections: List[Dict[str, Any]] = []

        for item in bboxes_data:
            # --- extract raw label ---
            if isinstance(item, dict):
                raw_label = item.get("label") or item.get("type") or ""
            else:
                raw_label = getattr(item, "label", "")

            # --- extract bbox and polygon ---
            if isinstance(item, dict):
                raw_polygon = item.get("polygon")
                raw_bbox = item.get("bbox")
            else:
                raw_polygon = getattr(item, "polygon", None)
                raw_bbox = getattr(item, "bbox", None)

            # --- extract confidence ---
            if isinstance(item, dict):
                top_k = item.get("top_k") or {}
            else:
                top_k = getattr(item, "top_k", None) or {}

            if isinstance(item, dict) and "confidence" in item:
                conf = float(item["confidence"])
            elif hasattr(item, "confidence") and item.confidence is not None:
                conf = float(item.confidence)
            elif isinstance(top_k, dict) and top_k:
                key = (
                    str(raw_label).strip().lower().replace(" ", "-")
                    if raw_label
                    else ""
                )
                value = (
                    top_k.get(raw_label)
                    or top_k.get(key)
                    or next(iter(top_k.values()))
                )
                conf = float(value)
            else:
                conf = 1.0

            if conf < self.confidence_threshold:
                continue

            # --- map to our class ---
            raw_label = str(raw_label or "")
            mapped = self._normalize_detection_label(
                raw_label,
                default_to_text=not filter_to_hff_only,
            )
            if mapped is None:
                continue
            our_class_id, our_class_name = mapped

            # --- build output bbox ---
            if not raw_bbox or len(raw_bbox) < 4:
                continue

            if normalize_bbox:
                output_bbox: Any = list(map(float, raw_bbox))
            else:
                if raw_polygon and len(raw_polygon) == 4:
                    output_bbox = [[float(c) for c in pt] for pt in raw_polygon]
                else:
                    output_bbox = list(map(float, raw_bbox))

            detections.append(
                {
                    "bbox": output_bbox,
                    "class_id": our_class_id,
                    "class_name": our_class_name,
                    "confidence": conf,
                }
            )

        return detections

    def detect_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        image_size: int = 1024,
        batch_size: int = 8,
        normalize_bbox: bool = True,
    ) -> List[List[Dict[str, Any]]]:
        """Detect headers, footers, and footnotes in multiple images.

        Args:
            images: List of numpy arrays (BGR format).
            image_size: Kept for API compatibility (unused by Surya).
            batch_size: Kept for API compatibility (Surya handles batching
                internally).
            normalize_bbox: When ``True`` (default), bboxes are
                axis-aligned ``[x1, y1, x2, y2]``.  When ``False``,
                bboxes are raw polygon corners from Surya.

        Returns:
            List of detection lists, one per input image.
        """
        _ = image_size, batch_size
        return [
            self.detect(image, normalize_bbox=normalize_bbox)
            for image in images
        ]

    def get_all_detections(
        self,
        image: Union[str, Path, np.ndarray],
        image_size: int = 1024,
        normalize_bbox: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return all document-layout detections (not just HFF).

        Useful for debugging or visualisation.

        Args:
            image: Numpy array (BGR format) of the image.
            image_size: Kept for API compatibility (unused by Surya).
            normalize_bbox: When ``True`` (default), bboxes are
                axis-aligned ``[x1, y1, x2, y2]``.  When ``False``,
                bboxes are raw polygon corners from Surya.

        Returns:
            List of detection dicts for all Surya layout classes.
        """
        return self.detect(
            image,
            image_size=image_size,
            filter_to_hff_only=False,
            normalize_bbox=normalize_bbox,
        )
