"""Surya2-based document layout HFF detector.

Uses the newer SuryaInferenceManager + LayoutPredictor pipeline (as seen in
run_layout.py) to detect headers, footers, and footnotes.  This differs from
the original SuryaLayoutDetector which uses FoundationPredictor directly.

Reference: https://github.com/VikParuchuri/surya
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from hff_remover.detector._base import BaseHFFDetector

SURYA2_HFF_CLASS_IDS: Dict[int, str] = {
    0: "text",
    1: "footer",
    2: "header",
    3: "footnote",
}

SURYA2_LABEL_TO_OUR_CLASS: Dict[str, tuple] = {
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
        return Image.fromarray(image[:, :, ::-1])
    return Image.fromarray(image).convert("RGB")


class Surya2LayoutDetector(BaseHFFDetector):
    """HFF detector using the SuryaInferenceManager-based layout pipeline.

    This detector uses the newer Surya inference API which manages model
    lifecycle through ``SuryaInferenceManager`` and produces ``LayoutResult``
    objects with richer metadata (position, polygon, top_k scores).
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        keep_server: bool = False,
    ) -> None:
        """Initialise the Surya2 layout detector.

        Args:
            confidence_threshold: Minimum confidence score for detections.
            keep_server: If ``True``, keep the Surya inference server alive
                after predictions for faster reuse across calls.
        """
        from surya.inference import SuryaInferenceManager
        from surya.layout import LayoutPredictor
        from surya.settings import settings

        if keep_server:
            settings.SURYA_INFERENCE_KEEP_ALIVE = True

        self.confidence_threshold = confidence_threshold
        self._manager = SuryaInferenceManager()
        self._layout_predictor = LayoutPredictor(self._manager)

    def _normalize_detection_label(
        self,
        raw_label: Union[int, str],
        default_to_text: bool = False,
    ) -> Optional[tuple]:
        """Normalise a Surya label to a standard HFF (class_id, class_name) pair.

        Args:
            raw_label: Raw label string from Surya's layout predictor.
            default_to_text: If ``True``, unmapped labels fall back to
                ``(0, "text-area")`` instead of ``None``.

        Returns:
            A ``(class_id, class_name)`` tuple, or ``None`` when the label
            is not HFF-relevant and *default_to_text* is ``False``.
        """
        normalised = SURYA2_LABEL_TO_OUR_CLASS.get(str(raw_label))
        if normalised is not None:
            return normalised
        return (0, "text-area") if default_to_text else None

    def _extract_detections_from_result(
        self,
        layout_result: Any,
        filter_to_hff_only: bool = True,
        normalize_bbox: bool = True,
    ) -> List[Dict[str, Any]]:
        """Extract detection dicts from a single LayoutResult.

        Args:
            layout_result: A ``LayoutResult`` object from Surya.
            filter_to_hff_only: Keep only HFF-relevant classes.
            normalize_bbox: Return axis-aligned bbox when ``True``.

        Returns:
            List of detection dictionaries.
        """
        if hasattr(layout_result, "error") and layout_result.error:
            return []

        bboxes_data = getattr(layout_result, "bboxes", None) or []
        detections: List[Dict[str, Any]] = []

        for item in bboxes_data:
            raw_label = getattr(item, "label", "") or ""
            raw_polygon = getattr(item, "polygon", None)
            raw_bbox = getattr(item, "bbox", None)

            top_k = getattr(item, "top_k", None) or {}
            conf: float
            if hasattr(item, "confidence") and item.confidence is not None:
                conf = float(item.confidence)
            elif isinstance(top_k, dict) and top_k:
                key = str(raw_label).strip().lower().replace(" ", "-")
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

            mapped = self._normalize_detection_label(
                raw_label,
                default_to_text=not filter_to_hff_only,
            )
            if mapped is None:
                continue
            our_class_id, our_class_name = mapped

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

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        image_size: int = 1024,
        filter_to_hff_only: bool = True,
        normalize_bbox: bool = True,
    ) -> List[Dict[str, Any]]:
        """Detect headers, footers, and footnotes in an image.

        Args:
            image: Path to an image, a ``pathlib.Path``, or a numpy array
                in BGR format (OpenCV convention).
            image_size: Kept for API compatibility (unused by Surya).
            filter_to_hff_only: When ``True`` only HFF-mapped classes are
                kept; when ``False`` unmapped labels default to text-area.
            normalize_bbox: When ``True`` (default), ``bbox`` is an
                axis-aligned ``[x1, y1, x2, y2]`` list.  When ``False``,
                ``bbox`` contains raw polygon corners.

        Returns:
            List of detection dicts with ``bbox``, ``class_id``,
            ``class_name``, and ``confidence`` keys.
        """
        _ = image_size

        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = _ndarray_to_pil(image)

        results = self._layout_predictor([pil_image])
        if not results:
            return []

        return self._extract_detections_from_result(
            results[0],
            filter_to_hff_only=filter_to_hff_only,
            normalize_bbox=normalize_bbox,
        )

    def detect_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        image_size: int = 1024,
        batch_size: int = 8,
        normalize_bbox: bool = True,
    ) -> List[List[Dict[str, Any]]]:
        """Detect headers, footers, and footnotes in multiple images.

        Surya handles batching internally via the inference manager, so all
        images are submitted together for efficient GPU utilisation.

        Args:
            images: List of image paths or numpy arrays (BGR format).
            image_size: Kept for API compatibility (unused by Surya).
            batch_size: Kept for API compatibility (Surya handles batching
                internally).
            normalize_bbox: When ``True`` (default), bboxes are
                axis-aligned ``[x1, y1, x2, y2]``.

        Returns:
            List of detection lists, one per input image.
        """
        _ = image_size, batch_size

        pil_images: List[Image.Image] = []
        for img in images:
            if isinstance(img, (str, Path)):
                pil_images.append(Image.open(img).convert("RGB"))
            else:
                pil_images.append(_ndarray_to_pil(img))

        results = self._layout_predictor(pil_images)

        all_detections: List[List[Dict[str, Any]]] = []
        for result in results:
            all_detections.append(
                self._extract_detections_from_result(
                    result,
                    filter_to_hff_only=True,
                    normalize_bbox=normalize_bbox,
                )
            )

        return all_detections

    def get_all_detections(
        self,
        image: Union[str, Path, np.ndarray],
        image_size: int = 1024,
        normalize_bbox: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return all document-layout detections (not just HFF).

        Useful for debugging or visualisation.

        Args:
            image: Path or numpy array (BGR format) of the image.
            image_size: Kept for API compatibility (unused by Surya).
            normalize_bbox: When ``True`` (default), bboxes are
                axis-aligned ``[x1, y1, x2, y2]``.

        Returns:
            List of detection dicts for all Surya layout classes.
        """
        return self.detect(
            image,
            image_size=image_size,
            filter_to_hff_only=False,
            normalize_bbox=normalize_bbox,
        )
