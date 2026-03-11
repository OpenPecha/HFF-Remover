"""PP-DocLayout (PaddlePaddle) based HFF detector."""

from pathlib import Path
from typing import List, Dict, Any, Union

import numpy as np

from hff_remover.detector._base import BaseHFFDetector

# PP-DocLayout class mapping for HFF + text
# Reference: https://huggingface.co/PaddlePaddle/PP-DocLayout-L
PP_DOCLAYOUT_HFF_CLASSES = {
    "header",
    "footer",
    "footnote",
    "footnotes",       # Alternative naming
    "page_number",
    "page-header",
    "page-footer",
    "text",            # Text area / body text
    "plain text",      # Alternative naming
    "plain_text",      # Alternative naming
    "paragraph",       # Alternative naming
}


class PPDocLayoutDetector(BaseHFFDetector):
    """Detector for headers, footers, and footnotes using PP-DocLayout (PaddlePaddle)."""

    def __init__(
        self,
        model_name: str = "PP-DocLayout-L",
        confidence_threshold: float = 0.5,
        use_gpu: bool = False,
    ):
        """
        Initialize the PP-DocLayout detector.

        Args:
            model_name: Model name (not used, kept for API compatibility).
            confidence_threshold: Minimum confidence score for detections.
            use_gpu: Whether to use GPU for inference.
        """
        from paddleocr import PPStructure

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.use_gpu = use_gpu

        # Initialize PPStructure for layout analysis
        self.model = PPStructure(
            layout=True,
            table=False,
            ocr=False,
            show_log=False,
            use_gpu=use_gpu,
            enable_mkldnn=False,  # Disable MKLDNN to avoid bugs
        )

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        image_size: int = 1024,
    ) -> List[Dict[str, Any]]:
        """
        Detect headers, footers, and footnotes in an image.

        Args:
            image: Path to image file or numpy array.
            image_size: Input size (not used for PP-DocLayout, kept for API compatibility).

        Returns:
            List of detection dictionaries with keys:
                - bbox: [x1, y1, x2, y2] bounding box coordinates
                - class_id: Class ID (mapped to string label)
                - class_name: Human-readable class name
                - confidence: Confidence score
        """
        import cv2

        # Load image if path provided
        if isinstance(image, (str, Path)):
            img_array = cv2.imread(str(image))
        else:
            img_array = image

        # Run inference - PPStructure takes numpy array directly
        results = self.model(img_array)

        detections = []

        # PPStructure returns list of dicts with 'type' and 'bbox' keys
        for item in results:
            if not isinstance(item, dict):
                continue

            label = item.get('type', '').lower()
            # PPStructure doesn't always provide confidence, default to 1.0
            score = item.get('score', 1.0)
            bbox = item.get('bbox', [])

            # Filter by confidence
            if score < self.confidence_threshold:
                continue

            # Only keep HFF classes
            if label not in PP_DOCLAYOUT_HFF_CLASSES:
                continue

            # Normalize label name
            normalized_label = self._normalize_label(label)

            # Convert bbox to [x1, y1, x2, y2] format
            if len(bbox) == 4:
                bbox = list(map(float, bbox))
            elif len(bbox) == 8:
                # Polygon format: get bounding box
                xs = bbox[0::2]
                ys = bbox[1::2]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
            else:
                continue

            detections.append({
                "bbox": bbox,
                "class_id": label,
                "class_name": normalized_label,
                "confidence": float(score),
            })

        return detections

    def detect_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        image_size: int = 1024,
        batch_size: int = 8,
    ) -> List[List[Dict[str, Any]]]:
        """
        Detect headers, footers, and footnotes in multiple images.

        Args:
            images: List of image paths or numpy arrays.
            image_size: Input size (not used for PP-DocLayout).
            batch_size: Number of images to process at once.

        Returns:
            List of detection lists, one per input image.
        """
        all_detections = []

        for image in images:
            detections = self.detect(image, image_size)
            all_detections.append(detections)

        return all_detections

    def _normalize_label(self, label: str) -> str:
        """Normalize label names to standard format."""
        label = label.lower().replace("-", "_").replace(" ", "_")

        if label in ("header", "page_header"):
            return "header"
        elif label in ("footer", "page_footer", "page_number"):
            return "footer"
        elif label in ("footnote", "footnotes"):
            return "footnote"
        elif label in ("text", "plain_text", "paragraph"):
            return "text-area"

        return label

    def get_all_detections(
        self,
        image: Union[str, Path, np.ndarray],
        image_size: int = 1024,
    ) -> List[Dict[str, Any]]:
        """
        Get all document layout detections (not just HFF).

        Args:
            image: Path to image file or numpy array.
            image_size: Input size.

        Returns:
            List of all detection dictionaries.
        """
        _ = image_size  # kept for API compatibility
        import cv2

        # Load image if path provided
        if isinstance(image, (str, Path)):
            img_array = cv2.imread(str(image))
        else:
            img_array = image

        # Run inference
        results = self.model(img_array)

        detections = []

        for item in results:
            if not isinstance(item, dict):
                continue

            label = item.get('type', '')
            score = item.get('score', 1.0)
            bbox = item.get('bbox', [])

            if score < self.confidence_threshold:
                continue

            # Convert bbox to [x1, y1, x2, y2] format
            if len(bbox) == 4:
                bbox = list(map(float, bbox))
            elif len(bbox) == 8:
                xs = bbox[0::2]
                ys = bbox[1::2]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
            else:
                continue

            detections.append({
                "bbox": bbox,
                "class_id": label,
                "class_name": label,
                "confidence": float(score),
            })

        return detections
