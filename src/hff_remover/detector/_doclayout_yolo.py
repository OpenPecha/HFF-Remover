"""DocLayout-YOLO based HFF detector."""

from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import numpy as np

from hff_remover.detector._base import BaseHFFDetector

# DocLayout-YOLO class mapping
# Reference: https://github.com/opendatalab/DocLayout-YOLO
DOCLAYOUT_YOLO_CLASS_NAMES = {
    0: "title",
    1: "plain_text",
    2: "abandon",
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "isolate_formula",
    9: "formula_caption",
}

# Classes we want to detect (Header, Footer, Footnote, Text)
# In DocLayout-YOLO, "abandon" (class 2) represents headers/footers/page numbers
# "table_footnote" (class 7) represents footnotes in tables
# "plain_text" (class 1) represents text areas / body text
DOCLAYOUT_YOLO_HFF_CLASSES: Dict[int, str] = {
    1: "plain_text",     # Text area / body text
    2: "abandon",        # Headers, footers, page numbers
    7: "table_footnote", # Table footnotes
}

# Map raw DocLayout-YOLO class names to normalised HFF labels.
# "abandon" is kept as-is here; it requires positional resolution to
# become "header" or "footer" (handled by the detector class).
_DOCLAYOUT_LABEL_MAP: Dict[str, str] = {
    "plain_text": "text-area",
    "table_footnote": "footnote",
    "abandon": "abandon",  # sentinel — resolved by position later
}


class HFFDetector(BaseHFFDetector):
    """Detector for headers, footers, and footnotes using DocLayout-YOLO."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        header_region_ratio: float = 0.33,
        footer_region_ratio: float = 0.67,
    ):
        """
        Initialize the HFF detector.

        Args:
            model_path: Path to the DocLayout-YOLO model weights.
                       If None, downloads from HuggingFace Hub.
            device: Device to run inference on ('cuda' or 'cpu').
            confidence_threshold: Minimum confidence score for detections.
        """
        from doclayout_yolo import YOLOv10
        from huggingface_hub import hf_hub_download

        self.device = device
        self.confidence_threshold = confidence_threshold
        self.header_region_ratio = header_region_ratio
        self.footer_region_ratio = footer_region_ratio

        if not (0.0 <= self.header_region_ratio <= 1.0) or not (0.0 <= self.footer_region_ratio <= 1.0):
            raise ValueError("header_region_ratio and footer_region_ratio must be within [0, 1]")
        if self.header_region_ratio >= self.footer_region_ratio:
            raise ValueError("header_region_ratio must be < footer_region_ratio")

        if model_path is None:
            # Download model from HuggingFace Hub
            model_path = hf_hub_download(
                repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
                filename="doclayout_yolo_docstructbench_imgsz1024.pt",
            )

        self.model = YOLOv10(model_path)
        self.model_path = model_path

    def _normalize_detection_label(self, raw_label: Union[int, str]) -> Optional[str]:
        """Normalise a DocLayout-YOLO class ID to a standard HFF label.

        Args:
            raw_label: Integer class ID from DocLayout-YOLO.

        Returns:
            Normalised label (``"text-area"``, ``"footnote"``, or the
            sentinel ``"abandon"``), or ``None`` if not HFF-relevant.
        """
        class_id = int(raw_label)
        if class_id not in DOCLAYOUT_YOLO_HFF_CLASSES:
            return None

        class_name = DOCLAYOUT_YOLO_HFF_CLASSES[class_id]
        return _DOCLAYOUT_LABEL_MAP.get(class_name)

    def _get_image_height(self, image: Union[str, Path, np.ndarray]) -> Optional[int]:
        """Best-effort extraction of image height for positional labeling."""
        if isinstance(image, np.ndarray):
            return int(image.shape[0])
        try:
            import cv2
            img_arr = cv2.imread(str(image))
            if img_arr is None:
                return None
            return int(img_arr.shape[0])
        except Exception:
            return None

    def _doclayout_to_hff_detection(
        self,
        class_id: int,
        bbox: List[float],
        confidence: float,
        image_height: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """Convert DocLayout-YOLO box into an HFF detection dict, or None to ignore."""
        label = self._normalize_detection_label(class_id)
        if label is None:
            return None

        # "abandon" is a sentinel that needs positional resolution.
        if label == "abandon":
            if image_height is None:
                label = "abandon"
            else:
                resolved = self._classify_abandon_by_position(bbox, image_height)
                if resolved is None:
                    return None
                label = resolved

        return {
            "bbox": bbox,
            "class_id": class_id,
            "class_name": label,
            "confidence": confidence,
        }

    def _get_result_image_height(self, result: Any, src: Union[str, Path, np.ndarray]) -> Optional[int]:
        """Best-effort extraction of image height for a single prediction result."""
        try:
            if hasattr(result, "orig_shape") and result.orig_shape is not None:
                return int(result.orig_shape[0])
        except Exception:
            pass
        return self._get_image_height(src)

    def _extract_hff_detections_from_boxes(
        self,
        boxes: Any,
        image_height: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Extract HFF detections from a DocLayout-YOLO boxes object."""
        if boxes is None:
            return []

        detections: List[Dict[str, Any]] = []
        for j in range(len(boxes)):
            class_id = int(boxes.cls[j].item())
            bbox = boxes.xyxy[j].cpu().numpy().tolist()
            confidence = float(boxes.conf[j].item())

            det = self._doclayout_to_hff_detection(
                class_id=class_id,
                bbox=bbox,
                confidence=confidence,
                image_height=image_height,
            )
            if det is not None:
                detections.append(det)

        return detections

    def _classify_abandon_by_position(
        self,
        bbox_xyxy: List[float],
        image_height: int,
    ) -> Optional[str]:
        """
        For DocLayout-YOLO class_id=2 ("abandon"), decide whether it's a header/footer.

        - If bbox vertical center is in top `header_region_ratio` of page -> "header"
        - If bbox vertical center is in bottom `footer_region_ratio` of page -> "footer"
        - Otherwise -> None (ignore)
        """
        if image_height <= 0:
            return None

        y1 = float(bbox_xyxy[1])
        y2 = float(bbox_xyxy[3])
        y_center = (y1 + y2) / 2.0
        y_norm = y_center / float(image_height)

        if y_norm <= self.header_region_ratio:
            return "header"
        if y_norm >= self.footer_region_ratio:
            return "footer"
        return None

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        image_size: int = 1024,
    ) -> List[Dict[str, Any]]:
        """
        Detect headers, footers, and footnotes in an image.

        Args:
            image: Path to image file or numpy array (BGR format).
            image_size: Input size for the model.

        Returns:
            List of detection dictionaries with keys:
                - bbox: [x1, y1, x2, y2] bounding box coordinates
                - class_id: Class ID of the detection
                - class_name: Human-readable class name
                - confidence: Confidence score
        """
        # Run inference
        results = self.model.predict(
            image,
            imgsz=image_size,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        # Determine image height for positional header/footer labeling
        image_height = self._get_image_height(image)

        detections: List[Dict[str, Any]] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())

                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                confidence = boxes.conf[i].item()

                det = self._doclayout_to_hff_detection(
                    class_id=class_id,
                    bbox=bbox,
                    confidence=float(confidence),
                    image_height=image_height,
                )
                if det is not None:
                    detections.append(det)

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
            image_size: Input size for the model.
            batch_size: Number of images to process at once.

        Returns:
            List of detection lists, one per input image.
        """
        all_detections = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            # Process batch
            results = self.model.predict(
                batch,
                imgsz=image_size,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
            )

            for idx, result in enumerate(results):
                src = batch[idx]
                image_height = self._get_result_image_height(result, src)
                all_detections.append(
                    self._extract_hff_detections_from_boxes(result.boxes, image_height)
                )

        return all_detections

    def get_all_detections(
        self,
        image: Union[str, Path, np.ndarray],
        image_size: int = 1024,
    ) -> List[Dict[str, Any]]:
        """
        Get all document layout detections (not just HFF).

        Useful for debugging or visualization.

        Args:
            image: Path to image file or numpy array.
            image_size: Input size for the model.

        Returns:
            List of all detection dictionaries.
        """
        results = self.model.predict(
            image,
            imgsz=image_size,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                confidence = boxes.conf[i].item()

                detections.append({
                    "bbox": bbox,
                    "class_id": class_id,
                    "class_name": DOCLAYOUT_YOLO_CLASS_NAMES.get(class_id, f"unknown_{class_id}"),
                    "confidence": confidence,
                })

        return detections
