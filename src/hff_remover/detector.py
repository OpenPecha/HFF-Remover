"""Document layout detectors for headers, footers, and footnotes."""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import numpy as np


# =============================================================================
# Base Detector Interface
# =============================================================================

class BaseHFFDetector(ABC):
    """Abstract base class for HFF detectors."""

    @abstractmethod
    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        image_size: int = 1024,
    ) -> List[Dict[str, Any]]:
        """
        Detect headers, footers, and footnotes in an image.

        Args:
            image: Path to image file or numpy array.
            image_size: Input size for the model.

        Returns:
            List of detection dictionaries with keys:
                - bbox: [x1, y1, x2, y2] bounding box coordinates
                - class_id: Class ID of the detection
                - class_name: Human-readable class name
                - confidence: Confidence score
        """
        pass

    @abstractmethod
    def detect_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        image_size: int = 1024,
        batch_size: int = 8,
    ) -> List[List[Dict[str, Any]]]:
        """Detect HFF in multiple images."""
        pass


# =============================================================================
# DocLayout-YOLO Detector
# =============================================================================

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

# Classes we want to detect and remove (Header, Footer, Footnote)
# In DocLayout-YOLO, "abandon" (class 2) represents headers/footers/page numbers
# "table_footnote" (class 7) represents footnotes in tables
DOCLAYOUT_YOLO_HFF_CLASSES = {
    2: "abandon",  # Headers, footers, page numbers
    7: "table_footnote",  # Table footnotes
}


class HFFDetector(BaseHFFDetector):
    """Detector for headers, footers, and footnotes using DocLayout-YOLO."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
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

        if model_path is None:
            # Download model from HuggingFace Hub
            model_path = hf_hub_download(
                repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
                filename="doclayout_yolo_docstructbench_imgsz1024.pt",
            )

        self.model = YOLOv10(model_path)
        self.model_path = model_path

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

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())

                # Only keep HFF classes (abandon and table_footnote)
                if class_id not in DOCLAYOUT_YOLO_HFF_CLASSES:
                    continue

                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                confidence = boxes.conf[i].item()

                detections.append({
                    "bbox": bbox,
                    "class_id": class_id,
                    "class_name": DOCLAYOUT_YOLO_HFF_CLASSES[class_id],
                    "confidence": confidence,
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

            for result in results:
                detections = []
                boxes = result.boxes

                if boxes is not None:
                    for j in range(len(boxes)):
                        class_id = int(boxes.cls[j].item())

                        if class_id not in DOCLAYOUT_YOLO_HFF_CLASSES:
                            continue

                        bbox = boxes.xyxy[j].cpu().numpy().tolist()
                        confidence = boxes.conf[j].item()

                        detections.append({
                            "bbox": bbox,
                            "class_id": class_id,
                            "class_name": DOCLAYOUT_YOLO_HFF_CLASSES[class_id],
                            "confidence": confidence,
                        })

                all_detections.append(detections)

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


# =============================================================================
# PP-DocLayout-L Detector (PaddlePaddle)
# =============================================================================

# PP-DocLayout class mapping for HFF
# Reference: https://huggingface.co/PaddlePaddle/PP-DocLayout-L
PP_DOCLAYOUT_HFF_CLASSES = {
    "header",
    "footer", 
    "footnote",
    "footnotes",  # Alternative naming
    "page_number",
    "page-header",
    "page-footer",
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
        elif label in ("footer", "page_footer"):
            return "footer"
        elif label in ("footnote", "footnotes"):
            return "footnote"
        elif label == "page_number":
            return "page_number"

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


# =============================================================================
# Ensemble Detector
# =============================================================================

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
        elif self.merge_strategy == "intersection":
            # Only keep boxes detected by all detectors
            # (simplified: just return union for now)
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

    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area


# =============================================================================
# Convenience aliases (backward compatibility)
# =============================================================================

# Keep old names for backward compatibility
CLASS_NAMES = DOCLAYOUT_YOLO_CLASS_NAMES
HFF_CLASSES = DOCLAYOUT_YOLO_HFF_CLASSES
