"""Document layout detectors for headers, footers, and footnotes."""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import numpy as np

# Optional imports for Surya layout detector (set to None if unavailable)
try:
    from surya.foundation import FoundationPredictor  # type: ignore
    from surya.layout import LayoutPredictor  # type: ignore
    from surya.settings import settings as surya_settings  # type: ignore
except Exception:  # ImportError and any runtime issues
    FoundationPredictor = None  # type: ignore
    LayoutPredictor = None  # type: ignore
    surya_settings = None  # type: ignore

# Optional alias for DocLayout-YOLO model (for tests/backward compatibility)
try:  # pragma: no cover - exercised indirectly in tests via mocking
    from doclayout_yolo import YOLOv10  # type: ignore
except Exception:  # ImportError and others
    YOLOv10 = None  # type: ignore

# Optional alias for HuggingFace Hub download helper (for tests/backward compatibility)
try:  # pragma: no cover - exercised indirectly in tests via mocking
    from huggingface_hub import hf_hub_download  # type: ignore
except Exception:  # ImportError and others
    hf_hub_download = None  # type: ignore


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
        # Use module-level aliases so tests can mock these symbols.
        global YOLOv10, hf_hub_download

        # Lazily import if not already available (for robustness).
        if YOLOv10 is None:
            from doclayout_yolo import YOLOv10 as _YOLOv10  # type: ignore
            YOLOv10 = _YOLOv10  # type: ignore
        if hf_hub_download is None:
            from huggingface_hub import hf_hub_download as _hf_hub_download  # type: ignore
            hf_hub_download = _hf_hub_download  # type: ignore

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

            # Use the length of `boxes.cls` for robustness and better testability.
            for i in range(len(boxes.cls)):
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
                    # See comment in `detect` about using len(boxes.cls)
                    for j in range(len(boxes.cls)):
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
# Surya Layout Detector
# =============================================================================

# Surya layout labels for header, footer, and footnotes.
# Reference: https://github.com/datalab-to/surya (layout and reading order section)
SURYA_LAYOUT_HFF_LABELS = {
    "Page-header",
    "Page-footer",
    "Footnote",
}


class SuryaLayoutDetector(BaseHFFDetector):
    """Detector for headers, footers, and footnotes using Surya layout model.

    This uses the Surya layout predictor, which is language-agnostic and works
    well for scanned documents in many scripts, including Tibetan.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        layout_checkpoint: Optional[str] = None,
    ):
        """
        Initialize the Surya layout detector.

        Args:
            confidence_threshold: Minimum confidence score for detections.
            layout_checkpoint: Optional custom checkpoint path for the layout model.
                               If None, uses Surya's default LAYOUT_MODEL_CHECKPOINT.
        """
        if LayoutPredictor is None or FoundationPredictor is None:
            raise ImportError(
                "Surya is not available. Please install 'surya-ocr' to use SuryaLayoutDetector."
            )

        self.confidence_threshold = confidence_threshold

        # Use Surya's default layout checkpoint if not provided
        if layout_checkpoint is None:
            if surya_settings is None:
                raise RuntimeError(
                    "Surya settings are not available and no layout_checkpoint was provided."
                )
            layout_checkpoint = surya_settings.LAYOUT_MODEL_CHECKPOINT

        self.layout_checkpoint = layout_checkpoint

        # Initialize Surya layout predictor
        foundation = FoundationPredictor(checkpoint=layout_checkpoint)
        self.model = LayoutPredictor(foundation)

    def _load_image(self, image: Union[str, Path, np.ndarray]):
        """Load input as a PIL image in RGB format."""
        from PIL import Image

        if isinstance(image, (str, Path)):
            return Image.open(str(image)).convert("RGB")

        if isinstance(image, np.ndarray):
            # Handle grayscale or color arrays. Assume BGR channel order for 3-channel images.
            if image.ndim == 2:
                return Image.fromarray(image).convert("RGB")
            if image.ndim == 3 and image.shape[2] in (3, 4):
                # Convert BGR (OpenCV) to RGB
                rgb = image[..., :3][:, :, ::-1]
                return Image.fromarray(rgb)

        raise TypeError(f"Unsupported image type for SuryaLayoutDetector: {type(image)}")

    def _convert_page_predictions(
        self,
        page_prediction: Any,
        filter_hff_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """Convert a single page's Surya layout predictions into HFF detections.

        Handles both dicts and Surya's pydantic LayoutResult objects.
        """
        # Surya >=0.17 returns a LayoutResult (pydantic BaseModel) with attribute access.
        if hasattr(page_prediction, "bboxes"):
            boxes = getattr(page_prediction, "bboxes") or []
        elif isinstance(page_prediction, dict):
            boxes = page_prediction.get("bboxes") or []
        else:
            boxes = []
        detections: List[Dict[str, Any]] = []

        for box in boxes:
            # LayoutBox is a pydantic model; support both attribute and dict-style access
            if hasattr(box, "label"):
                label = box.label
                top_k = box.top_k or {}
                bbox = getattr(box, "bbox", None)
            elif isinstance(box, dict):
                label = box.get("label")
                top_k = box.get("top_k") or {}
                bbox = box.get("bbox")
            else:
                continue

            # Surya stores confidences in the top_k dict keyed by label
            score = float(top_k.get(label, 1.0)) if label is not None else 1.0

            # Filter by confidence
            if score < self.confidence_threshold:
                continue

            # Optionally keep only header/footer/footnote labels
            if filter_hff_only and label not in SURYA_LAYOUT_HFF_LABELS:
                continue

            if not bbox or len(bbox) != 4:
                continue

            bbox = [float(x) for x in bbox]
            normalized_label = self._normalize_label(str(label))

            detections.append(
                {
                    "bbox": bbox,
                    "class_id": label,
                    "class_name": normalized_label,
                    "confidence": score,
                }
            )

        return detections

    def _normalize_label(self, label: str) -> str:
        """Normalize Surya layout labels to standard HFF names."""
        label_lower = label.lower().replace("-", "_").replace(" ", "_")

        if label_lower in ("page_header", "header"):
            return "header"
        if label_lower in ("page_footer", "footer"):
            return "footer"
        if label_lower == "footnote":
            return "footnote"

        return label_lower

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        image_size: int = 1024,
    ) -> List[Dict[str, Any]]:
        """
        Detect headers, footers, and footnotes in an image using Surya layout.

        Args:
            image: Path to image file or numpy array.
            image_size: Unused (kept for API compatibility).

        Returns:
            List of detection dictionaries with keys:
                - bbox: [x1, y1, x2, y2] bounding box coordinates
                - class_id: Original Surya label
                - class_name: Normalized class name (header/footer/footnote)
                - confidence: Confidence score
        """
        pil_image = self._load_image(image)

        # Surya layout predictor expects a list of images
        predictions = self.model([pil_image])
        if not predictions:
            return []

        page_prediction = predictions[0]
        return self._convert_page_predictions(page_prediction, filter_hff_only=True)

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
            image_size: Unused (kept for API compatibility).
            batch_size: Unused (Surya handles batching internally via env vars).

        Returns:
            List of detection lists, one per input image.
        """
        if not images:
            return []

        pil_images = [self._load_image(img) for img in images]
        predictions = self.model(pil_images)

        all_detections: List[List[Dict[str, Any]]] = []
        for page_prediction in predictions:
            page_detections = self._convert_page_predictions(
                page_prediction, filter_hff_only=True
            )
            all_detections.append(page_detections)

        return all_detections

    def get_all_detections(
        self,
        image: Union[str, Path, np.ndarray],
        image_size: int = 1024,
    ) -> List[Dict[str, Any]]:
        """
        Get all layout detections from Surya (not just HFF).

        Args:
            image: Path to image file or numpy array.
            image_size: Unused (kept for API compatibility).

        Returns:
            List of all detection dictionaries, including non-HFF layout elements.
        """
        pil_image = self._load_image(image)
        predictions = self.model([pil_image])
        if not predictions:
            return []

        page_prediction = predictions[0]
        # Do not filter by HFF labels here
        return self._convert_page_predictions(page_prediction, filter_hff_only=False)


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
