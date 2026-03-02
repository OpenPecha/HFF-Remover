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

# Classes we want to detect (Header, Footer, Footnote, Text)
# In DocLayout-YOLO, "abandon" (class 2) represents headers/footers/page numbers
# "table_footnote" (class 7) represents footnotes in tables
# "plain_text" (class 1) represents text areas / body text
DOCLAYOUT_YOLO_HFF_CLASSES = {
    1: "plain_text",     # Text area / body text
    2: "abandon",        # Headers, footers, page numbers
    7: "table_footnote", # Table footnotes
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
        # Keep plain_text / text area (class 1)
        if class_id == 1:
            return {
                "bbox": bbox,
                "class_id": class_id,
                "class_name": "text",
                "confidence": confidence,
            }

        # Keep table_footnote (class 7)
        if class_id == 7:
            return {
                "bbox": bbox,
                "class_id": class_id,
                "class_name": "table_footnote",
                "confidence": confidence,
            }

        # Handle abandon (class 2) by position: header/footer/middle-ignore
        if class_id == 2:
            if image_height is None:
                class_name: Optional[str] = "abandon"
            else:
                class_name = self._classify_abandon_by_position(bbox, image_height)
            if class_name is None:
                return None
            return {
                "bbox": bbox,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
            }

        # Ignore all other classes
        return None

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

            # Use the length of `boxes.cls` for robustness and better testability.
            for i in range(len(boxes.cls)):
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


# =============================================================================
# PP-DocLayout-L Detector (PaddlePaddle)
# =============================================================================

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
            return "text"

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


# =============================================================================
# Surya Layout Detector
# =============================================================================

# Surya layout labels we output (text, header, footer, footnote).
# We filter using normalized labels from _normalize_label.
SURYA_LAYOUT_HFF_LABELS = {
    "text",
    "header",
    "footer",
    "footnote",
}

# YOLO format class IDs (manager-specified): 0=text, 1=footer, 2=header, 3=footnote
SURYA_HFF_CLASS_IDS = {
    "text": 0,
    "footer": 1,
    "header": 2,
    "footnote": 3,
}
SURYA_HFF_CLASS_NAMES = {v: k for k, v in SURYA_HFF_CLASS_IDS.items()}  # id -> name for data.yaml


def save_yolo_data_yaml(
    output_path: Union[str, Path],
    class_names: Optional[Dict[int, str]] = None,
    images_dir: Optional[Union[str, Path]] = None,
) -> None:
    """
    Write a YOLO-style data.yaml (class names and optional path).
    Use with the .txt label files from save_detections_yolo_format.

    Args:
        output_path: Where to write data.yaml (e.g. test_images/data.yaml).
        class_names: Map class_id (int) -> name (str). Defaults to SURYA_HFF_CLASS_NAMES.
        images_dir: Optional path to images directory for the 'path' key.
    """
    if class_names is None:
        class_names = SURYA_HFF_CLASS_NAMES
    nc = len(class_names)
    path_line = f"path: {Path(images_dir).resolve()}" if images_dir else "path: ."
    names_lines = "\n".join(f"  {i}: {class_names[i]}" for i in range(nc))
    yaml_content = f"""# YOLO dataset config (HFF classes from SuryaLayoutDetector)
# class_id: 0=text, 1=footer, 2=header, 3=footnote

{path_line}
train: .
val: .

nc: {nc}
names:
{names_lines}
"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(yaml_content.strip() + "\n", encoding="utf-8")


def merge_detections_by_class(
    detections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Combine all detections of the same class into one box per class (union bbox).

    Args:
        detections: List of dicts with "bbox" [x1,y1,x2,y2] and "class_name".

    Returns:
        One detection per class_name with bbox = union of all bboxes for that class.
    """
    by_class: Dict[str, List[Dict[str, Any]]] = {}
    for det in detections:
        bbox = det.get("bbox")
        class_name = det.get("class_name", "")
        if not bbox or len(bbox) != 4:
            continue
        if class_name not in by_class:
            by_class[class_name] = []
        by_class[class_name].append(det)

    merged = []
    for class_name, group in by_class.items():
        x1 = min(d["bbox"][0] for d in group)
        y1 = min(d["bbox"][1] for d in group)
        x2 = max(d["bbox"][2] for d in group)
        y2 = max(d["bbox"][3] for d in group)
        best = max(group, key=lambda d: d.get("confidence", 0.0))
        merged.append({
            "bbox": [x1, y1, x2, y2],
            "class_name": class_name,
            "class_id": best.get("class_id"),
            "confidence": best.get("confidence"),
        })
    return merged


def save_detections_yolo_format(
    detections: List[Dict[str, Any]],
    image_path: Union[str, Path],
    output_path: Union[str, Path],
    class_name_to_id: Optional[Dict[str, int]] = None,
    merge_same_class: bool = True,
) -> None:
    """
    Save detections to a file in YOLO format: one line per box,
    "class_id cx cy w h" (space-separated, normalized 0-1).

    When merge_same_class is True, all boxes of the same class are combined
    into one union bounding box per class before saving.

    Args:
        detections: List of dicts with "bbox" [x1,y1,x2,y2] and "class_name".
        image_path: Path to the image (used to read width/height for normalization).
        output_path: Path to the .txt file to write.
        class_name_to_id: Map class_name -> integer. Defaults to SURYA_HFF_CLASS_IDS.
        merge_same_class: If True, merge all same-class boxes into one per class.
    """
    from PIL import Image

    if class_name_to_id is None:
        class_name_to_id = SURYA_HFF_CLASS_IDS

    if merge_same_class:
        detections = merge_detections_by_class(detections)

    with Image.open(str(image_path)) as im:
        img_w, img_h = im.size

    lines = []
    for det in detections:
        bbox = det.get("bbox")
        class_name = det.get("class_name", "")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(x) for x in bbox]
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        cid = class_name_to_id.get(class_name, 0)
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


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
                top_k = getattr(box, "top_k", None) or {}
                bbox = getattr(box, "bbox", None)
                polygon = getattr(box, "polygon", None)
                box_confidence = getattr(box, "confidence", None)
            elif isinstance(box, dict):
                label = box.get("label")
                top_k = box.get("top_k") or {}
                bbox = box.get("bbox")
                polygon = box.get("polygon")
                box_confidence = box.get("confidence")
            else:
                continue

            # Prefer LayoutBox.confidence; else use top_k[label]
            if box_confidence is not None:
                score = float(box_confidence)
            else:
                score = float(top_k.get(label, 1.0)) if label is not None else 1.0

            # Filter by confidence
            if score < self.confidence_threshold:
                continue

            # Normalize first so we can filter by canonical names
            normalized_label = self._normalize_label(str(label))

            # Optionally keep only header/footer/footnote labels
            if filter_hff_only and normalized_label not in SURYA_LAYOUT_HFF_LABELS:
                continue

            # Derive bbox from polygon if bbox missing or wrong length (Surya LayoutBox often has polygon)
            if not bbox or len(bbox) != 4:
                if polygon and len(polygon) >= 3:
                    flat = []
                    for p in polygon:
                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                            flat.extend([float(p[0]), float(p[1])])
                        else:
                            flat = []
                            break
                    if flat:
                        xs, ys = flat[0::2], flat[1::2]
                        bbox = [min(xs), min(ys), max(xs), max(ys)]
                else:
                    continue

            bbox = [float(x) for x in bbox]

            # YOLO-style: numeric class_id (0=header, 1=footer, 2=footnote, 3=page_number) for HFF
            class_id = (
                SURYA_HFF_CLASS_IDS[normalized_label]
                if normalized_label in SURYA_HFF_CLASS_IDS
                else label
            )

            detections.append(
                {
                    "bbox": bbox,
                    "class_id": class_id,
                    "class_name": normalized_label,
                    "confidence": score,
                }
            )

        return detections

    def _normalize_label(self, label: str) -> str:
        """Normalize Surya layout labels to our standard format (same as in HFF-Remover detector)."""
        label = label.lower().replace("-", "_").replace(" ", "_")

        if label in ("header", "page_header", "pageheader"):
            return "header"
        elif label in ("footer", "page_footer", "pagefooter"):
            return "footer"
        elif label in ("footnote", "footnotes"):
            return "footnote"
        elif label in ("page_number", "pagenumber"):
            return "page_number"
        elif label in ("text", "plain_text", "paragraph"):
            return "text"

        return label

    def save_results_yolo(
        self,
        detections: List[Dict[str, Any]],
        image_path: Union[str, Path],
        output_txt_path: Union[str, Path],
        data_yaml_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Save detections in YOLO format (.txt) and create data.yaml (same as detector module helpers).
        Uses save_detections_yolo_format and save_yolo_data_yaml from this module.
        """
        save_detections_yolo_format(detections, image_path, output_txt_path)
        if data_yaml_path is None:
            data_yaml_path = Path(output_txt_path).parent / "data.yaml"
        save_yolo_data_yaml(data_yaml_path, images_dir=Path(output_txt_path).parent)

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
