"""Unified YOLO-based HFF detector.

A single configurable detector that supports multiple YOLO model
backends and class mappings.  Preset configurations are provided for
common models:

- **TDLA** – Tibetan Document Layout Analysis (YOLO26-m via doclayout-yolo)
- **DocLayout-YOLO** – DocStructBench (YOLOv10 via doclayout-yolo)
- **Eric-YOLO** – tiled YOLO11-nano (via ultralytics)

Usage::

    from hff_remover.detector import YoloDetector, TDLA_CONFIG
    detector = YoloDetector(model_path="model.pt", config=TDLA_CONFIG)
    detections = detector.detect("page.jpg")
"""

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from hff_remover.detector._base import BaseHFFDetector

logger = logging.getLogger(__name__)


# =====================================================================
# Utility functions
# =====================================================================

def draw_bboxes(
    img: np.ndarray,
    boxes: List[List[float]],
    classes: List[int],
    color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """Draw bounding boxes on an image for debugging/visualisation.

    Args:
        img: BGR image array (will be modified in place).
        boxes: List of ``[x1, y1, x2, y2]`` bounding boxes.
        classes: List of integer class IDs (same length as *boxes*).
        color_map: Optional ``{class_id: (B, G, R)}`` colour map.

    Returns:
        The image with rectangles drawn on it.
    """
    _default: Dict[int, Tuple[int, int, int]] = {
        0: (255, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
        3: (255, 0, 255),
        4: (100, 0, 255),
        5: (100, 100, 255),
    }
    cmap = color_map or _default
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        color = cmap.get(int(cls), (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img


# =====================================================================
# Model configuration
# =====================================================================

@dataclass(frozen=True)
class YoloModelConfig:
    """Configuration describing a YOLO model variant.

    Attributes:
        class_names: Mapping of model class IDs to human-readable names.
        label_map: Mapping of raw class names to normalised HFF labels.
            Keys present here are kept during HFF-filtered detection;
            everything else is discarded.  Use ``"abandon"`` as a
            sentinel for classes needing positional resolution.
        backend: YOLO library – ``"ultralytics"`` for
            ``ultralytics.YOLO`` or ``"doclayout_yolo"`` for
            ``doclayout_yolo.YOLOv10``.
        hf_repo_id: HuggingFace Hub repository for auto-download.
        hf_model_filename: Filename within the HF repository.
        default_imgsz: Default image size passed to inference.
        tiled: Whether to use tiled inference (scale + tile + merge).
        positional_abandon: Whether to resolve ``"abandon"`` labels
            into ``"header"`` / ``"footer"`` by page position.
        header_region_ratio: Vertical fraction (0–1) – abandon boxes
            with y-centre above this become ``"header"``.
        footer_region_ratio: Vertical fraction (0–1) – abandon boxes
            with y-centre below this become ``"footer"``.
        debug_color_map: BGR colour map for visualisation.
    """

    class_names: Dict[int, str] = field(default_factory=dict)
    label_map: Dict[str, Optional[str]] = field(default_factory=dict)
    backend: str = "ultralytics"
    hf_repo_id: Optional[str] = None
    hf_model_filename: Optional[str] = None
    default_imgsz: int = 640
    tiled: bool = False
    positional_abandon: bool = False
    header_region_ratio: float = 0.33
    footer_region_ratio: float = 0.67
    debug_color_map: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)


# ── Preset configurations ────────────────────────────────────────────

TDLA_CONFIG = YoloModelConfig(
    class_names={0: "header", 1: "Text area", 2: "footnote", 3: "footer"},
    label_map={
        "header": "header",
        "Text area": "text-area",
        "footnote": "footnote",
        "footer": "footer",
    },
    backend="doclayout_yolo",
    hf_repo_id="BDRC/Tibetan_Modern_Book_Layout_Detection_Model",
    hf_model_filename="Tibetan_modern_book_Layout_detection.pt",
    default_imgsz=640,
    debug_color_map={
        0: (255, 0, 0),
        1: (100, 0, 255),
        2: (0, 0, 255),
        3: (0, 255, 0),
    },
)

DOCLAYOUT_YOLO_CONFIG = YoloModelConfig(
    class_names={
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
    },
    label_map={
        "plain_text": "text-area",
        "table_footnote": "footnote",
        "abandon": "abandon",
    },
    backend="doclayout_yolo",
    hf_repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
    hf_model_filename="doclayout_yolo_docstructbench_imgsz1024.pt",
    default_imgsz=1024,
    positional_abandon=True,
    debug_color_map={
        0: (255, 200, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
        3: (255, 0, 255),
        4: (255, 128, 0),
        5: (0, 255, 255),
        6: (128, 0, 255),
        7: (0, 128, 255),
        8: (255, 0, 128),
        9: (128, 255, 0),
    },
)

ERIC_YOLO_CONFIG = YoloModelConfig(
    class_names={
        0: "header",
        1: "footer",
        2: "footnote",
        3: "page_number",
        4: "text_area",
        5: "image",
    },
    label_map={
        "header": "header",
        "footer": "footer",
        "footnote": "footnote",
        "page_number": "footer",
        "text_area": "text-area",
        "image": "text-area",
    },
    backend="ultralytics",
    default_imgsz=640,
    tiled=True,
    debug_color_map={
        0: (255, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
        3: (255, 0, 255),
        4: (100, 0, 255),
        5: (100, 100, 255),
    },
)


# =====================================================================
# Unified YOLO detector
# =====================================================================

class YoloDetector(BaseHFFDetector):
    """Unified YOLO-based detector for headers, footers, and footnotes.

    Supports multiple YOLO backends, class mappings, and inference
    strategies (full-image or tiled) through :class:`YoloModelConfig`.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: YoloModelConfig = TDLA_CONFIG,
        device: str = "cuda",
        confidence_threshold: float = 0.25,
        tile_size: int = 640,
    ) -> None:
        """Initialise the YOLO detector.

        Args:
            model_path: Path to YOLO ``.pt`` weights.  When ``None``
                and the config provides ``hf_repo_id``, the model is
                downloaded from HuggingFace Hub automatically.
            config: Model configuration preset.
            device: Inference device (``"cuda"`` or ``"cpu"``).
            confidence_threshold: Minimum detection confidence.
            tile_size: Tile dimension for tiled inference.
        """
        self.config = config
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.tile_size = tile_size

        resolved_path = self._resolve_model_path(model_path)
        self.model = self._load_model(resolved_path)
        self.model_path = resolved_path

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _resolve_model_path(self, model_path: Optional[str]) -> str:
        """Return a valid local path to the model weights.

        If *model_path* points to an existing file it is returned as-is.
        Otherwise the weights are downloaded from HuggingFace Hub using
        the repository configured in ``self.config``.

        Args:
            model_path: User-supplied path, or ``None`` for auto-download.

        Returns:
            Absolute path to the model file.

        Raises:
            ValueError: If *model_path* is ``None`` and the config has
                no HuggingFace repository configured.
        """
        if model_path is not None and Path(model_path).is_file():
            return model_path

        if self.config.hf_repo_id is None or self.config.hf_model_filename is None:
            if model_path is not None:
                return model_path
            raise ValueError(
                "model_path does not point to an existing file and "
                "the config does not specify a HuggingFace repository "
                "for auto-download."
            )

        from huggingface_hub import hf_hub_download

        logger.info(
            "Downloading model from HuggingFace (%s / %s)...",
            self.config.hf_repo_id,
            self.config.hf_model_filename,
        )
        downloaded = hf_hub_download(
            repo_id=self.config.hf_repo_id,
            filename=self.config.hf_model_filename,
        )
        logger.info("Model downloaded to: %s", downloaded)
        return downloaded

    def _load_model(self, model_path: str) -> Any:
        """Load the YOLO model using the configured backend.

        Args:
            model_path: Local path to the ``.pt`` weights.

        Returns:
            A loaded YOLO model instance.

        Raises:
            ValueError: If the configured backend is unknown.
        """
        if self.config.backend == "doclayout_yolo":
            from doclayout_yolo import YOLOv10 as YOLO
            return YOLO(model_path)
        if self.config.backend == "ultralytics":
            from ultralytics import YOLO
            return YOLO(model_path, task="detect")
        raise ValueError(f"Unknown YOLO backend: {self.config.backend!r}")

    # ------------------------------------------------------------------
    # Label normalisation
    # ------------------------------------------------------------------

    def _normalize_detection_label(
        self, raw_label: Union[int, str],
    ) -> Optional[str]:
        """Map a model class ID to a canonical HFF label.

        Args:
            raw_label: Integer class ID produced by the model.

        Returns:
            Normalised label string, or ``None`` if the class should
            be discarded.
        """
        class_name = self.config.class_names.get(int(raw_label))
        if class_name is None:
            return None
        return self.config.label_map.get(class_name)

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Load an image as a BGR numpy array.

        Args:
            image: File path or pre-loaded array.

        Returns:
            BGR ``uint8`` numpy array.

        Raises:
            FileNotFoundError: If the path does not point to a readable
                image.
        """
        if isinstance(image, np.ndarray):
            return image
        path = str(image)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image at {path}")
        return img

    # ------------------------------------------------------------------
    # Positional abandon resolution
    # ------------------------------------------------------------------

    def _classify_abandon_by_position(
        self,
        bbox_xyxy: List[float],
        image_height: int,
    ) -> Optional[str]:
        """Resolve an ``"abandon"`` detection to header/footer by position.

        Args:
            bbox_xyxy: ``[x1, y1, x2, y2]`` bounding box.
            image_height: Full image height in pixels.

        Returns:
            ``"header"`` or ``"footer"`` if in the expected region,
            ``None`` otherwise (box is discarded).
        """
        if image_height <= 0:
            return None
        y_center = (float(bbox_xyxy[1]) + float(bbox_xyxy[3])) / 2.0
        y_norm = y_center / float(image_height)
        if y_norm <= self.config.header_region_ratio:
            return "header"
        if y_norm >= self.config.footer_region_ratio:
            return "footer"
        return None

    # ------------------------------------------------------------------
    # Full-image inference
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_single_result(
        result: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract numpy arrays from a YOLO ``Results`` object.

        Args:
            result: One element from the list returned by ``model()``.

        Returns:
            Tuple of ``(boxes, scores, classes)`` arrays.
        """
        if result.boxes is None or len(result.boxes) == 0:
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        return boxes, scores, classes

    def _run_inference(
        self,
        image: Union[str, Path, np.ndarray],
        imgsz: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run YOLO inference on the full image.

        Args:
            image: File path, ``Path``, or pre-loaded BGR array.
            imgsz: Image size passed to the model.

        Returns:
            Tuple of ``(boxes, scores, classes)`` arrays.
        """
        results = self.model(
            image,
            imgsz=imgsz,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )
        return self._parse_single_result(results[0])

    # ------------------------------------------------------------------
    # Tiled inference
    # ------------------------------------------------------------------

    def _scale_image(self, img: np.ndarray) -> np.ndarray:
        """Scale *img* so its width equals ``2 * tile_size``.

        Args:
            img: BGR image array.

        Returns:
            Resized image (or the original if no scaling is needed).
        """
        h, w = img.shape[:2]
        target_width = 2 * self.tile_size
        scale_factor = target_width / w
        if abs(scale_factor - 1.0) < 1e-9:
            return img
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        return cv2.resize(img, (new_w, new_h))

    @staticmethod
    def _generate_tiles(
        img_w: int,
        img_h: int,
        tile_size: int,
    ) -> List[Dict[str, int]]:
        """Build a grid of non-overlapping tile coordinates.

        Args:
            img_w: Image width in pixels.
            img_h: Image height in pixels.
            tile_size: Tile side length.

        Returns:
            List of dicts with keys ``x0``, ``y0``, ``x1``, ``y1``.
        """
        tiles: List[Dict[str, int]] = []
        for y0 in range(0, img_h, tile_size):
            for x0 in range(0, img_w, tile_size):
                tiles.append({
                    "x0": x0,
                    "y0": y0,
                    "x1": min(x0 + tile_size, img_w),
                    "y1": min(y0 + tile_size, img_h),
                })
        return tiles

    def _run_tiled_inference(
        self,
        img: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run YOLO inference over a grid of tiles.

        Args:
            img: Pre-scaled BGR image.

        Returns:
            Tuple of ``(boxes, scores, classes)`` in global coordinates.
        """
        img_h, img_w = img.shape[:2]
        tiles = self._generate_tiles(img_w, img_h, self.tile_size)

        global_boxes: List[List[float]] = []
        global_scores: List[float] = []
        global_classes: List[int] = []

        for tile in tiles:
            tile_img = img[tile["y0"]:tile["y1"], tile["x0"]:tile["x1"]]
            h, w = tile_img.shape[:2]

            padded = cv2.copyMakeBorder(
                tile_img,
                0, self.tile_size - h,
                0, self.tile_size - w,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

            results = self.model(
                padded,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
            )[0]

            if results.boxes is None or len(results.boxes) == 0:
                continue

            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()

            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box
                x1 = min(float(x1), w)
                x2 = min(float(x2), w)
                y1 = min(float(y1), h)
                y2 = min(float(y2), h)

                global_boxes.append([
                    x1 + tile["x0"],
                    y1 + tile["y0"],
                    x2 + tile["x0"],
                    y2 + tile["y0"],
                ])
                global_scores.append(float(score))
                global_classes.append(int(cls))

        if not global_boxes:
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

        return (
            np.array(global_boxes),
            np.array(global_scores),
            np.array(global_classes, dtype=int),
        )

    # ------------------------------------------------------------------
    # Detection dict builders
    # ------------------------------------------------------------------

    def _boxes_to_hff_detections(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
        image_height: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Convert raw arrays to HFF-filtered detection dicts.

        Args:
            boxes: ``(N, 4)`` array of xyxy coordinates.
            scores: ``(N,)`` confidence scores.
            classes: ``(N,)`` integer class IDs.
            image_height: Image height for positional abandon resolution.

        Returns:
            Detection dicts with only HFF-relevant classes.
        """
        detections: List[Dict[str, Any]] = []
        for bbox, score, cls_id in zip(boxes, scores, classes):
            label = self._normalize_detection_label(int(cls_id))
            if label is None:
                continue

            bbox_list = bbox.tolist() if hasattr(bbox, "tolist") else list(bbox)

            if label == "abandon" and self.config.positional_abandon:
                if image_height is None:
                    continue
                resolved = self._classify_abandon_by_position(
                    bbox_list, image_height,
                )
                if resolved is None:
                    continue
                label = resolved

            detections.append({
                "bbox": bbox_list,
                "class_id": int(cls_id),
                "class_name": label,
                "confidence": float(score),
            })
        return detections

    def _boxes_to_all_detections(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Convert raw arrays to detection dicts for *all* classes.

        Args:
            boxes: ``(N, 4)`` array of xyxy coordinates.
            scores: ``(N,)`` confidence scores.
            classes: ``(N,)`` integer class IDs.

        Returns:
            Detection dicts for every class the model recognises.
        """
        detections: List[Dict[str, Any]] = []
        for bbox, score, cls_id in zip(boxes, scores, classes):
            class_name = self.config.class_names.get(
                int(cls_id), f"unknown_{int(cls_id)}",
            )
            detections.append({
                "bbox": bbox.tolist() if hasattr(bbox, "tolist") else list(bbox),
                "class_id": int(cls_id),
                "class_name": class_name,
                "confidence": float(score),
            })
        return detections

    # ------------------------------------------------------------------
    # Internal inference dispatch
    # ------------------------------------------------------------------

    def _detect_full(
        self,
        image: Union[str, Path, np.ndarray],
        imgsz: int,
        all_classes: bool = False,
    ) -> List[Dict[str, Any]]:
        """Full-image detection pipeline.

        Args:
            image: Image path or array.
            imgsz: Inference image size.
            all_classes: Return all classes, not just HFF.

        Returns:
            List of detection dicts.
        """
        img = self._load_image(image)
        boxes, scores, classes = self._run_inference(img, imgsz=imgsz)

        if all_classes:
            return self._boxes_to_all_detections(boxes, scores, classes)

        image_height = int(img.shape[0]) if self.config.positional_abandon else None
        return self._boxes_to_hff_detections(
            boxes, scores, classes, image_height=image_height,
        )

    def _detect_tiled(
        self,
        image: Union[str, Path, np.ndarray],
        imgsz: int,
        all_classes: bool = False,
    ) -> List[Dict[str, Any]]:
        """Tiled detection pipeline.

        Args:
            image: Image path or array.
            imgsz: Tile size (overrides ``self.tile_size`` for this call).
            all_classes: Return all classes, not just HFF.

        Returns:
            List of detection dicts.
        """
        self.tile_size = imgsz
        img = self._load_image(image)
        img = self._scale_image(img)
        boxes, scores, classes = self._run_tiled_inference(img)

        if all_classes:
            return self._boxes_to_all_detections(boxes, scores, classes)

        image_height = int(img.shape[0]) if self.config.positional_abandon else None
        return self._boxes_to_hff_detections(
            boxes, scores, classes, image_height=image_height,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        image_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Detect headers, footers, and footnotes in an image.

        Args:
            image: Path to an image file or a BGR numpy array.
            image_size: Image/tile size for inference.  Defaults to the
                config's ``default_imgsz``.
            **kwargs: Accepted for API compatibility; ignored.

        Returns:
            List of detection dicts with keys ``bbox``, ``class_id``,
            ``class_name``, and ``confidence``.
        """
        imgsz = image_size or self.config.default_imgsz
        if self.config.tiled:
            return self._detect_tiled(image, imgsz)
        return self._detect_full(image, imgsz)

    def detect_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        image_size: Optional[int] = None,
        batch_size: int = 8,
        **kwargs: Any,
    ) -> List[List[Dict[str, Any]]]:
        """Detect headers, footers, and footnotes in multiple images.

        For non-tiled models images are batched through the YOLO model
        in chunks of *batch_size*.  Tiled models process one image at a
        time (each image is internally tiled).

        Args:
            images: List of image paths or BGR numpy arrays.
            image_size: Image/tile size for inference.
            batch_size: GPU batch size (non-tiled models only).
            **kwargs: Accepted for API compatibility; ignored.

        Returns:
            List of detection lists, one per input image.
        """
        imgsz = image_size or self.config.default_imgsz

        if self.config.tiled:
            return [
                self._detect_tiled(img, imgsz)
                for img in images
            ]

        all_detections: List[List[Dict[str, Any]]] = []
        for i in range(0, len(images), batch_size):
            chunk = [self._load_image(img) for img in images[i : i + batch_size]]

            results = self.model(
                chunk,
                imgsz=imgsz,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
            )

            for idx, result in enumerate(results):
                boxes, scores, classes = self._parse_single_result(result)
                image_height = (
                    int(chunk[idx].shape[0])
                    if self.config.positional_abandon
                    else None
                )
                all_detections.append(
                    self._boxes_to_hff_detections(
                        boxes, scores, classes, image_height=image_height,
                    )
                )

            del chunk, results

        return all_detections

    def get_all_detections(
        self,
        image: Union[str, Path, np.ndarray],
        image_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return *all* detections (not just HFF).

        Useful for debugging or visualisation.

        Args:
            image: Path to an image file or a BGR numpy array.
            image_size: Image/tile size for inference.

        Returns:
            Detection dicts for all classes the model recognises.
        """
        imgsz = image_size or self.config.default_imgsz
        if self.config.tiled:
            return self._detect_tiled(image, imgsz, all_classes=True)
        return self._detect_full(image, imgsz, all_classes=True)

    def prepare_image(
        self,
        image: Union[str, Path, np.ndarray],
    ) -> np.ndarray:
        """Load and scale an image for tiled inference.

        Args:
            image: File path or pre-loaded BGR array.

        Returns:
            Scaled BGR image ready for tiling.
        """
        img = self._load_image(image)
        return self._scale_image(img)


# =====================================================================
# Backward-compatible wrapper classes
# =====================================================================

class HFFDetector(YoloDetector):
    """DocLayout-YOLO based HFF detector (backward-compatible wrapper).

    Equivalent to ``YoloDetector(config=DOCLAYOUT_YOLO_CONFIG, ...)``.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        header_region_ratio: float = 0.33,
        footer_region_ratio: float = 0.67,
    ) -> None:
        """Initialise the DocLayout-YOLO detector.

        Args:
            model_path: Path to model weights.  When ``None`` the
                weights are downloaded from HuggingFace Hub.
            device: Inference device (``"cuda"`` or ``"cpu"``).
            confidence_threshold: Minimum confidence score.
            header_region_ratio: Page fraction for header region.
            footer_region_ratio: Page fraction for footer region.
        """
        config = replace(
            DOCLAYOUT_YOLO_CONFIG,
            header_region_ratio=header_region_ratio,
            footer_region_ratio=footer_region_ratio,
        )
        super().__init__(
            model_path=model_path,
            config=config,
            device=device,
            confidence_threshold=confidence_threshold,
        )


class EricYoloDetector(YoloDetector):
    """Tiled YOLO11-nano HFF detector (backward-compatible wrapper).

    Equivalent to ``YoloDetector(config=ERIC_YOLO_CONFIG, ...)``.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        confidence_threshold: float = 0.25,
        tile_size: int = 640,
    ) -> None:
        """Initialise the Eric YOLO tiled detector.

        Args:
            model_path: Path to the YOLO11-nano ``.pt`` weights.
            device: Inference device (``"cuda"`` or ``"cpu"``).
            confidence_threshold: Minimum confidence score.
            tile_size: Tile dimension in pixels.
        """
        super().__init__(
            model_path=model_path,
            config=ERIC_YOLO_CONFIG,
            device=device,
            confidence_threshold=confidence_threshold,
            tile_size=tile_size,
        )


TDLADetector = YoloDetector
TMBDLADetector = YoloDetector


# =====================================================================
# Backward-compatible constants
# =====================================================================

DOCLAYOUT_YOLO_CLASS_NAMES: Dict[int, str] = dict(DOCLAYOUT_YOLO_CONFIG.class_names)
DOCLAYOUT_YOLO_HFF_CLASSES: Dict[int, str] = {
    1: "plain_text",
    2: "abandon",
    7: "table_footnote",
}

ERIC_YOLO_CLASS_NAMES: Dict[int, str] = dict(ERIC_YOLO_CONFIG.class_names)
ERIC_YOLO_HFF_CLASSES: Dict[int, str] = {
    0: "header",
    1: "footer",
    2: "footnote",
    4: "text_area",
}
ERIC_YOLO_MODEL_TO_COCO: Dict[int, int] = {
    0: 0,
    1: 3,
    2: 2,
    3: 3,
    4: 1,
    5: 1,
}
ERIC_YOLO_DEBUG_COLOR_MAP: Dict[int, Tuple[int, int, int]] = dict(
    ERIC_YOLO_CONFIG.debug_color_map,
)

TDLA_CLASS_NAMES: Dict[int, str] = dict(TDLA_CONFIG.class_names)
TDLA_HFF_CLASSES: Dict[int, str] = {
    0: "header",
    1: "text-area",
    2: "footnote",
    3: "footer",
}
TDLA_DEBUG_COLOR_MAP: Dict[int, Tuple[int, int, int]] = dict(
    TDLA_CONFIG.debug_color_map,
)
TDLA_MODEL_TO_COCO: Dict[int, int] = {0: 0, 1: 1, 2: 2, 3: 3}

TMB_DLA_CLASS_NAMES: Dict[int, str] = TDLA_CLASS_NAMES
TMB_DLA_HFF_CLASSES: Dict[int, str] = TDLA_HFF_CLASSES
