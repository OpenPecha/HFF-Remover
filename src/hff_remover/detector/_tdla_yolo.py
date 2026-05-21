"""Tibetan Document Layout Analysis (TDLA) YOLO26-m detector.

Uses a custom-trained YOLO26-m model loaded via the ``doclayout_yolo``
library (which supports both YOLO26 and DocLayout-YOLO architectures).
Full-page images are passed directly to the model at the configured
``imgsz``; nearby same-class boxes are merged afterwards.

The model recognises four classes:

========  ============  ===================
Class ID  Raw name      Normalised HFF label
========  ============  ===================
0         header        ``"header"``
1         Text area     ``"text-area"``
2         footnote      ``"footnote"``
3         footer        ``"footer"``
========  ============  ===================
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from hff_remover.detector._base import BaseHFFDetector
from hff_remover.detector._eric_yolo import merge_boxes_by_class

TDLA_CLASS_NAMES: Dict[int, str] = {
    0: "header",
    1: "Text area",
    2: "footnote",
    3: "footer",
}

TDLA_HFF_CLASSES: Dict[int, str] = {
    0: "header",
    1: "text-area",
    2: "footnote",
    3: "footer",
}

_TDLA_LABEL_MAP: Dict[str, Optional[str]] = {
    "header": "header",
    "Text area": "text-area",
    "footnote": "footnote",
    "footer": "footer",
}

TDLA_MODEL_TO_COCO: Dict[int, int] = {
    0: 0,  # header    → header
    1: 1,  # Text area → text-area
    2: 2,  # footnote  → footnote
    3: 3,  # footer    → footer
}

TDLA_DEBUG_COLOR_MAP: Dict[int, Tuple[int, int, int]] = {
    0: (255, 0, 0),      # header    – blue
    1: (100, 0, 255),     # Text area – purple
    2: (0, 0, 255),       # footnote  – red
    3: (0, 255, 0),       # footer    – green
}


def draw_tdla_bboxes(
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
            Defaults to :data:`TDLA_DEBUG_COLOR_MAP`.

    Returns:
        The image with rectangles drawn on it.
    """
    if color_map is None:
        color_map = TDLA_DEBUG_COLOR_MAP
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        color = color_map.get(int(cls), (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img


class TDLADetector(BaseHFFDetector):
    """YOLO26-m-based Tibetan Document Layout Analysis detector.

    The model is loaded via ``doclayout_yolo.YOLOv10`` (which handles
    both YOLO26 and DocLayout-YOLO architectures).  Conv+BN fusing is
    disabled because the YOLO26 ``DilatedBlock`` accesses ``.bn`` at
    runtime.  Full-page images are sent directly to the model at the
    requested ``imgsz``; nearby same-class boxes are merged afterwards.

    Bounding boxes returned by :meth:`detect` are in the **original**
    image coordinate space.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        confidence_threshold: float = 0.25,
        merge_margin: int = 20,
    ) -> None:
        """Initialise the TDLA YOLO26-m detector.

        Args:
            model_path: Path to the YOLO26-m ``.pt`` weights file.
            device: Device for inference (``"cuda"`` or ``"cpu"``).
            confidence_threshold: Minimum confidence score for detections.
            merge_margin: Max gap in pixels for merging nearby boxes of
                the same class.
        """
        from doclayout_yolo import YOLOv10 as YOLO

        self.device = device
        self.confidence_threshold = confidence_threshold
        self.merge_margin = merge_margin

        self.model = YOLO(model_path)
        self.model.model.fuse = lambda verbose=True: self.model.model
        self.model_path = model_path

    # ------------------------------------------------------------------
    # Label normalisation
    # ------------------------------------------------------------------

    def _normalize_detection_label(self, raw_label: Union[int, str]) -> Optional[str]:
        """Map a model class ID to a canonical HFF label.

        Args:
            raw_label: Integer class ID produced by the model.

        Returns:
            Normalised label string, or ``None`` if the class should be
            discarded.
        """
        class_name = TDLA_CLASS_NAMES.get(int(raw_label))
        if class_name is None:
            return None
        return _TDLA_LABEL_MAP.get(class_name)

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Load an image as a BGR numpy array.

        Args:
            image: File path or pre-loaded array.

        Returns:
            BGR ``uint8`` numpy array.

        Raises:
            FileNotFoundError: If the path does not point to a readable image.
        """
        if isinstance(image, np.ndarray):
            return image

        path = str(image)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image at {path}")
        return img

    # ------------------------------------------------------------------
    # Full-image inference
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_single_result(
        result: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract numpy arrays from a single YOLO ``Results`` object.

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
            imgsz: Image size passed to the model for inference.

        Returns:
            Tuple of ``(boxes, scores, classes)`` arrays in the
            original image coordinate space.
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
    # Helpers to build detection dicts
    # ------------------------------------------------------------------

    def _boxes_to_hff_detections(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Convert raw arrays to HFF-filtered detection dicts.

        Args:
            boxes: ``(N, 4)`` array of xyxy coordinates.
            scores: ``(N,)`` confidence scores.
            classes: ``(N,)`` integer class IDs.

        Returns:
            Detection dicts with only HFF-relevant classes.
        """
        detections: List[Dict[str, Any]] = []
        for bbox, score, cls_id in zip(boxes, scores, classes):
            label = self._normalize_detection_label(int(cls_id))
            if label is None:
                continue
            detections.append({
                "bbox": bbox.tolist(),
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
            class_name = TDLA_CLASS_NAMES.get(
                int(cls_id), f"unknown_{int(cls_id)}"
            )
            detections.append({
                "bbox": bbox.tolist(),
                "class_id": int(cls_id),
                "class_name": class_name,
                "confidence": float(score),
            })
        return detections

    # ------------------------------------------------------------------
    # Public API (BaseHFFDetector)
    # ------------------------------------------------------------------

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        image_size: int = 640,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Detect headers, footers, and footnotes in a full-page image.

        The full image is passed directly to the model at ``imgsz``.
        Nearby boxes of the same class are merged before returning.

        Args:
            image: Path to an image file or a BGR numpy array.
            image_size: Image size for inference (default ``640``).
            **kwargs: Accepted for API compatibility (e.g.
                ``normalize_bbox``); ignored by this detector.

        Returns:
            List of detection dicts with keys ``bbox``, ``class_id``,
            ``class_name``, and ``confidence``.
        """
        img = self._load_image(image)
        boxes, scores, classes = self._run_inference(img, imgsz=image_size)
        if len(boxes) > 0:
            boxes, scores, classes = merge_boxes_by_class(
                boxes, scores, classes, margin=self.merge_margin,
            )
        return self._boxes_to_hff_detections(boxes, scores, classes)

    def detect_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        image_size: int = 640,
        batch_size: int = 8,
        **kwargs: Any,
    ) -> List[List[Dict[str, Any]]]:
        """Detect headers, footers, and footnotes in multiple images.

        Images are passed to the YOLO model in chunks of ``batch_size``
        so the GPU can process them in parallel, rather than running a
        separate inference pass per image.

        Args:
            images: List of image paths or BGR numpy arrays.
            image_size: Image size for inference.
            batch_size: Number of images per GPU inference call.
            **kwargs: Accepted for API compatibility; ignored.

        Returns:
            List of detection lists, one per input image.
        """
        loaded = [self._load_image(img) for img in images]

        all_detections: List[List[Dict[str, Any]]] = []
        for i in range(0, len(loaded), batch_size):
            chunk = loaded[i : i + batch_size]

            results = self.model(
                chunk,
                imgsz=image_size,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
            )

            for result in results:
                boxes, scores, classes = self._parse_single_result(result)
                if len(boxes) > 0:
                    boxes, scores, classes = merge_boxes_by_class(
                        boxes, scores, classes, margin=self.merge_margin,
                    )
                all_detections.append(
                    self._boxes_to_hff_detections(boxes, scores, classes)
                )

        return all_detections

    # ------------------------------------------------------------------
    # Extra utility
    # ------------------------------------------------------------------

    def get_all_detections(
        self,
        image: Union[str, Path, np.ndarray],
        image_size: int = 640,
    ) -> List[Dict[str, Any]]:
        """Return *all* detections (not just HFF).

        Useful for debugging or visualisation.

        Args:
            image: Path to an image file or a BGR numpy array.
            image_size: Image size for inference.

        Returns:
            Detection dicts for all four model classes.
        """
        img = self._load_image(image)
        boxes, scores, classes = self._run_inference(img, imgsz=image_size)
        if len(boxes) > 0:
            boxes, scores, classes = merge_boxes_by_class(
                boxes, scores, classes, margin=self.merge_margin,
            )
        return self._boxes_to_all_detections(boxes, scores, classes)
