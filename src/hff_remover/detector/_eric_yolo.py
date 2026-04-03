"""Eric's tiled YOLO11-nano HFF detector.

Uses a custom-trained YOLO11-nano model (``ultralytics``) that was trained on
640x640 patches.  Full page images are scaled so their width equals
``2 * tile_size``, tiled into ``tile_size x tile_size`` patches with black
padding on the right/bottom edges, and per-tile predictions are mapped back
to global image coordinates.

After tiled inference, nearby boxes of the same class are merged so that
regions split across tile boundaries are returned as single detections.

The model recognises six classes; four are HFF-relevant:

========  ============  ===================
Class ID  Raw name      Normalised HFF label
========  ============  ===================
0         header        ``"header"``
1         footer        ``"footer"``
2         footnote      ``"footnote"``
3         page_number   → mapped to ``"footer"``
4         text_area     ``"text-area"``
5         image         → mapped to ``"text-area"``
========  ============  ===================
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from hff_remover.detector._base import BaseHFFDetector

ERIC_YOLO_CLASS_NAMES: Dict[int, str] = {
    0: "header",
    1: "footer",
    2: "footnote",
    3: "page_number",
    4: "text_area",
    5: "image",
}

ERIC_YOLO_HFF_CLASSES: Dict[int, str] = {
    0: "header",
    1: "footer",
    2: "footnote",
    4: "text_area",
}

_ERIC_YOLO_LABEL_MAP: Dict[str, Optional[str]] = {
    "header": "header",
    "footer": "footer",
    "footnote": "footnote",
    "page_number": "footer",
    "text_area": "text-area",
    "image": "text-area",
}

ERIC_YOLO_MODEL_TO_COCO: Dict[int, int] = {
    0: 0,  # header     → header
    1: 3,  # footer     → footer
    2: 2,  # footnote   → footnote
    3: 3,  # page_number → footer
    4: 1,  # text_area  → text-area
    5: 1,  # image      → text-area
}

ERIC_YOLO_DEBUG_COLOR_MAP: Dict[int, Tuple[int, int, int]] = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 0, 255),
    4: (100, 0, 255),
    5: (100, 100, 255),
}


def _boxes_are_nearby(a: List[float], b: List[float], margin: int) -> bool:
    """Check if two ``[x1, y1, x2, y2]`` boxes overlap or are within *margin* px."""
    return not (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def merge_boxes_by_class(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    margin: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge nearby bounding boxes of the same class into single boxes.

    Iteratively merges ``[x1, y1, x2, y2]`` boxes whose gap is within
    *margin* pixels until no more merges are possible.  Keeps the max
    confidence of each merged group.

    Args:
        boxes: ``(N, 4)`` array of xyxy coordinates.
        scores: ``(N,)`` confidence scores.
        classes: ``(N,)`` integer class IDs.
        margin: Max gap in pixels to still merge two boxes.

    Returns:
        Tuple of ``(merged_boxes, merged_scores, merged_classes)``.
    """
    if len(boxes) == 0:
        return boxes, scores, classes

    unique_classes = {int(c) for c in classes}
    out_boxes: List[List[float]] = []
    out_scores: List[float] = []
    out_classes: List[int] = []

    for cls in unique_classes:
        mask = classes == cls
        cls_boxes = [list(b) for b in boxes[mask]]
        cls_scores = list(scores[mask])

        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(cls_boxes):
                j = i + 1
                while j < len(cls_boxes):
                    if _boxes_are_nearby(cls_boxes[i], cls_boxes[j], margin):
                        cls_boxes[i] = [
                            min(cls_boxes[i][0], cls_boxes[j][0]),
                            min(cls_boxes[i][1], cls_boxes[j][1]),
                            max(cls_boxes[i][2], cls_boxes[j][2]),
                            max(cls_boxes[i][3], cls_boxes[j][3]),
                        ]
                        cls_scores[i] = max(cls_scores[i], cls_scores[j])
                        cls_boxes.pop(j)
                        cls_scores.pop(j)
                        changed = True
                    else:
                        j += 1
                i += 1

        for b, s in zip(cls_boxes, cls_scores):
            out_boxes.append(b)
            out_scores.append(s)
            out_classes.append(cls)

    return np.array(out_boxes), np.array(out_scores), np.array(out_classes)


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
            Defaults to :data:`ERIC_YOLO_DEBUG_COLOR_MAP`.

    Returns:
        The image with rectangles drawn on it.
    """
    if color_map is None:
        color_map = ERIC_YOLO_DEBUG_COLOR_MAP
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        color = color_map.get(int(cls), (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img


class EricYoloDetector(BaseHFFDetector):
    """Tiled YOLO11-nano detector for headers, footers, and footnotes.

    The model was trained on 640x640 crops, so full page images are
    automatically scaled and tiled before inference.  Nearby boxes of the
    same class are merged after tiling to heal tile-boundary splits.

    Bounding boxes returned by :meth:`detect` are in the **scaled** image
    coordinate space (width = ``2 * tile_size``).
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        confidence_threshold: float = 0.25,
        tile_size: int = 640,
        merge_margin: int = 20,
    ) -> None:
        """Initialise the Eric YOLO tiled detector.

        Args:
            model_path: Path to the YOLO11-nano ``.pt`` weights file.
            device: Device for inference (``"cuda"`` or ``"cpu"``).
            confidence_threshold: Minimum confidence score for detections.
            tile_size: Tile dimension in pixels.  The image is scaled so
                its width equals ``2 * tile_size`` before tiling.
            merge_margin: Max gap in pixels for merging nearby boxes of
                the same class after tiled inference.
        """
        from ultralytics import YOLO

        self.device = device
        self.confidence_threshold = confidence_threshold
        self.tile_size = tile_size
        self.merge_margin = merge_margin

        self.model = YOLO(model_path, task="detect")
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
        class_name = ERIC_YOLO_CLASS_NAMES.get(int(raw_label))
        if class_name is None:
            return None
        return _ERIC_YOLO_LABEL_MAP.get(class_name)

    # ------------------------------------------------------------------
    # Image pre-processing helpers
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

    def _scale_image(self, img: np.ndarray) -> np.ndarray:
        """Scale *img* so its width equals ``2 * tile_size``.

        Args:
            img: BGR image array.

        Returns:
            Resized image (or the original if the scale factor is 1.0).
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
        img_w: int, img_h: int, tile_size: int,
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

    def prepare_image(
        self,
        image: Union[str, Path, np.ndarray],
    ) -> np.ndarray:
        """Load and scale an image for tiled inference.

        Convenience wrapper combining :meth:`_load_image` and
        :meth:`_scale_image`.

        Args:
            image: File path or pre-loaded BGR array.

        Returns:
            Scaled BGR image ready for tiling.
        """
        img = self._load_image(image)
        return self._scale_image(img)

    # ------------------------------------------------------------------
    # Tiled inference
    # ------------------------------------------------------------------

    def _run_tiled_inference(
        self,
        img: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run YOLO inference over a grid of tiles.

        Args:
            img: Pre-scaled BGR image.

        Returns:
            Tuple of ``(boxes, scores, classes)`` arrays in global image
            coordinates.  All three are empty ``(0,)``-shaped arrays when
            no detections are found.
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
            class_name = ERIC_YOLO_CLASS_NAMES.get(
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
    ) -> List[Dict[str, Any]]:
        """Detect headers, footers, and footnotes in a full page image.

        The image is scaled so its width equals ``2 * image_size``, then
        tiled into ``image_size x image_size`` patches for inference.
        Nearby boxes of the same class are merged before returning.

        Args:
            image: Path to an image file or a BGR numpy array.
            image_size: Tile size in pixels (default ``640``, the
                resolution the model was trained at).

        Returns:
            List of detection dicts with keys ``bbox``, ``class_id``,
            ``class_name``, and ``confidence``.
        """
        self.tile_size = image_size
        img = self.prepare_image(image)
        boxes, scores, classes = self._run_tiled_inference(img)
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
    ) -> List[List[Dict[str, Any]]]:
        """Detect headers, footers, and footnotes in multiple images.

        Each image is processed independently with the tiling pipeline.

        Args:
            images: List of image paths or BGR numpy arrays.
            image_size: Tile size in pixels.
            batch_size: Unused (kept for API compatibility); each image
                is tiled internally.

        Returns:
            List of detection lists, one per input image.
        """
        return [self.detect(img, image_size=image_size) for img in images]

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
            image_size: Tile size in pixels.

        Returns:
            Detection dicts for all six model classes.
        """
        self.tile_size = image_size
        img = self.prepare_image(image)
        boxes, scores, classes = self._run_tiled_inference(img)
        if len(boxes) > 0:
            boxes, scores, classes = merge_boxes_by_class(
                boxes, scores, classes, margin=self.merge_margin,
            )
        return self._boxes_to_all_detections(boxes, scores, classes)
