"""Image processor for merging detections, masking regions, and writing inference data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

import numpy as np
import cv2

from hff_remover.utils import save_image


class HFFProcessor:
    """Processor for merging nearby detections of the same class."""

    def __init__(self, margin: int = 0):
        """Initialize the HFF processor.

        Args:
            margin: Extra pixels to add around detected regions when merging.
        """
        self.margin = margin

    @staticmethod
    def _boxes_are_nearby(
        a: List[float],
        b: List[float],
        margin: int,
    ) -> bool:
        """Check whether two [x1,y1,x2,y2] boxes overlap or are within *margin* px.

        Args:
            a: First bounding box as [x1, y1, x2, y2].
            b: Second bounding box as [x1, y1, x2, y2].
            margin: Maximum gap (in pixels) to still consider boxes "nearby".

        Returns:
            True if the boxes overlap or the gap between them is ≤ margin.
        """
        return not (
            a[2] + margin < b[0]
            or b[2] + margin < a[0]
            or a[3] + margin < b[1]
            or b[3] + margin < a[1]
        )

    @staticmethod
    def _merge_two_boxes(a: List[float], b: List[float]) -> List[float]:
        """Return the union bounding box of *a* and *b*."""
        return [
            min(a[0], b[0]),
            min(a[1], b[1]),
            max(a[2], b[2]),
            max(a[3], b[3]),
        ]

    @staticmethod
    def _merge_pass(
        boxes: List[List[float]],
        confs: List[float],
        class_ids: List[Any],
        margin: int,
    ) -> bool:
        """Run one merge pass over *boxes*, mutating the lists in-place.

        Returns:
            True if at least one merge happened (caller should re-run).
        """
        changed = False
        i = 0
        while i < len(boxes):
            j = i + 1
            while j < len(boxes):
                if HFFProcessor._boxes_are_nearby(boxes[i], boxes[j], margin):
                    boxes[i] = HFFProcessor._merge_two_boxes(boxes[i], boxes[j])
                    confs[i] = max(confs[i], confs[j])
                    boxes.pop(j)
                    confs.pop(j)
                    class_ids.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1
        return changed

    def merge_nearby_detections(
        self,
        detections: List[Dict[str, Any]],
        margin: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Merge bounding boxes of the same class that are nearby.

        Iteratively merges boxes of the same ``class_name`` whose gap is
        within *margin* pixels until no more merges are possible.  The
        resulting detection keeps the **maximum** confidence of the merged
        group and inherits the ``class_id`` of the first box in the group.

        Args:
            detections: List of detection dicts (``bbox``, ``class_name``,
                ``confidence``, and optionally ``class_id``).
            margin: Maximum gap in pixels between two boxes to still
                merge them.  ``0`` merges only overlapping boxes.
                Defaults to ``self.margin`` when not provided.

        Returns:
            New list of detections with nearby same-class boxes merged.
        """
        effective_margin = self.margin if margin is None else margin

        # Group detections by class_name
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for det in detections:
            class_name = det.get("class_name", "")
            groups.setdefault(class_name, []).append(det)

        merged: List[Dict[str, Any]] = []

        for class_name, class_dets in groups.items():
            boxes: List[List[float]] = [list(d["bbox"]) for d in class_dets]
            confs: List[float] = [d.get("confidence", 1.0) for d in class_dets]
            class_ids: List[Any] = [d.get("class_id") for d in class_dets]

            # Iteratively merge until stable
            while self._merge_pass(boxes, confs, class_ids, effective_margin):
                continue

            for bbox, conf, cid in zip(boxes, confs, class_ids):
                merged.append({
                    "bbox": bbox,
                    "class_name": class_name,
                    "class_id": cid,
                    "confidence": conf,
                })

        return merged


def _xyxy_to_yolo_xywh_norm(
    bbox_xyxy: List[float],
    image_width: int,
    image_height: int,
) -> Tuple[float, float, float, float]:
    """
    Convert [x1,y1,x2,y2] in pixels into normalized YOLO [x_center,y_center,w,h].
    """
    x1, y1, x2, y2 = bbox_xyxy
    # Ensure ordering
    x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
    y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)

    # Clamp to image bounds
    x_min = max(0.0, min(float(image_width), float(x_min)))
    x_max = max(0.0, min(float(image_width), float(x_max)))
    y_min = max(0.0, min(float(image_height), float(y_min)))
    y_max = max(0.0, min(float(image_height), float(y_max)))

    w_px = max(0.0, x_max - x_min)
    h_px = max(0.0, y_max - y_min)
    xc_px = x_min + w_px / 2.0
    yc_px = y_min + h_px / 2.0

    if image_width <= 0 or image_height <= 0:
        return 0.0, 0.0, 0.0, 0.0

    xc = xc_px / float(image_width)
    yc = yc_px / float(image_height)
    w = w_px / float(image_width)
    h = h_px / float(image_height)

    def clamp01(v: float) -> float:
        return max(0.0, min(1.0, v))

    return clamp01(xc), clamp01(yc), clamp01(w), clamp01(h)


# Canonical class mapping – must stay in sync with data.yaml
COCO_CLASS_NAME_TO_ID: Dict[str, int] = {
    "header": 0,
    "text-area": 1,
    "footnote": 2,
    "footer": 3,
}


@dataclass
class COCODatasetWriter:
    """
    Save inference as a COCO-style dataset:
    - inference_data/images/<image>
    - inference_data/labels/<image_stem>.txt
    - inference_data/data.yaml
    """

    base_dir: Union[str, Path] = "inference_data"
    images_subdir: str = "images"
    labels_subdir: str = "labels"

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir)
        self.images_dir = self.base_dir / self.images_subdir
        self.labels_dir = self.base_dir / self.labels_subdir
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.write_data_yaml()

    def _class_id_for(self, class_name: str) -> int:
        """Return the fixed class id for *class_name*.

        Raises:
            ValueError: If the class name is not in the canonical mapping.
        """
        class_name = str(class_name)
        if class_name not in COCO_CLASS_NAME_TO_ID:
            raise ValueError(
                f"Unknown class name '{class_name}'. "
                f"Expected one of {list(COCO_CLASS_NAME_TO_ID.keys())}"
            )
        return COCO_CLASS_NAME_TO_ID[class_name]

    def write_data_yaml(self) -> None:
        """Write/overwrite ``data.yaml`` with dataset path and class names."""
        dataset_path = str(self.base_dir.resolve())

        lines: List[str] = [
            f"path: {dataset_path}",
            f"nc: {len(COCO_CLASS_NAME_TO_ID)}",
            "names:",
        ]
        for class_id, class_name in sorted(
            ((v, k) for k, v in COCO_CLASS_NAME_TO_ID.items())
        ):
            lines.append(f'  {class_id}: "{class_name}"')

        (self.base_dir / "data.yaml").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def write_sample(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        image_rel_path: Union[str, Path],
    ) -> Tuple[Path, Path]:
        """
        Save one image + its COCO label file.

        Label format per line:
            <class_id> <x_center> <y_center> <width> <height>
        where coords are normalized to [0,1].
        """
        image_rel_path = Path(image_rel_path)
        if image_rel_path.suffix == "":
            image_rel_path = image_rel_path.with_suffix(".jpg")

        image_out_path = self.images_dir / image_rel_path
        label_out_path = (self.labels_dir / image_rel_path).with_suffix(".txt")

        image_out_path.parent.mkdir(parents=True, exist_ok=True)
        label_out_path.parent.mkdir(parents=True, exist_ok=True)

        save_image(image, image_out_path)

        h, w = image.shape[:2]
        label_lines: List[str] = []
        for det in detections:
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            class_name = det.get("class_name")
            if not class_name:
                class_name = str(det.get("class_id", "unknown"))

            class_id = self._class_id_for(str(class_name))
            xc, yc, bw, bh = _xyxy_to_yolo_xywh_norm(list(map(float, bbox)), w, h)
            label_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        label_out_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        return image_out_path, label_out_path


# Per-class overlay colors (RGB)
CLASS_OVERLAY_COLORS: Dict[str, Tuple[int, int, int]] = {
    "header":         (0, 255, 0),      # Green
    "footer":         (0, 0, 255),      # Blue
    "footnote":       (255, 0, 0),      # Red
    "text-area":      (64, 64, 64),     # Dark grey
}

# Fallback color if class_name is not in the map above
DEFAULT_OVERLAY_COLOR: Tuple[int, int, int] = (255, 255, 0)  # Yellow


@dataclass
class MaskedInferenceImageWriter:
    """Apply overlay mask to detected regions and save the result.

    The writer receives the **original** (unmasked) image together with
    detections, applies :meth:`apply_overlay_mask` internally, and saves
    the masked image to ``<base_dir>/<images_subdir>/<image_rel_path>``.
    """

    base_dir: Union[str, Path] = "inference_data"
    images_subdir: str = "images"
    margin: int = 0
    overlay_alpha: float = 0.35

    def __post_init__(self) -> None:
        """Create the output directory tree."""
        self.base_dir = Path(self.base_dir)
        self.images_dir = self.base_dir / self.images_subdir
        self.images_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Masking helpers (static so callers can use without instantiation)
    # ------------------------------------------------------------------

    @staticmethod
    def _color_for_class(class_name: str) -> Tuple[int, int, int]:
        """Return the BGR colour for a given class_name.

        Args:
            class_name: Detection class name (e.g. ``"header"``).

        Returns:
            BGR colour tuple for use with OpenCV.
        """
        rgb = CLASS_OVERLAY_COLORS.get(class_name, DEFAULT_OVERLAY_COLOR)
        return (rgb[2], rgb[1], rgb[0])  # RGB → BGR

    @staticmethod
    def apply_overlay_mask(
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        *,
        margin: int = 0,
        overlay_alpha: float = 0.35,
        min_confidence: Optional[float] = None,
    ) -> np.ndarray:
        """Draw translucent colored overlays on detected regions.

        Each class gets its own color (see :data:`CLASS_OVERLAY_COLORS`).

        Args:
            image: Input image as numpy array (BGR format from OpenCV).
            detections: List of detection dicts with ``bbox`` keys.
            margin: Extra pixels to add around detected regions.
            overlay_alpha: Opacity of the overlay (0.0–1.0). Default 0.35.
            min_confidence: Optional minimum confidence filter.

        Returns:
            Copy of *image* with translucent colored overlays on detected
            regions.
        """
        result = image.copy()
        overlay = result.copy()
        height, width = result.shape[:2]

        for detection in detections:
            if min_confidence is not None:
                if detection.get("confidence", 1.0) < min_confidence:
                    continue

            bbox = detection["bbox"]
            x1, y1, x2, y2 = map(int, bbox)

            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(width, x2 + margin)
            y2 = min(height, y2 + margin)

            bgr_color = MaskedInferenceImageWriter._color_for_class(
                detection.get("class_name", ""),
            )
            cv2.rectangle(overlay, (x1, y1), (x2, y2), bgr_color, -1)

        cv2.addWeighted(
            overlay, overlay_alpha, result, 1 - overlay_alpha, 0, result,
        )
        return result

    # ------------------------------------------------------------------
    # Writer interface
    # ------------------------------------------------------------------

    def write_sample(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        image_rel_path: Union[str, Path],
    ) -> Tuple[Path, None]:
        """Mask detected regions on *image* and save the result.

        Args:
            image: Original (unmasked) image in BGR format.
            detections: Detection dicts with ``bbox`` and ``class_name`` keys.
            image_rel_path: Relative filename for the saved image.

        Returns:
            Tuple of (saved image path, ``None``).
        """
        masked = self.apply_overlay_mask(
            image,
            detections,
            margin=self.margin,
            overlay_alpha=self.overlay_alpha,
        )

        image_rel_path = Path(image_rel_path)
        if image_rel_path.suffix == "":
            image_rel_path = image_rel_path.with_suffix(".jpg")

        image_out_path = self.images_dir / image_rel_path
        image_out_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(masked, image_out_path)
        return image_out_path, None


# Module-level convenience alias so callers don't need to reference the class.
apply_overlay_mask = MaskedInferenceImageWriter.apply_overlay_mask
