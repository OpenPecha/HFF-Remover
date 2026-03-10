"""Image processor for masking detected HFF regions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

import numpy as np
import cv2

from hff_remover.utils import save_image


# Per-class overlay colors (RGB)
CLASS_OVERLAY_COLORS: Dict[str, Tuple[int, int, int]] = {
    "header":         (0, 255, 0),      # Green
    "footer":         (0, 0, 255),      # Blue
    "footnote":       (255, 0, 0),      # Red
    "text-area":      (64, 64, 64),     # Dark grey
}

# Fallback color if class_name is not in the map above
DEFAULT_OVERLAY_COLOR: Tuple[int, int, int] = (255, 255, 0)  # Yellow


class HFFProcessor:
    """Processor for overlaying translucent colored boxes on detected regions."""

    def __init__(
        self,
        mask_color: Tuple[int, int, int] = (255, 255, 255),
        padding: int = 0,
        overlay_alpha: float = 0.35,
    ):
        """
        Initialize the HFF processor.

        Args:
            mask_color: RGB fallback color (kept for API compat).
            padding: Extra pixels to add around detected regions.
            overlay_alpha: Opacity of the translucent overlay (0.0 = fully
                transparent, 1.0 = fully opaque). Default 0.35.
        """
        self.mask_color = mask_color
        self.padding = padding
        self.overlay_alpha = overlay_alpha

    @staticmethod
    def _color_for_class(class_name: str) -> Tuple[int, int, int]:
        """Return the BGR colour for a given class_name."""
        rgb = CLASS_OVERLAY_COLORS.get(class_name, DEFAULT_OVERLAY_COLOR)
        return (rgb[2], rgb[1], rgb[0])  # RGB → BGR

    def mask_regions(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        min_confidence: Optional[float] = None,
    ) -> np.ndarray:
        """
        Draw translucent colored overlays on detected regions.

        Each class gets its own color (see CLASS_OVERLAY_COLORS).

        Args:
            image: Input image as numpy array (BGR format from OpenCV).
            detections: List of detection dictionaries with 'bbox' keys.
            min_confidence: Optional minimum confidence filter.

        Returns:
            Image with translucent colored overlays on detected regions.
        """
        # Make a copy to avoid modifying the original
        result = image.copy()
        overlay = result.copy()
        height, width = result.shape[:2]

        for detection in detections:
            # Filter by confidence if specified
            if min_confidence is not None:
                if detection.get("confidence", 1.0) < min_confidence:
                    continue

            bbox = detection["bbox"]
            x1, y1, x2, y2 = map(int, bbox)

            # Apply padding
            x1 = max(0, x1 - self.padding)
            y1 = max(0, y1 - self.padding)
            x2 = min(width, x2 + self.padding)
            y2 = min(height, y2 + self.padding)

            # Get class-specific BGR color
            bgr_color = self._color_for_class(detection.get("class_name", ""))

            # Draw filled rectangle on overlay layer
            cv2.rectangle(overlay, (x1, y1), (x2, y2), bgr_color, -1)

        # Blend the overlay with the original
        cv2.addWeighted(overlay, self.overlay_alpha, result, 1 - self.overlay_alpha, 0, result)

        return result

    def mask_regions_smooth(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        blur_radius: int = 5,
        min_confidence: Optional[float] = None,
    ) -> np.ndarray:
        """
        Draw translucent colored overlays with smooth (blurred) edges.

        Each class gets its own color (see CLASS_OVERLAY_COLORS).

        Args:
            image: Input image as numpy array (BGR format).
            detections: List of detection dictionaries with 'bbox' keys.
            blur_radius: Radius for edge blurring.
            min_confidence: Optional minimum confidence filter.

        Returns:
            Image with translucent overlays with smooth edges.
        """
        result = image.copy()
        height, width = result.shape[:2]

        # Build a coloured overlay image and a corresponding intensity mask
        overlay = np.zeros_like(result)
        mask = np.zeros((height, width), dtype=np.uint8)

        for detection in detections:
            if min_confidence is not None:
                if detection.get("confidence", 1.0) < min_confidence:
                    continue

            bbox = detection["bbox"]
            x1, y1, x2, y2 = map(int, bbox)

            # Apply padding
            x1 = max(0, x1 - self.padding)
            y1 = max(0, y1 - self.padding)
            x2 = min(width, x2 + self.padding)
            y2 = min(height, y2 + self.padding)

            bgr_color = self._color_for_class(detection.get("class_name", ""))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), bgr_color, -1)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # Blur the mask for smooth edges
        if blur_radius > 0:
            mask = cv2.GaussianBlur(mask, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)

        # Normalize mask to 0-1 and apply alpha
        alpha_map = (mask.astype(np.float32) / 255.0) * self.overlay_alpha
        alpha_3ch = np.stack([alpha_map] * 3, axis=-1)

        result = (result * (1 - alpha_3ch) + overlay * alpha_3ch).astype(np.uint8)

        return result

    def get_clean_region_mask(
        self,
        image_shape: Tuple[int, int],
        detections: List[Dict[str, Any]],
        min_confidence: Optional[float] = None,
    ) -> np.ndarray:
        """
        Create a binary mask where clean regions (non-HFF) are white.

        Useful for extracting only the main content.

        Args:
            image_shape: (height, width) of the image.
            detections: List of detection dictionaries.
            min_confidence: Optional minimum confidence filter.

        Returns:
            Binary mask (255 for clean regions, 0 for HFF regions).
        """
        height, width = image_shape[:2]
        mask = np.ones((height, width), dtype=np.uint8) * 255

        for detection in detections:
            if min_confidence is not None:
                if detection.get("confidence", 1.0) < min_confidence:
                    continue

            bbox = detection["bbox"]
            x1, y1, x2, y2 = map(int, bbox)

            # Apply padding
            x1 = max(0, x1 - self.padding)
            y1 = max(0, y1 - self.padding)
            x2 = min(width, x2 + self.padding)
            y2 = min(height, y2 + self.padding)

            # Mark detected region as 0
            cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

        return mask

    def extract_main_content(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        min_confidence: Optional[float] = None,
        background_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """
        Extract main content by replacing HFF regions with background.

        Args:
            image: Input image as numpy array (BGR format).
            detections: List of detection dictionaries.
            min_confidence: Optional minimum confidence filter.
            background_color: RGB color for replaced regions.

        Returns:
            Image with only main content visible.
        """
        mask = self.get_clean_region_mask(
            image.shape, detections, min_confidence
        )

        result = image.copy()
        bgr_color = (background_color[2], background_color[1], background_color[0])

        # Replace HFF regions with background color
        result[mask == 0] = bgr_color

        return result


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
    expects_masked_images: bool = False

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


@dataclass
class MaskedInferenceImageWriter:
    """
    Save masked inference images only:
    - inference_data/images/<image>
    """

    base_dir: Union[str, Path] = "inference_data"
    images_subdir: str = "images"
    expects_masked_images: bool = True

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir)
        self.images_dir = self.base_dir / self.images_subdir
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def write_sample(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        image_rel_path: Union[str, Path],
    ) -> Tuple[Path, None]:
        """
        Save one image to `images/`. `detections` is ignored (kept for API compatibility).
        """
        _ = detections
        image_rel_path = Path(image_rel_path)
        if image_rel_path.suffix == "":
            image_rel_path = image_rel_path.with_suffix(".jpg")

        image_out_path = self.images_dir / image_rel_path
        image_out_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(image, image_out_path)
        return image_out_path, None
