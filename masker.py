"""Visualise YOLO bounding-box annotations on images with translucent overlays.

Reads images from ``benckmark_dataset/images/`` and their corresponding YOLO
label files from ``benckmark_dataset/label/``, draws class-coloured translucent
rectangles plus text labels, and writes annotated copies to
``benckmark_dataset/masked_output/``.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent / "./data/tdla_batch11"
IMAGES_DIR = BASE_DIR / "images"
LABELS_DIR = BASE_DIR / "labels"
OUTPUT_DIR = BASE_DIR / "masked_output"
DATA_YAML = BASE_DIR / "data.yaml"

# BGR colours keyed by class index
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 0, 0),       # header  → blue
    1: (180, 180, 180),   # Text area → grey
    2: (0, 0, 255),       # footnote → red
    3: (203, 192, 255),   # footer  → pink
}

OVERLAY_ALPHA = 0.35  # translucency for the filled rectangle

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BBox:
    """A single YOLO bounding box converted to pixel coordinates."""

    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_class_names(yaml_path: Path) -> dict[int, str]:
    """Load class-id → name mapping from a YOLO ``data.yaml``.

    Args:
        yaml_path: Path to the ``data.yaml`` file.

    Returns:
        Dictionary mapping integer class ids to their string names.

    Raises:
        FileNotFoundError: If *yaml_path* does not exist.
        KeyError: If the YAML lacks a ``names`` key.
    """
    with yaml_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    names: dict[int, str] = {int(k): v for k, v in data["names"].items()}
    return names


def parse_label_file(
    label_path: Path,
    img_width: int,
    img_height: int,
) -> list[BBox]:
    """Parse a YOLO label ``.txt`` file into pixel-space bounding boxes.

    Each line has the format ``class_id cx cy w h`` where all coordinates are
    normalised to [0, 1].

    Args:
        label_path: Path to the label text file.
        img_width: Width of the corresponding image in pixels.
        img_height: Height of the corresponding image in pixels.

    Returns:
        List of :class:`BBox` instances.
    """
    boxes: list[BBox] = []
    with label_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                logger.warning("Skipping malformed line in %s: %s", label_path, line)
                continue

            class_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            # Convert normalised YOLO coords → pixel coords
            x1 = int((cx - w / 2) * img_width)
            y1 = int((cy - h / 2) * img_height)
            x2 = int((cx + w / 2) * img_width)
            y2 = int((cy + h / 2) * img_height)

            # Clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)

            boxes.append(BBox(class_id=class_id, x1=x1, y1=y1, x2=x2, y2=y2))
    return boxes


def draw_overlays(
    image: np.ndarray,
    boxes: Sequence[BBox],
    class_names: dict[int, str],
    alpha: float = OVERLAY_ALPHA,
) -> np.ndarray:
    """Draw translucent coloured rectangles and class labels onto *image*.

    Args:
        image: BGR image array (will **not** be modified in place).
        boxes: Bounding boxes to draw.
        class_names: Mapping of class id → display name.
        alpha: Opacity of the filled rectangle overlay (0 = invisible, 1 = opaque).

    Returns:
        A copy of *image* with the overlays applied.
    """
    result = image.copy()

    for box in boxes:
        color = CLASS_COLORS.get(box.class_id, (255, 255, 255))
        label = class_names.get(box.class_id, f"class_{box.class_id}")

        # --- translucent filled rectangle ---
        overlay = result.copy()
        cv2.rectangle(overlay, (box.x1, box.y1), (box.x2, box.y2), color, thickness=-1)
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)

        # --- solid border ---
        cv2.rectangle(result, (box.x1, box.y1), (box.x2, box.y2), color, thickness=2)

        # --- text label above the box ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Position: just above top-left of the box
        text_x = box.x1
        text_y = box.y1 - 6
        if text_y - th < 0:
            text_y = box.y1 + th + 6  # fall back below the top edge

        # Draw a small filled background behind the text for readability
        cv2.rectangle(
            result,
            (text_x, text_y - th - baseline),
            (text_x + tw, text_y + baseline),
            color,
            thickness=-1,
        )
        cv2.putText(result, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: process all images and save annotated copies."""
    if not IMAGES_DIR.is_dir():
        logger.error("Images directory not found: %s", IMAGES_DIR)
        sys.exit(1)
    if not LABELS_DIR.is_dir():
        logger.error("Labels directory not found: %s", LABELS_DIR)
        sys.exit(1)

    class_names = load_class_names(DATA_YAML)
    logger.info("Loaded %d class names from %s", len(class_names), DATA_YAML)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(IMAGES_DIR.glob("*.jpg"))
    if not image_paths:
        logger.warning("No .jpg images found in %s", IMAGES_DIR)
        return

    processed = 0
    skipped = 0

    for img_path in image_paths:
        label_path = LABELS_DIR / f"{img_path.stem}.txt"
        if not label_path.exists():
            logger.warning("No label file for %s — skipping", img_path.name)
            skipped += 1
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Failed to read image %s — skipping", img_path.name)
            skipped += 1
            continue

        img_h, img_w = image.shape[:2]
        boxes = parse_label_file(label_path, img_w, img_h)
        annotated = draw_overlays(image, boxes, class_names)

        out_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(out_path), annotated)
        processed += 1

    logger.info(
        "Done — %d images annotated, %d skipped. Output → %s",
        processed,
        skipped,
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
