"""Utility functions for image I/O and file handling."""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image


# Supported image formats
SUPPORTED_FORMATS = {
    ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp", ".gif"
}


def is_supported_image(path: Union[str, Path]) -> bool:
    """Check if a file is a supported image format."""
    return Path(path).suffix.lower() in SUPPORTED_FORMATS


def find_images(
    directory: Union[str, Path],
    recursive: bool = True,
) -> List[Path]:
    """
    Find all supported images in a directory.

    Args:
        directory: Directory to search.
        recursive: Whether to search subdirectories.

    Returns:
        List of paths to image files.
    """
    directory = Path(directory)
    images = []

    if recursive:
        for ext in SUPPORTED_FORMATS:
            images.extend(directory.rglob(f"*{ext}"))
            images.extend(directory.rglob(f"*{ext.upper()}"))
    else:
        for ext in SUPPORTED_FORMATS:
            images.extend(directory.glob(f"*{ext}"))
            images.extend(directory.glob(f"*{ext.upper()}"))

    # Remove duplicates and sort
    images = sorted(set(images))
    return images


def load_image(
    path: Union[str, Path],
    color_mode: str = "bgr",
) -> np.ndarray:
    """
    Load an image from disk.

    Args:
        path: Path to the image file.
        color_mode: Color mode ('bgr', 'rgb', or 'gray').

    Returns:
        Image as numpy array.

    Raises:
        FileNotFoundError: If the image file doesn't exist.
        ValueError: If the image cannot be loaded.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # Use PIL for TIFF files (better support)
    if path.suffix.lower() in {".tiff", ".tif"}:
        pil_image = Image.open(path)

        # Convert to RGB if needed
        if pil_image.mode not in ("RGB", "L"):
            pil_image = pil_image.convert("RGB")

        image = np.array(pil_image)

        if color_mode == "bgr" and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif color_mode == "gray":
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        # Use OpenCV for other formats
        if color_mode == "gray":
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Failed to load image: {path}")

        if color_mode == "rgb":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def save_image(
    image: np.ndarray,
    path: Union[str, Path],
    quality: int = 95,
    color_mode: str = "bgr",
) -> None:
    """
    Save an image to disk.

    Args:
        image: Image as numpy array.
        path: Output path.
        quality: JPEG quality (0-100) or PNG compression (0-9).
        color_mode: Color mode of input image ('bgr' or 'rgb').
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert color if needed
    if color_mode == "rgb" and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    suffix = path.suffix.lower()

    # Set compression parameters based on format
    if suffix in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif suffix == ".png":
        # PNG uses 0-9 scale for compression
        compression = min(9, max(0, (100 - quality) // 10))
        params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
    elif suffix in {".tiff", ".tif"}:
        # Use PIL for TIFF to preserve quality
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
        pil_image.save(str(path), compression="tiff_lzw")
        return
    else:
        params = []

    success = cv2.imwrite(str(path), image, params)
    if not success:
        raise IOError(f"Failed to save image: {path}")


def get_output_path(
    input_path: Union[str, Path],
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    preserve_format: bool = True,
    output_format: Optional[str] = None,
) -> Path:
    """
    Generate output path preserving directory structure.

    Args:
        input_path: Path to the input image.
        input_dir: Base input directory.
        output_dir: Base output directory.
        preserve_format: Whether to keep the original format.
        output_format: Override output format (e.g., '.png').

    Returns:
        Output path.
    """
    input_path = Path(input_path)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Get relative path from input directory
    try:
        relative_path = input_path.relative_to(input_dir)
    except ValueError:
        # If not relative, just use the filename
        relative_path = input_path.name

    output_path = output_dir / relative_path

    if not preserve_format and output_format:
        output_path = output_path.with_suffix(output_format)

    return output_path


def get_image_info(path: Union[str, Path]) -> dict:
    """
    Get basic information about an image.

    Args:
        path: Path to the image file.

    Returns:
        Dictionary with image info (width, height, channels, format).
    """
    path = Path(path)

    with Image.open(path) as img:
        return {
            "width": img.width,
            "height": img.height,
            "channels": len(img.getbands()),
            "format": img.format,
            "mode": img.mode,
            "file_size": path.stat().st_size,
        }


def resize_image(
    image: np.ndarray,
    max_size: Optional[int] = None,
    min_size: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """
    Resize an image while maintaining aspect ratio.

    Args:
        image: Input image.
        max_size: Maximum dimension (width or height).
        min_size: Minimum dimension (width or height).

    Returns:
        Tuple of (resized image, scale factor).
    """
    height, width = image.shape[:2]
    scale = 1.0

    if max_size is not None:
        max_dim = max(height, width)
        if max_dim > max_size:
            scale = max_size / max_dim

    if min_size is not None:
        min_dim = min(height, width)
        if min_dim * scale < min_size:
            scale = min_size / min_dim

    if scale != 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return image, scale


def scale_bboxes(
    bboxes: List[List[float]],
    scale: float,
) -> List[List[float]]:
    """
    Scale bounding boxes by a factor.

    Args:
        bboxes: List of [x1, y1, x2, y2] bounding boxes.
        scale: Scale factor.

    Returns:
        Scaled bounding boxes.
    """
    return [[coord / scale for coord in bbox] for bbox in bboxes]
