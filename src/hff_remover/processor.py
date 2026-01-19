"""Image processor for masking detected HFF regions."""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2


class HFFProcessor:
    """Processor for masking headers, footers, and footnotes in images."""

    def __init__(
        self,
        mask_color: Tuple[int, int, int] = (255, 255, 255),
        padding: int = 0,
    ):
        """
        Initialize the HFF processor.

        Args:
            mask_color: RGB color to use for masking (default: white).
            padding: Extra pixels to add around detected regions.
        """
        self.mask_color = mask_color
        self.padding = padding

    def mask_regions(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        min_confidence: Optional[float] = None,
    ) -> np.ndarray:
        """
        Mask detected HFF regions in an image.

        Args:
            image: Input image as numpy array (BGR format from OpenCV).
            detections: List of detection dictionaries with 'bbox' keys.
            min_confidence: Optional minimum confidence filter.

        Returns:
            Image with HFF regions masked.
        """
        # Make a copy to avoid modifying the original
        result = image.copy()
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

            # Convert RGB to BGR for OpenCV
            bgr_color = (self.mask_color[2], self.mask_color[1], self.mask_color[0])

            # Fill the region with mask color
            cv2.rectangle(result, (x1, y1), (x2, y2), bgr_color, -1)

        return result

    def mask_regions_smooth(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        blur_radius: int = 5,
        min_confidence: Optional[float] = None,
    ) -> np.ndarray:
        """
        Mask detected HFF regions with smooth edges.

        This creates a softer transition at the edges of masked regions.

        Args:
            image: Input image as numpy array (BGR format).
            detections: List of detection dictionaries with 'bbox' keys.
            blur_radius: Radius for edge blurring.
            min_confidence: Optional minimum confidence filter.

        Returns:
            Image with HFF regions masked with smooth edges.
        """
        result = image.copy()
        height, width = result.shape[:2]

        # Create a mask for all regions
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

            # Mark region in mask
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # Blur the mask for smooth edges
        if blur_radius > 0:
            mask = cv2.GaussianBlur(mask, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)

        # Normalize mask to 0-1 range
        mask_normalized = mask.astype(np.float32) / 255.0

        # Create white background
        bgr_color = (self.mask_color[2], self.mask_color[1], self.mask_color[0])
        white_bg = np.full_like(result, bgr_color, dtype=np.uint8)

        # Blend based on mask
        mask_3ch = np.stack([mask_normalized] * 3, axis=-1)
        result = (result * (1 - mask_3ch) + white_bg * mask_3ch).astype(np.uint8)

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

            # Mark HFF region as 0
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
