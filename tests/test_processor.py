"""Tests for the HFF processor module."""

import numpy as np
import pytest

from hff_remover.processor import HFFProcessor


class TestHFFProcessor:
    """Tests for HFFProcessor class."""

    def test_init_default(self):
        """Test default initialization."""
        processor = HFFProcessor()
        assert processor.mask_color == (255, 255, 255)
        assert processor.padding == 0

    def test_init_custom(self):
        """Test custom initialization."""
        processor = HFFProcessor(
            mask_color=(0, 0, 0),
            padding=10,
        )
        assert processor.mask_color == (0, 0, 0)
        assert processor.padding == 10

    def test_mask_regions_empty(self):
        """Test masking with no detections."""
        processor = HFFProcessor()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image.fill(128)  # Gray image

        result = processor.mask_regions(image, [])

        # Should be unchanged
        assert np.array_equal(result, image)

    def test_mask_regions_single(self):
        """Test masking a single region."""
        processor = HFFProcessor()
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        detections = [{
            "bbox": [10, 10, 50, 30],
            "class_id": 2,
            "class_name": "abandon",
            "confidence": 0.9,
        }]

        result = processor.mask_regions(image, detections)

        # Check that the region is white (BGR format)
        assert np.all(result[10:30, 10:50] == [255, 255, 255])
        # Check that outside region is unchanged
        assert np.all(result[0:9, 0:9] == [0, 0, 0])

    def test_mask_regions_with_padding(self):
        """Test masking with padding."""
        processor = HFFProcessor(padding=5)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        detections = [{
            "bbox": [20, 20, 40, 40],
            "class_id": 2,
            "class_name": "abandon",
            "confidence": 0.9,
        }]

        result = processor.mask_regions(image, detections)

        # Check that padded region is white
        assert np.all(result[15:45, 15:45] == [255, 255, 255])

    def test_mask_regions_confidence_filter(self):
        """Test confidence filtering."""
        processor = HFFProcessor()
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        detections = [
            {
                "bbox": [10, 10, 30, 30],
                "class_id": 2,
                "class_name": "abandon",
                "confidence": 0.3,
            },
            {
                "bbox": [50, 50, 70, 70],
                "class_id": 2,
                "class_name": "abandon",
                "confidence": 0.8,
            },
        ]

        result = processor.mask_regions(image, detections, min_confidence=0.5)

        # Low confidence region should not be masked
        assert np.all(result[10:30, 10:30] == [0, 0, 0])
        # High confidence region should be masked
        assert np.all(result[50:70, 50:70] == [255, 255, 255])

    def test_mask_regions_multiple(self):
        """Test masking multiple regions."""
        processor = HFFProcessor()
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        detections = [
            {"bbox": [0, 0, 20, 10], "confidence": 0.9},
            {"bbox": [0, 90, 100, 100], "confidence": 0.9},
        ]

        result = processor.mask_regions(image, detections)

        # Both regions should be masked
        assert np.all(result[0:10, 0:20] == [255, 255, 255])
        assert np.all(result[90:100, 0:100] == [255, 255, 255])

    def test_mask_regions_custom_color(self):
        """Test masking with custom color."""
        processor = HFFProcessor(mask_color=(255, 0, 0))  # Red in RGB
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        detections = [{
            "bbox": [10, 10, 50, 30],
            "confidence": 0.9,
        }]

        result = processor.mask_regions(image, detections)

        # Check that the region is red (BGR format: Blue, Green, Red)
        assert np.all(result[10:30, 10:50] == [0, 0, 255])

    def test_mask_regions_preserves_original(self):
        """Test that original image is not modified."""
        processor = HFFProcessor()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        original = image.copy()

        detections = [{"bbox": [10, 10, 50, 30], "confidence": 0.9}]
        processor.mask_regions(image, detections)

        assert np.array_equal(image, original)

    def test_get_clean_region_mask(self):
        """Test clean region mask generation."""
        processor = HFFProcessor()

        detections = [
            {"bbox": [0, 0, 50, 10], "confidence": 0.9},  # Header
            {"bbox": [0, 90, 100, 100], "confidence": 0.9},  # Footer
        ]

        mask = processor.get_clean_region_mask((100, 100), detections)

        # HFF regions should be 0
        assert np.all(mask[0:10, 0:50] == 0)
        assert np.all(mask[90:100, :] == 0)
        # Clean regions should be 255
        assert np.all(mask[20:80, :] == 255)

    def test_mask_regions_clamps_to_image_bounds(self):
        """Test that bounding boxes are clamped to image bounds."""
        processor = HFFProcessor(padding=20)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Detection near edge with padding that would go out of bounds
        detections = [{
            "bbox": [0, 0, 30, 20],
            "confidence": 0.9,
        }]

        result = processor.mask_regions(image, detections)

        # Should not raise an error and region should be masked
        # Padding should be clamped to image bounds
        assert result.shape == image.shape
