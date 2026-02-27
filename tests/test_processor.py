"""Tests for the HFF processor module."""

import numpy as np
import pytest

from hff_remover.processor import (
    HFFProcessor,
    CLASS_OVERLAY_COLORS,
    DEFAULT_OVERLAY_COLOR,
    YOLOInferenceDatasetWriter,
    MaskedInferenceImageWriter,
)


def _expected_overlay(base_bgr, overlay_rgb, alpha):
    """Compute the expected BGR pixel value after translucent overlay blending."""
    overlay_bgr = np.array([overlay_rgb[2], overlay_rgb[1], overlay_rgb[0]], dtype=np.float64)
    base = np.array(base_bgr, dtype=np.float64)
    blended = base * (1 - alpha) + overlay_bgr * alpha
    return np.round(blended).astype(np.uint8)


class TestHFFProcessor:
    """Tests for HFFProcessor class."""

    def test_init_default(self):
        """Test default initialization."""
        processor = HFFProcessor()
        assert processor.mask_color == (255, 255, 255)
        assert processor.padding == 0
        assert processor.overlay_alpha == 0.35

    def test_init_custom(self):
        """Test custom initialization."""
        processor = HFFProcessor(
            mask_color=(0, 0, 0),
            padding=10,
            overlay_alpha=0.5,
        )
        assert processor.mask_color == (0, 0, 0)
        assert processor.padding == 10
        assert processor.overlay_alpha == 0.5

    def test_mask_regions_empty(self):
        """Test masking with no detections."""
        processor = HFFProcessor()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image.fill(128)  # Gray image

        result = processor.mask_regions(image, [])

        # Should be unchanged
        assert np.array_equal(result, image)

    def test_mask_regions_single_header(self):
        """Test overlay for a header detection (green)."""
        alpha = 0.35
        processor = HFFProcessor(overlay_alpha=alpha)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        detections = [{
            "bbox": [10, 10, 50, 30],
            "class_id": 2,
            "class_name": "header",
            "confidence": 0.9,
        }]

        result = processor.mask_regions(image, detections)

        expected = _expected_overlay([0, 0, 0], CLASS_OVERLAY_COLORS["header"], alpha)
        # All pixels in the region should be the blended colour
        assert np.all(result[10:30, 10:50] == expected)
        # Outside region is unchanged (black)
        assert np.all(result[0:9, 0:9] == [0, 0, 0])

    def test_mask_regions_footer_blue(self):
        """Test overlay for a footer detection (blue)."""
        alpha = 0.5
        processor = HFFProcessor(overlay_alpha=alpha)
        image = np.full((100, 100, 3), 200, dtype=np.uint8)

        detections = [{
            "bbox": [0, 90, 100, 100],
            "class_name": "footer",
            "confidence": 0.9,
        }]

        result = processor.mask_regions(image, detections)

        expected = _expected_overlay([200, 200, 200], CLASS_OVERLAY_COLORS["footer"], alpha)
        assert np.all(result[90:100, 0:100] == expected)

    def test_mask_regions_text_dark_grey(self):
        """Text areas get a dark-grey overlay, not skipped."""
        alpha = 0.35
        processor = HFFProcessor(overlay_alpha=alpha)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        detections = [{
            "bbox": [10, 10, 50, 50],
            "class_name": "text",
            "confidence": 0.9,
        }]

        result = processor.mask_regions(image, detections)

        expected = _expected_overlay([0, 0, 0], CLASS_OVERLAY_COLORS["text"], alpha)
        assert np.all(result[10:50, 10:50] == expected)

    def test_mask_regions_with_padding(self):
        """Test overlay with padding."""
        alpha = 1.0  # fully opaque for easy assertion
        processor = HFFProcessor(padding=5, overlay_alpha=alpha)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        detections = [{
            "bbox": [20, 20, 40, 40],
            "class_name": "header",
            "confidence": 0.9,
        }]

        result = processor.mask_regions(image, detections)

        expected = _expected_overlay([0, 0, 0], CLASS_OVERLAY_COLORS["header"], alpha)
        # Padded region (15-45) should be overlaid
        assert np.all(result[15:45, 15:45] == expected)

    def test_mask_regions_confidence_filter(self):
        """Test confidence filtering."""
        alpha = 0.35
        processor = HFFProcessor(overlay_alpha=alpha)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        detections = [
            {
                "bbox": [10, 10, 30, 30],
                "class_name": "header",
                "confidence": 0.3,
            },
            {
                "bbox": [50, 50, 70, 70],
                "class_name": "header",
                "confidence": 0.8,
            },
        ]

        result = processor.mask_regions(image, detections, min_confidence=0.5)

        # Low confidence region should not be overlaid
        assert np.all(result[10:30, 10:30] == [0, 0, 0])
        # High confidence region should be overlaid
        expected = _expected_overlay([0, 0, 0], CLASS_OVERLAY_COLORS["header"], alpha)
        assert np.all(result[50:70, 50:70] == expected)

    def test_mask_regions_preserves_original(self):
        """Test that original image is not modified."""
        processor = HFFProcessor()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        original = image.copy()

        detections = [{"bbox": [10, 10, 50, 30], "class_name": "footer", "confidence": 0.9}]
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

        # Detected regions should be 0
        assert np.all(mask[0:10, 0:50] == 0)
        assert np.all(mask[90:100, :] == 0)
        # Clean regions should be 255
        assert np.all(mask[20:80, :] == 255)

    def test_mask_regions_clamps_to_image_bounds(self):
        """Test that bounding boxes are clamped to image bounds."""
        processor = HFFProcessor(padding=20)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        detections = [{
            "bbox": [0, 0, 30, 20],
            "confidence": 0.9,
        }]

        result = processor.mask_regions(image, detections)

        # Should not raise and shape is preserved
        assert result.shape == image.shape


class TestYOLOInferenceDatasetWriter:
    def test_writes_image_label_and_data_yaml(self, tmp_path):
        writer = YOLOInferenceDatasetWriter(base_dir=tmp_path / "inference_data")

        image = np.zeros((100, 200, 3), dtype=np.uint8)  # h=100, w=200
        detections = [
            {"bbox": [0, 0, 100, 50], "class_name": "header", "confidence": 0.9},
        ]

        img_path, lbl_path = writer.write_sample(
            image=image,
            detections=detections,
            image_rel_path="page1.jpg",
        )

        assert img_path.exists()
        assert lbl_path.exists()
        assert (tmp_path / "inference_data" / "data.yaml").exists()

        label_text = lbl_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(label_text) == 1
        # class id should start at 0
        parts = label_text[0].split()
        assert parts[0] == "0"
        # xc=50/200=0.25, yc=25/100=0.25, w=100/200=0.5, h=50/100=0.5
        assert parts[1] == "0.250000"
        assert parts[2] == "0.250000"
        assert parts[3] == "0.500000"
        assert parts[4] == "0.500000"

        data_yaml = (tmp_path / "inference_data" / "data.yaml").read_text(encoding="utf-8")
        assert "path:" in data_yaml
        assert "names:" in data_yaml
        assert "header" in data_yaml

    def test_class_ids_are_consistent(self, tmp_path):
        writer = YOLOInferenceDatasetWriter(base_dir=tmp_path / "inference_data")

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        writer.write_sample(
            image=image,
            detections=[{"bbox": [0, 0, 5, 5], "class_name": "header"}],
            image_rel_path="a.jpg",
        )
        writer.write_sample(
            image=image,
            detections=[{"bbox": [0, 0, 5, 5], "class_name": "table_footnote"}],
            image_rel_path="b.jpg",
        )

        a_lbl = (tmp_path / "inference_data" / "labels" / "a.txt").read_text(encoding="utf-8").strip()
        b_lbl = (tmp_path / "inference_data" / "labels" / "b.txt").read_text(encoding="utf-8").strip()

        assert a_lbl.split()[0] == "0"
        assert b_lbl.split()[0] == "1"


class TestMaskedInferenceImageWriter:
    def test_writes_only_images(self, tmp_path):
        writer = MaskedInferenceImageWriter(base_dir=tmp_path / "inference_data")
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        img_path, lbl_path = writer.write_sample(
            image=image,
            detections=[{"bbox": [0, 0, 5, 5], "class_name": "header"}],
            image_rel_path="x.jpg",
        )

        assert img_path.exists()
        assert lbl_path is None
        assert (tmp_path / "inference_data" / "images" / "x.jpg").exists()
        # Should not create labels folder or data.yaml
        assert not (tmp_path / "inference_data" / "labels").exists()
        assert not (tmp_path / "inference_data" / "data.yaml").exists()
