"""Tests for the HFF processor module."""

import numpy as np
import pytest

from hff_remover.processor import (
    HFFProcessor,
    CLASS_OVERLAY_COLORS,
    DEFAULT_OVERLAY_COLOR,
    COCODatasetWriter,
    MaskedInferenceImageWriter,
    apply_overlay_mask,
)


def _expected_overlay(base_bgr, overlay_rgb, alpha):
    """Compute the expected BGR pixel value after translucent overlay blending."""
    overlay_bgr = np.array([overlay_rgb[2], overlay_rgb[1], overlay_rgb[0]], dtype=np.float64)
    base = np.array(base_bgr, dtype=np.float64)
    blended = base * (1 - alpha) + overlay_bgr * alpha
    return np.round(blended).astype(np.uint8)


# ---------------------------------------------------------------------------
# Tests for the standalone apply_overlay_mask function
# ---------------------------------------------------------------------------

class TestApplyOverlayMask:
    """Tests for the apply_overlay_mask standalone function."""

    def test_empty_detections(self):
        """With no detections the image is returned unchanged."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image.fill(128)

        result = apply_overlay_mask(image, [])

        assert np.array_equal(result, image)

    def test_single_header(self):
        """Header detection produces a green overlay."""
        alpha = 0.35
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        detections = [{
            "bbox": [10, 10, 50, 30],
            "class_id": 2,
            "class_name": "header",
            "confidence": 0.9,
        }]

        result = apply_overlay_mask(image, detections, overlay_alpha=alpha)

        expected = _expected_overlay([0, 0, 0], CLASS_OVERLAY_COLORS["header"], alpha)
        assert np.all(result[10:30, 10:50] == expected)
        assert np.all(result[0:9, 0:9] == [0, 0, 0])

    def test_footer_blue(self):
        """Footer detection produces a blue overlay."""
        alpha = 0.5
        image = np.full((100, 100, 3), 200, dtype=np.uint8)

        detections = [{
            "bbox": [0, 90, 100, 100],
            "class_name": "footer",
            "confidence": 0.9,
        }]

        result = apply_overlay_mask(image, detections, overlay_alpha=alpha)

        expected = _expected_overlay([200, 200, 200], CLASS_OVERLAY_COLORS["footer"], alpha)
        assert np.all(result[90:100, 0:100] == expected)

    def test_with_margin(self):
        """Margin expands the overlay region."""
        alpha = 1.0
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        detections = [{
            "bbox": [20, 20, 40, 40],
            "class_name": "header",
            "confidence": 0.9,
        }]

        result = apply_overlay_mask(image, detections, margin=5, overlay_alpha=alpha)

        expected = _expected_overlay([0, 0, 0], CLASS_OVERLAY_COLORS["header"], alpha)
        assert np.all(result[15:45, 15:45] == expected)

    def test_confidence_filter(self):
        """Only detections above min_confidence are overlaid."""
        alpha = 0.35
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        detections = [
            {"bbox": [10, 10, 30, 30], "class_name": "header", "confidence": 0.3},
            {"bbox": [50, 50, 70, 70], "class_name": "header", "confidence": 0.8},
        ]

        result = apply_overlay_mask(image, detections, overlay_alpha=alpha, min_confidence=0.5)

        assert np.all(result[10:30, 10:30] == [0, 0, 0])
        expected = _expected_overlay([0, 0, 0], CLASS_OVERLAY_COLORS["header"], alpha)
        assert np.all(result[50:70, 50:70] == expected)

    def test_preserves_original(self):
        """The original image array must not be mutated."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        original = image.copy()

        detections = [{"bbox": [10, 10, 50, 30], "class_name": "footer", "confidence": 0.9}]
        apply_overlay_mask(image, detections)

        assert np.array_equal(image, original)

    def test_clamps_to_image_bounds(self):
        """Large margin must not produce out-of-bounds coordinates."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        detections = [{"bbox": [0, 0, 30, 20], "confidence": 0.9}]
        result = apply_overlay_mask(image, detections, margin=20)

        assert result.shape == image.shape


# ---------------------------------------------------------------------------
# Tests for HFFProcessor (merging only)
# ---------------------------------------------------------------------------

class TestHFFProcessor:
    """Tests for HFFProcessor class."""

    def test_init_default(self):
        """Test default initialization."""
        processor = HFFProcessor()
        assert processor.margin == 0

    def test_init_custom(self):
        """Test custom initialization."""
        processor = HFFProcessor(margin=10)
        assert processor.margin == 10


# ---------------------------------------------------------------------------
# Tests for COCODatasetWriter
# ---------------------------------------------------------------------------

class TestCOCODatasetWriter:
    def test_writes_image_label_and_data_yaml(self, tmp_path):
        writer = COCODatasetWriter(base_dir=tmp_path / "inference_data")

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
        # header -> class id 0
        parts = label_text[0].split()
        assert parts[0] == "0"
        # xc=50/200=0.25, yc=25/100=0.25, w=100/200=0.5, h=50/100=0.5
        assert parts[1] == "0.250000"
        assert parts[2] == "0.250000"
        assert parts[3] == "0.500000"
        assert parts[4] == "0.500000"

        data_yaml = (tmp_path / "inference_data" / "data.yaml").read_text(encoding="utf-8")
        assert "path:" in data_yaml
        assert "nc: 4" in data_yaml
        assert "names:" in data_yaml
        assert '0: "header"' in data_yaml
        assert '1: "text-area"' in data_yaml
        assert '2: "footnote"' in data_yaml
        assert '3: "footer"' in data_yaml

    def test_class_ids_are_consistent(self, tmp_path):
        writer = COCODatasetWriter(base_dir=tmp_path / "inference_data")

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        writer.write_sample(
            image=image,
            detections=[{"bbox": [0, 0, 5, 5], "class_name": "header"}],
            image_rel_path="a.jpg",
        )
        writer.write_sample(
            image=image,
            detections=[{"bbox": [0, 0, 5, 5], "class_name": "footnote"}],
            image_rel_path="b.jpg",
        )

        a_lbl = (tmp_path / "inference_data" / "labels" / "a.txt").read_text(encoding="utf-8").strip()
        b_lbl = (tmp_path / "inference_data" / "labels" / "b.txt").read_text(encoding="utf-8").strip()

        # header=0, footnote=2 per COCO_CLASS_NAME_TO_ID
        assert a_lbl.split()[0] == "0"
        assert b_lbl.split()[0] == "2"


# ---------------------------------------------------------------------------
# Tests for MaskedInferenceImageWriter (now handles masking internally)
# ---------------------------------------------------------------------------

class TestMaskedInferenceImageWriter:
    def test_writes_masked_image(self, tmp_path):
        """Writer should apply overlay mask and save the result."""
        writer = MaskedInferenceImageWriter(
            base_dir=tmp_path / "inference_data",
            overlay_alpha=1.0,  # fully opaque for easy assertion
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [{"bbox": [10, 10, 50, 30], "class_name": "header", "confidence": 0.9}]

        img_path, lbl_path = writer.write_sample(
            image=image,
            detections=detections,
            image_rel_path="x.jpg",
        )

        assert img_path.exists()
        assert lbl_path is None
        assert (tmp_path / "inference_data" / "images" / "x.jpg").exists()
        # Should not create labels folder or data.yaml
        assert not (tmp_path / "inference_data" / "labels").exists()
        assert not (tmp_path / "inference_data" / "data.yaml").exists()

    def test_saved_image_is_masked(self, tmp_path):
        """The saved image must differ from the original when detections exist."""
        import cv2

        writer = MaskedInferenceImageWriter(
            base_dir=tmp_path / "inference_data",
            overlay_alpha=0.5,
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [{"bbox": [0, 0, 50, 50], "class_name": "header", "confidence": 0.9}]

        img_path, _ = writer.write_sample(
            image=image,
            detections=detections,
            image_rel_path="masked.png",
        )

        saved = cv2.imread(str(img_path))
        # The overlay region should NOT be all-black any more
        assert not np.array_equal(saved[0:50, 0:50], np.zeros((50, 50, 3), dtype=np.uint8))

    def test_no_detections_preserves_image(self, tmp_path):
        """With no detections the saved image should match the original."""
        import cv2

        writer = MaskedInferenceImageWriter(base_dir=tmp_path / "inference_data")
        image = np.full((10, 20, 3), 128, dtype=np.uint8)

        img_path, _ = writer.write_sample(
            image=image,
            detections=[],
            image_rel_path="unchanged.png",
        )

        saved = cv2.imread(str(img_path))
        assert np.array_equal(saved, image)
