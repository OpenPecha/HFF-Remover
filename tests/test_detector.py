"""Tests for the detector module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hff_remover.detector import HFFDetector, HFF_CLASSES, CLASS_NAMES


class TestHFFClasses:
    """Tests for class definitions."""

    def test_hff_classes_defined(self):
        """Test that HFF classes are properly defined."""
        assert 2 in HFF_CLASSES  # abandon (headers/footers)
        assert 7 in HFF_CLASSES  # table_footnote

    def test_class_names_complete(self):
        """Test that all class names are defined."""
        for class_id in range(10):
            assert class_id in CLASS_NAMES


class TestHFFDetector:
    """Tests for HFFDetector class."""

    @patch("hff_remover.detector._yolo.hf_hub_download")
    @patch("doclayout_yolo.YOLOv10")
    def test_init_downloads_model(self, mock_yolo, mock_download):
        """Test that model is downloaded if path not provided."""
        mock_download.return_value = "/path/to/model.pt"

        detector = HFFDetector()

        mock_download.assert_called_once()
        mock_yolo.assert_called_once_with("/path/to/model.pt")

    @patch("doclayout_yolo.YOLOv10")
    def test_init_with_custom_path(self, mock_yolo):
        """Test initialization with custom model path."""
        detector = HFFDetector(model_path="/custom/model.pt")

        mock_yolo.assert_called_once_with("/custom/model.pt")
        assert detector.model_path == "/custom/model.pt"

    @patch("hff_remover.detector._yolo.hf_hub_download")
    @patch("doclayout_yolo.YOLOv10")
    def test_detect_filters_hff_classes(self, mock_yolo, mock_download):
        """Test that detect only returns HFF classes (abandon in header region kept, title ignored)."""
        mock_download.return_value = "/path/to/model.pt"

        mock_result = MagicMock()
        mock_boxes = MagicMock()
        mock_boxes.__len__ = lambda self: 3
        mock_boxes.xyxy = MagicMock()
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([
            [10, 10, 50, 50],  # title
            [10, 10, 50, 50],  # abandon -> header (y_center=30 in 100px)
            [10, 10, 50, 50],  # plain_text -> text-area
        ])
        mock_boxes.conf = MagicMock()
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.9, 0.9, 0.9])
        mock_boxes.cls = MagicMock()
        mock_boxes.cls.cpu.return_value.numpy.return_value.astype.return_value = np.array([0, 2, 1])
        mock_result.boxes = mock_boxes

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        detector = HFFDetector()
        detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

        assert len(detections) == 2
        class_ids = {d["class_id"] for d in detections}
        assert 0 not in class_ids  # title filtered
        assert 2 in class_ids     # abandon kept as header
        assert 1 in class_ids     # text-area kept

    @patch("hff_remover.detector._yolo.hf_hub_download")
    @patch("doclayout_yolo.YOLOv10")
    def test_detect_ignores_middle_abandon(self, mock_yolo, mock_download):
        """Test that abandon detections in the middle of the page are ignored."""
        mock_download.return_value = "/path/to/model.pt"

        mock_result = MagicMock()
        mock_boxes = MagicMock()
        mock_boxes.__len__ = lambda self: 1
        mock_boxes.xyxy = MagicMock()
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([
            [10, 40, 50, 60],  # y_center=50 -> 50% -> ignored
        ])
        mock_boxes.conf = MagicMock()
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.9])
        mock_boxes.cls = MagicMock()
        mock_boxes.cls.cpu.return_value.numpy.return_value.astype.return_value = np.array([2])
        mock_result.boxes = mock_boxes

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        detector = HFFDetector()
        detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        assert detections == []

    @patch("hff_remover.detector._yolo.hf_hub_download")
    @patch("doclayout_yolo.YOLOv10")
    def test_detect_empty_result(self, mock_yolo, mock_download):
        """Test detection with no results."""
        mock_download.return_value = "/path/to/model.pt"

        mock_result = MagicMock()
        mock_result.boxes = None

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        detector = HFFDetector()
        detections = detector.detect(np.zeros((100, 100, 3)))

        assert len(detections) == 0

    @patch("hff_remover.detector._yolo.hf_hub_download")
    @patch("doclayout_yolo.YOLOv10")
    def test_confidence_threshold(self, mock_yolo, mock_download):
        """Test that confidence threshold is passed to model."""
        mock_download.return_value = "/path/to/model.pt"

        mock_result = MagicMock()
        mock_result.boxes = None
        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        detector = HFFDetector(confidence_threshold=0.7)
        detector.detect(np.zeros((100, 100, 3)))

        call_kwargs = mock_model.call_args[1]
        assert call_kwargs["conf"] == 0.7
