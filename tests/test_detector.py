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

    @patch("hff_remover.detector.hf_hub_download")
    @patch("hff_remover.detector.YOLOv10")
    def test_init_downloads_model(self, mock_yolo, mock_download):
        """Test that model is downloaded if path not provided."""
        mock_download.return_value = "/path/to/model.pt"

        detector = HFFDetector()

        mock_download.assert_called_once()
        mock_yolo.assert_called_once_with("/path/to/model.pt")

    @patch("hff_remover.detector.YOLOv10")
    def test_init_with_custom_path(self, mock_yolo):
        """Test initialization with custom model path."""
        detector = HFFDetector(model_path="/custom/model.pt")

        mock_yolo.assert_called_once_with("/custom/model.pt")
        assert detector.model_path == "/custom/model.pt"

    @patch("hff_remover.detector.hf_hub_download")
    @patch("hff_remover.detector.YOLOv10")
    def test_detect_filters_hff_classes(self, mock_yolo, mock_download):
        """Test that detect only returns HFF classes."""
        mock_download.return_value = "/path/to/model.pt"

        # Create mock detection results
        mock_boxes = MagicMock()
        mock_boxes.cls = MagicMock()
        mock_boxes.cls.__len__ = lambda self: 3
        mock_boxes.cls.__getitem__ = lambda self, i: MagicMock(
            item=lambda: [0, 2, 1][i]  # title, abandon, plain_text
        )
        mock_boxes.xyxy = MagicMock()
        mock_boxes.xyxy.__getitem__ = lambda self, i: MagicMock(
            cpu=lambda: MagicMock(numpy=lambda: MagicMock(tolist=lambda: [10, 10, 50, 50]))
        )
        mock_boxes.conf = MagicMock()
        mock_boxes.conf.__getitem__ = lambda self, i: MagicMock(item=lambda: 0.9)

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        detector = HFFDetector()
        detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

        # Should only return the header/footer/table_footnote classes
        assert len(detections) == 1
        assert detections[0]["class_id"] == 2
        assert detections[0]["class_name"] == "header"

    @patch("hff_remover.detector.hf_hub_download")
    @patch("hff_remover.detector.YOLOv10")
    def test_detect_ignores_middle_abandon(self, mock_yolo, mock_download):
        """Test that abandon detections in the middle of the page are ignored."""
        mock_download.return_value = "/path/to/model.pt"

        # bbox with y-center at 50% of page height -> should be ignored
        mock_boxes = MagicMock()
        mock_boxes.__len__ = lambda self: 1
        mock_boxes.cls = [MagicMock(item=lambda: 2)]
        mock_boxes.xyxy = [
            MagicMock(cpu=lambda: MagicMock(numpy=lambda: MagicMock(tolist=lambda: [10, 40, 50, 60])))
        ]
        mock_boxes.conf = [MagicMock(item=lambda: 0.9)]

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        detector = HFFDetector()
        detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        assert detections == []

    @patch("hff_remover.detector.hf_hub_download")
    @patch("hff_remover.detector.YOLOv10")
    def test_get_all_detections_returns_all(self, mock_yolo, mock_download):
        """Test that get_all_detections returns all classes."""
        mock_download.return_value = "/path/to/model.pt"

        # Create mock detection results with multiple classes
        mock_boxes = MagicMock()
        mock_boxes.__len__ = lambda self: 2
        mock_boxes.cls = [MagicMock(item=lambda: 0), MagicMock(item=lambda: 2)]
        mock_boxes.xyxy = [
            MagicMock(cpu=lambda: MagicMock(numpy=lambda: MagicMock(tolist=lambda: [10, 10, 50, 50]))),
            MagicMock(cpu=lambda: MagicMock(numpy=lambda: MagicMock(tolist=lambda: [60, 60, 90, 90]))),
        ]
        mock_boxes.conf = [MagicMock(item=lambda: 0.9), MagicMock(item=lambda: 0.8)]

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        detector = HFFDetector()

        # Mock the iteration over boxes
        with patch.object(detector, 'get_all_detections') as mock_get_all:
            mock_get_all.return_value = [
                {"class_id": 0, "class_name": "title", "bbox": [10, 10, 50, 50], "confidence": 0.9},
                {"class_id": 2, "class_name": "abandon", "bbox": [60, 60, 90, 90], "confidence": 0.8},
            ]

            detections = detector.get_all_detections(np.zeros((100, 100, 3)))

            assert len(detections) == 2

    @patch("hff_remover.detector.hf_hub_download")
    @patch("hff_remover.detector.YOLOv10")
    def test_detect_batch(self, mock_yolo, mock_download):
        """Test batch detection."""
        mock_download.return_value = "/path/to/model.pt"

        # Create mock results for batch
        mock_boxes = MagicMock()
        mock_boxes.__len__ = lambda self: 1
        mock_boxes.cls = [MagicMock(item=lambda: 2)]
        mock_boxes.xyxy = [
            MagicMock(cpu=lambda: MagicMock(numpy=lambda: MagicMock(tolist=lambda: [10, 10, 50, 50])))
        ]
        mock_boxes.conf = [MagicMock(item=lambda: 0.9)]

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result, mock_result]
        mock_yolo.return_value = mock_model

        detector = HFFDetector()
        images = [np.zeros((100, 100, 3)), np.zeros((100, 100, 3))]

        results = detector.detect_batch(images, batch_size=2)

        assert len(results) == 2

    @patch("hff_remover.detector.hf_hub_download")
    @patch("hff_remover.detector.YOLOv10")
    def test_detect_empty_result(self, mock_yolo, mock_download):
        """Test detection with no results."""
        mock_download.return_value = "/path/to/model.pt"

        mock_result = MagicMock()
        mock_result.boxes = None

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        detector = HFFDetector()
        detections = detector.detect(np.zeros((100, 100, 3)))

        assert len(detections) == 0

    @patch("hff_remover.detector.hf_hub_download")
    @patch("hff_remover.detector.YOLOv10")
    def test_confidence_threshold(self, mock_yolo, mock_download):
        """Test that confidence threshold is passed to model."""
        mock_download.return_value = "/path/to/model.pt"
        mock_model = MagicMock()
        mock_model.predict.return_value = [MagicMock(boxes=None)]
        mock_yolo.return_value = mock_model

        detector = HFFDetector(confidence_threshold=0.7)
        detector.detect(np.zeros((100, 100, 3)))

        # Check that predict was called with the confidence threshold
        call_kwargs = mock_model.predict.call_args[1]
        assert call_kwargs["conf"] == 0.7
