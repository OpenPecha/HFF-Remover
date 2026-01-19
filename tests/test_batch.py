"""Tests for the batch processing module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hff_remover.batch import (
    ProcessingStats,
    CheckpointData,
    BatchProcessor,
    generate_report,
)


class TestProcessingStats:
    """Tests for ProcessingStats dataclass."""

    def test_elapsed_time_not_started(self):
        """Test elapsed time when not started."""
        stats = ProcessingStats()
        assert stats.elapsed_time == 0.0

    def test_elapsed_time_running(self):
        """Test elapsed time while running."""
        import time
        stats = ProcessingStats()
        stats.start_time = time.time() - 10  # Started 10 seconds ago

        assert 9 < stats.elapsed_time < 11

    def test_elapsed_time_completed(self):
        """Test elapsed time when completed."""
        stats = ProcessingStats()
        stats.start_time = 100.0
        stats.end_time = 110.0

        assert stats.elapsed_time == 10.0

    def test_images_per_second(self):
        """Test images per second calculation."""
        stats = ProcessingStats()
        stats.start_time = 0.0
        stats.end_time = 10.0
        stats.processed_images = 100

        assert stats.images_per_second == 10.0

    def test_images_per_second_zero_time(self):
        """Test images per second with zero elapsed time."""
        stats = ProcessingStats()
        assert stats.images_per_second == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = ProcessingStats(
            total_images=100,
            processed_images=50,
            failed_images=5,
        )

        result = stats.to_dict()

        assert result["total_images"] == 100
        assert result["processed_images"] == 50
        assert result["failed_images"] == 5
        assert "elapsed_time" in result
        assert "images_per_second" in result


class TestCheckpointData:
    """Tests for CheckpointData."""

    def test_save_and_load(self):
        """Test checkpoint save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.json"

            checkpoint = CheckpointData(
                processed_files=["file1.jpg", "file2.jpg"],
                stats={"total": 100},
            )
            checkpoint.save(path)

            loaded = CheckpointData.load(path)

            assert loaded.processed_files == ["file1.jpg", "file2.jpg"]
            assert loaded.stats == {"total": 100}
            assert loaded.last_updated != ""

    def test_load_nonexistent(self):
        """Test loading nonexistent checkpoint returns empty."""
        checkpoint = CheckpointData.load("/nonexistent/path.json")

        assert checkpoint.processed_files == []
        assert checkpoint.stats == {}

    def test_save_creates_directory(self):
        """Test that save creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "checkpoint.json"

            checkpoint = CheckpointData()
            checkpoint.save(path)

            assert path.exists()


class TestBatchProcessor:
    """Tests for BatchProcessor."""

    @patch("hff_remover.batch.HFFDetector")
    @patch("hff_remover.batch.HFFProcessor")
    def test_init_creates_components(self, mock_processor, mock_detector):
        """Test that init creates detector and processor if not provided."""
        processor = BatchProcessor()

        mock_detector.assert_called_once()
        mock_processor.assert_called_once()

    def test_init_uses_provided_components(self):
        """Test that init uses provided components."""
        detector = MagicMock()
        processor = MagicMock()

        batch = BatchProcessor(detector=detector, processor=processor)

        assert batch.detector is detector
        assert batch.processor is processor

    @patch("hff_remover.batch.HFFDetector")
    @patch("hff_remover.batch.HFFProcessor")
    def test_process_single(self, mock_processor_class, mock_detector_class):
        """Test processing a single image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image
            input_path = Path(tmpdir) / "input.png"
            output_path = Path(tmpdir) / "output.png"

            # Create actual test image
            import cv2
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(input_path), test_image)

            # Setup mocks
            mock_detector = MagicMock()
            mock_detector.detect.return_value = [
                {"bbox": [10, 10, 50, 30], "class_id": 2, "class_name": "abandon", "confidence": 0.9}
            ]
            mock_detector_class.return_value = mock_detector

            mock_processor = MagicMock()
            mock_processor.mask_regions.return_value = test_image
            mock_processor_class.return_value = mock_processor

            batch = BatchProcessor()
            result = batch.process_single(input_path, output_path)

            assert "detections" in result
            assert result["saved"] is True
            assert output_path.exists()


class TestGenerateReport:
    """Tests for report generation."""

    def test_generate_report(self):
        """Test report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"

            stats = ProcessingStats(
                total_images=100,
                processed_images=95,
                failed_images=5,
                total_detections=200,
                headers_detected=150,
                footnotes_detected=50,
            )
            stats.start_time = 0.0
            stats.end_time = 10.0

            generate_report(stats, path)

            assert path.exists()

            with open(path) as f:
                report = json.load(f)

            assert report["summary"]["total_images"] == 100
            assert report["summary"]["processed"] == 95
            assert report["summary"]["failed"] == 5
            assert report["detections"]["total"] == 200
            assert report["performance"]["elapsed_seconds"] == 10.0
