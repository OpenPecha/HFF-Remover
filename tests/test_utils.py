"""Tests for the utils module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from hff_remover.utils import (
    is_supported_image,
    find_images,
    load_image,
    save_image,
    get_output_path,
    get_image_info,
    resize_image,
    scale_bboxes,
)


class TestSupportedFormats:
    """Tests for image format support."""

    def test_supported_formats(self):
        """Test supported image formats."""
        assert is_supported_image("test.jpg")
        assert is_supported_image("test.jpeg")
        assert is_supported_image("test.png")
        assert is_supported_image("test.tiff")
        assert is_supported_image("test.tif")
        assert is_supported_image("test.bmp")
        assert is_supported_image("test.webp")

    def test_unsupported_formats(self):
        """Test unsupported formats."""
        assert not is_supported_image("test.txt")
        assert not is_supported_image("test.pdf")
        assert not is_supported_image("test.doc")

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert is_supported_image("test.JPG")
        assert is_supported_image("test.PNG")
        assert is_supported_image("test.TIFF")


class TestFindImages:
    """Tests for find_images function."""

    def test_find_images_empty_dir(self):
        """Test finding images in empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            images = find_images(tmpdir)
            assert len(images) == 0

    def test_find_images_with_images(self):
        """Test finding images in directory with images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images
            (Path(tmpdir) / "test1.jpg").touch()
            (Path(tmpdir) / "test2.png").touch()
            (Path(tmpdir) / "test3.txt").touch()

            images = find_images(tmpdir)
            assert len(images) == 2

    def test_find_images_recursive(self):
        """Test recursive image search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()

            (Path(tmpdir) / "test1.jpg").touch()
            (subdir / "test2.jpg").touch()

            # Recursive
            images = find_images(tmpdir, recursive=True)
            assert len(images) == 2

            # Non-recursive
            images = find_images(tmpdir, recursive=False)
            assert len(images) == 1


class TestLoadSaveImage:
    """Tests for image loading and saving."""

    def test_load_save_roundtrip_png(self):
        """Test PNG load/save roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"

            # Create test image
            original = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            save_image(original, path)

            # Load and compare
            loaded = load_image(path)
            assert loaded.shape == original.shape
            # PNG is lossless
            assert np.array_equal(loaded, original)

    def test_load_save_roundtrip_jpeg(self):
        """Test JPEG load/save roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jpg"

            # Create test image
            original = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            save_image(original, path, quality=100)

            # Load and compare (JPEG is lossy)
            loaded = load_image(path)
            assert loaded.shape == original.shape

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/path/image.jpg")

    def test_save_creates_directory(self):
        """Test that save creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "another" / "test.png"

            image = np.zeros((10, 10, 3), dtype=np.uint8)
            save_image(image, path)

            assert path.exists()


class TestGetOutputPath:
    """Tests for output path generation."""

    def test_basic_output_path(self):
        """Test basic output path generation."""
        result = get_output_path(
            input_path="/input/subdir/image.jpg",
            input_dir="/input",
            output_dir="/output",
        )
        assert result == Path("/output/subdir/image.jpg")

    def test_preserve_format(self):
        """Test format preservation."""
        result = get_output_path(
            input_path="/input/image.tiff",
            input_dir="/input",
            output_dir="/output",
            preserve_format=True,
        )
        assert result.suffix == ".tiff"

    def test_change_format(self):
        """Test format change."""
        result = get_output_path(
            input_path="/input/image.tiff",
            input_dir="/input",
            output_dir="/output",
            preserve_format=False,
            output_format=".png",
        )
        assert result.suffix == ".png"


class TestResizeImage:
    """Tests for image resizing."""

    def test_resize_max_size(self):
        """Test resizing with max size constraint."""
        image = np.zeros((200, 100, 3), dtype=np.uint8)
        resized, scale = resize_image(image, max_size=100)

        assert resized.shape[0] == 100
        assert resized.shape[1] == 50
        assert scale == 0.5

    def test_resize_no_change(self):
        """Test that small images are not resized."""
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        resized, scale = resize_image(image, max_size=100)

        assert resized.shape == image.shape
        assert scale == 1.0


class TestScaleBboxes:
    """Tests for bounding box scaling."""

    def test_scale_bboxes(self):
        """Test bounding box scaling."""
        bboxes = [[10, 20, 30, 40], [100, 200, 300, 400]]
        scaled = scale_bboxes(bboxes, 0.5)

        assert scaled[0] == [20, 40, 60, 80]
        assert scaled[1] == [200, 400, 600, 800]

    def test_scale_bboxes_empty(self):
        """Test scaling empty list."""
        assert scale_bboxes([], 0.5) == []
