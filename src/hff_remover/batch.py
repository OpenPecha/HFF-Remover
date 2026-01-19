"""Batch processing module for large-scale HFF removal."""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
from queue import Queue
from threading import Thread

from tqdm import tqdm

from hff_remover.detector import HFFDetector
from hff_remover.processor import HFFProcessor
from hff_remover.utils import (
    find_images,
    load_image,
    save_image,
    get_output_path,
)


logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Statistics for batch processing."""

    total_images: int = 0
    processed_images: int = 0
    failed_images: int = 0
    skipped_images: int = 0
    total_detections: int = 0
    headers_detected: int = 0
    footers_detected: int = 0
    footnotes_detected: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    failed_files: List[str] = field(default_factory=list)

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def images_per_second(self) -> float:
        """Get processing speed."""
        elapsed = self.elapsed_time
        if elapsed <= 0:
            return 0.0
        return self.processed_images / elapsed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["elapsed_time"] = self.elapsed_time
        result["images_per_second"] = self.images_per_second
        return result


@dataclass
class CheckpointData:
    """Checkpoint data for resumable processing."""

    processed_files: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    last_updated: str = ""

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CheckpointData":
        """Load checkpoint from file."""
        path = Path(path)
        if not path.exists():
            return cls()

        with open(path, "r") as f:
            data = json.load(f)

        return cls(
            processed_files=data.get("processed_files", []),
            stats=data.get("stats", {}),
            last_updated=data.get("last_updated", ""),
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save checkpoint to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.last_updated = datetime.now().isoformat()

        with open(path, "w") as f:
            json.dump({
                "processed_files": self.processed_files,
                "stats": self.stats,
                "last_updated": self.last_updated,
            }, f, indent=2)


class BatchProcessor:
    """Batch processor for large-scale HFF removal."""

    def __init__(
        self,
        detector: Optional[HFFDetector] = None,
        processor: Optional[HFFProcessor] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        padding: int = 0,
        batch_size: int = 8,
        num_io_workers: int = 4,
        checkpoint_interval: int = 100,
    ):
        """
        Initialize batch processor.

        Args:
            detector: HFF detector instance. Created if not provided.
            processor: HFF processor instance. Created if not provided.
            device: Device for inference ('cuda' or 'cpu').
            confidence_threshold: Minimum confidence for detections.
            padding: Padding around detected regions.
            batch_size: Number of images to process at once.
            num_io_workers: Number of threads for I/O operations.
            checkpoint_interval: Save checkpoint every N images.
        """
        self.detector = detector or HFFDetector(
            device=device,
            confidence_threshold=confidence_threshold,
        )
        self.processor = processor or HFFProcessor(padding=padding)
        self.batch_size = batch_size
        self.num_io_workers = num_io_workers
        self.checkpoint_interval = checkpoint_interval

        self._stop_requested = False

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        recursive: bool = True,
        resume: bool = False,
        checkpoint_path: Optional[Union[str, Path]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        image_size: int = 1024,
        quality: int = 95,
    ) -> ProcessingStats:
        """
        Process all images in a directory.

        Args:
            input_dir: Input directory containing images.
            output_dir: Output directory for processed images.
            recursive: Whether to search subdirectories.
            resume: Whether to resume from checkpoint.
            checkpoint_path: Path for checkpoint file.
            progress_callback: Optional callback(processed, total).
            image_size: Input size for the model.
            quality: Output image quality.

        Returns:
            Processing statistics.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # Find all images
        logger.info(f"Scanning for images in {input_dir}...")
        all_images = find_images(input_dir, recursive=recursive)
        logger.info(f"Found {len(all_images)} images")

        if not all_images:
            logger.warning("No images found to process")
            return ProcessingStats()

        # Setup checkpoint
        if checkpoint_path is None:
            checkpoint_path = output_dir / ".hff_checkpoint.json"
        checkpoint_path = Path(checkpoint_path)

        checkpoint = CheckpointData()
        if resume and checkpoint_path.exists():
            checkpoint = CheckpointData.load(checkpoint_path)
            logger.info(f"Resuming from checkpoint: {len(checkpoint.processed_files)} already processed")

        # Filter already processed
        processed_set = set(checkpoint.processed_files)
        images_to_process = [
            img for img in all_images
            if str(img) not in processed_set
        ]

        logger.info(f"Images to process: {len(images_to_process)}")

        # Initialize stats
        stats = ProcessingStats(
            total_images=len(all_images),
            processed_images=len(processed_set),
            skipped_images=len(processed_set),
        )
        stats.start_time = time.time()

        # Process in batches with progress bar
        self._stop_requested = False

        with tqdm(total=len(images_to_process), desc="Processing images") as pbar:
            for i in range(0, len(images_to_process), self.batch_size):
                if self._stop_requested:
                    logger.info("Stop requested, saving checkpoint...")
                    break

                batch_paths = images_to_process[i:i + self.batch_size]

                try:
                    batch_stats = self._process_batch(
                        batch_paths,
                        input_dir,
                        output_dir,
                        image_size,
                        quality,
                    )

                    # Update stats
                    stats.processed_images += batch_stats["processed"]
                    stats.failed_images += batch_stats["failed"]
                    stats.total_detections += batch_stats["detections"]
                    stats.headers_detected += batch_stats["headers"]
                    stats.footers_detected += batch_stats["footers"]
                    stats.footnotes_detected += batch_stats["footnotes"]
                    stats.failed_files.extend(batch_stats["failed_files"])

                    # Update checkpoint
                    checkpoint.processed_files.extend(
                        str(p) for p in batch_paths
                        if str(p) not in batch_stats["failed_files"]
                    )

                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    stats.failed_images += len(batch_paths)
                    stats.failed_files.extend(str(p) for p in batch_paths)

                # Save checkpoint periodically
                if stats.processed_images % self.checkpoint_interval < self.batch_size:
                    checkpoint.stats = stats.to_dict()
                    checkpoint.save(checkpoint_path)

                pbar.update(len(batch_paths))

                if progress_callback:
                    progress_callback(stats.processed_images, stats.total_images)

        # Final checkpoint
        stats.end_time = time.time()
        checkpoint.stats = stats.to_dict()
        checkpoint.save(checkpoint_path)

        return stats

    def _process_batch(
        self,
        image_paths: List[Path],
        input_dir: Path,
        output_dir: Path,
        image_size: int,
        quality: int,
    ) -> Dict[str, Any]:
        """Process a batch of images."""
        batch_stats = {
            "processed": 0,
            "failed": 0,
            "detections": 0,
            "headers": 0,
            "footers": 0,
            "footnotes": 0,
            "failed_files": [],
        }

        # Load images in parallel
        images = []
        valid_paths = []

        with ThreadPoolExecutor(max_workers=self.num_io_workers) as executor:
            load_results = list(executor.map(
                lambda p: self._safe_load(p),
                image_paths
            ))

        for path, result in zip(image_paths, load_results):
            if result is not None:
                images.append(result)
                valid_paths.append(path)
            else:
                batch_stats["failed"] += 1
                batch_stats["failed_files"].append(str(path))

        if not images:
            return batch_stats

        # Detect HFF regions
        try:
            all_detections = self.detector.detect_batch(
                images,
                image_size=image_size,
                batch_size=len(images),
            )
        except Exception as e:
            logger.error(f"Detection error: {e}")
            batch_stats["failed"] += len(images)
            batch_stats["failed_files"].extend(str(p) for p in valid_paths)
            return batch_stats

        # Process and save images
        save_tasks = []

        for path, image, detections in zip(valid_paths, images, all_detections):
            try:
                # Count detection types
                for det in detections:
                    batch_stats["detections"] += 1
                    class_name = det.get("class_name", "")
                    if class_name == "abandon":
                        batch_stats["headers"] += 1  # abandon includes headers/footers
                    elif class_name == "table_footnote":
                        batch_stats["footnotes"] += 1

                # Apply masking
                processed = self.processor.mask_regions(image, detections)

                # Queue for saving
                output_path = get_output_path(path, input_dir, output_dir)
                save_tasks.append((processed, output_path, quality))
                batch_stats["processed"] += 1

            except Exception as e:
                logger.error(f"Processing error for {path}: {e}")
                batch_stats["failed"] += 1
                batch_stats["failed_files"].append(str(path))

        # Save images in parallel
        with ThreadPoolExecutor(max_workers=self.num_io_workers) as executor:
            executor.map(
                lambda args: self._safe_save(*args),
                save_tasks
            )

        return batch_stats

    def _safe_load(self, path: Path) -> Optional[Any]:
        """Safely load an image."""
        try:
            return load_image(path)
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return None

    def _safe_save(self, image, path: Path, quality: int) -> bool:
        """Safely save an image."""
        try:
            save_image(image, path, quality=quality)
            return True
        except Exception as e:
            logger.error(f"Failed to save {path}: {e}")
            return False

    def stop(self) -> None:
        """Request stop of current processing."""
        self._stop_requested = True

    def process_single(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        image_size: int = 1024,
        quality: int = 95,
    ) -> Dict[str, Any]:
        """
        Process a single image.

        Args:
            input_path: Path to input image.
            output_path: Path for output. If None, returns image without saving.
            image_size: Input size for the model.
            quality: Output image quality.

        Returns:
            Dictionary with 'image', 'detections', and 'saved' keys.
        """
        input_path = Path(input_path)

        # Load image
        image = load_image(input_path)

        # Detect HFF regions
        detections = self.detector.detect(image, image_size=image_size)

        # Apply masking
        processed = self.processor.mask_regions(image, detections)

        result = {
            "image": processed,
            "detections": detections,
            "saved": False,
        }

        # Save if output path provided
        if output_path is not None:
            save_image(processed, output_path, quality=quality)
            result["saved"] = True
            result["output_path"] = str(output_path)

        return result


def generate_report(
    stats: ProcessingStats,
    output_path: Union[str, Path],
) -> None:
    """
    Generate a processing report.

    Args:
        stats: Processing statistics.
        output_path: Path for the report JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "summary": {
            "total_images": stats.total_images,
            "processed": stats.processed_images,
            "failed": stats.failed_images,
            "skipped": stats.skipped_images,
            "success_rate": (
                stats.processed_images / stats.total_images * 100
                if stats.total_images > 0 else 0
            ),
        },
        "detections": {
            "total": stats.total_detections,
            "headers_and_footers": stats.headers_detected,
            "footnotes": stats.footnotes_detected,
        },
        "performance": {
            "elapsed_seconds": stats.elapsed_time,
            "images_per_second": stats.images_per_second,
        },
        "failed_files": stats.failed_files,
        "generated_at": datetime.now().isoformat(),
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Report saved to {output_path}")
