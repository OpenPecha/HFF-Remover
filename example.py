"""
Example script to remove headers, footers, and footnotes from images in a directory.

Supports multiple detector backends:
- DocLayout-YOLO (default)
- PP-DocLayout-L (PaddlePaddle)
- Ensemble (both detectors combined)

Usage:
    python example.py
    
    Or modify settings in the main block.
"""

# MUST be set before any paddle imports to disable oneDNN and PIR (avoids PaddlePaddle bugs)
import os
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_enable_pir_api'] = '0'  # Disable PIR (fixes ConvertPirAttribute error)
os.environ['FLAGS_enable_pir_in_executor'] = '0'
os.environ['FLAGS_pir_apply_inplace_pass'] = '0'
os.environ['DNNL_VERBOSE'] = '0'
os.environ['FLAGS_use_onednn'] = '0'
os.environ['PADDLE_USE_PIR'] = '0'

# Force CPU mode for paddle
import paddle
paddle.set_device('cpu')

import sys
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from hff_remover.detector import (
    HFFDetector,
    PPDocLayoutDetector,
    EnsembleDetector,
    BaseHFFDetector,
)
from hff_remover.processor import HFFProcessor
from hff_remover.utils import load_image, save_image, find_images


def create_detector(
    detector_type: str = "yolo",
    device: str = "cpu",
    confidence: float = 0.5,
) -> BaseHFFDetector:
    """
    Create a detector based on the specified type.

    Args:
        detector_type: Type of detector ('yolo', 'paddle', 'ensemble', 'cascade').
        device: Device for inference ('cuda' or 'cpu').
        confidence: Minimum confidence threshold.

    Returns:
        Detector instance.
    """
    if detector_type == "yolo":
        print("Using DocLayout-YOLO detector")
        return HFFDetector(
            device=device,
            confidence_threshold=confidence,
        )

    elif detector_type == "paddle":
        print("Using PP-DocLayout-L detector (PaddlePaddle)")
        use_gpu = device == "cuda"
        return PPDocLayoutDetector(
            model_name="PP-DocLayout-L",
            confidence_threshold=confidence,
            use_gpu=use_gpu,
        )

    elif detector_type == "ensemble":
        print("Using Ensemble detector (YOLO + PP-DocLayout-L, union)")
        yolo = HFFDetector(device=device, confidence_threshold=confidence)
        paddle = PPDocLayoutDetector(
            model_name="PP-DocLayout-L",
            confidence_threshold=confidence,
            use_gpu=(device == "cuda"),
        )
        return EnsembleDetector(
            detectors=[yolo, paddle],
            merge_strategy="union",
        )

    elif detector_type == "cascade":
        print("Using Cascade detector (YOLO first, PP-DocLayout-L as fallback)")
        yolo = HFFDetector(device=device, confidence_threshold=confidence)
        paddle = PPDocLayoutDetector(
            model_name="PP-DocLayout-L",
            confidence_threshold=confidence,
            use_gpu=(device == "cuda"),
        )
        return EnsembleDetector(
            detectors=[yolo, paddle],
            merge_strategy="cascade",
        )

    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


def process_directory(
    input_dir: str,
    output_dir: str,
    detector_type: str = "yolo",
    device: str = "cpu",
    confidence: float = 0.5,
    padding: int = 0,
) -> dict:
    """
    Process all images in a directory to remove headers, footers, and footnotes.

    Args:
        input_dir: Path to input directory containing images.
        output_dir: Path to output directory for processed images.
        detector_type: Type of detector ('yolo', 'paddle', 'ensemble', 'cascade').
        device: Device for inference ('cuda' or 'cpu').
        confidence: Minimum confidence threshold for detections.
        padding: Extra pixels to add around detected regions.

    Returns:
        Dictionary with processing statistics.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all images in input directory
    print(f"Scanning for images in: {input_path}")
    images = find_images(input_path, recursive=True)
    print(f"Found {len(images)} images")

    if not images:
        print("No images found!")
        return {"processed": 0, "failed": 0, "total_detections": 0}

    # Initialize detector and processor
    print(f"Loading model on {device}...")
    detector = create_detector(detector_type, device, confidence)
    processor = HFFProcessor(padding=padding)
    print("Model loaded successfully!")

    # Process statistics
    stats = {
        "processed": 0,
        "failed": 0,
        "skipped": 0,
        "total_detections": 0,
        "failed_files": [],
    }

    # Process each image with progress bar
    print(f"\nProcessing images...")
    for image_path in tqdm(images, desc="Processing"):
        try:
            # Load image
            image = load_image(image_path)

            # Detect HFF regions
            detections = detector.detect(image)
            stats["total_detections"] += len(detections)

            # Mask detected regions with white
            result_image = processor.mask_regions(image, detections)

            # Create output filename with rm_ prefix
            output_filename = f"rm_{image_path.name}"
            output_file_path = output_path / output_filename

            # Save result
            save_image(result_image, output_file_path)
            stats["processed"] += 1

        except Exception as e:
            print(f"\nError processing {image_path}: {e}")
            stats["failed"] += 1
            stats["failed_files"].append(str(image_path))

    return stats


def main(input_dir: str, output_dir: str):
    """Main entry point."""

    # Check input directory exists
    if not Path(input_dir).exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    if not Path(input_dir).is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        sys.exit(1)

    # ==========================================================================
    # CONFIGURATION - Modify these settings as needed
    # ==========================================================================
    
    # Detector type options:
    #   "yolo"     - DocLayout-YOLO (fast, good general performance)
    #   "paddle"   - PP-DocLayout-L (higher precision, 23 classes)
    #   "ensemble" - Both detectors, merge results (best recall)
    #   "cascade"  - Try YOLO first, use Paddle as fallback if no detections
    detector_type = "ensemble"
    
    # Device: "cpu" or "cuda" (for GPU)
    device = "cpu"
    
    # Confidence threshold (0.0 - 1.0)
    confidence = 0.3
    
    # Padding around detected regions in pixels
    padding = 0
    
    # ==========================================================================

    # Process directory
    stats = process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        detector_type=detector_type,
        device=device,
        confidence=confidence,
        padding=padding,
    )

    # Print summary
    print("\n" + "=" * 50)
    print("Processing Complete!")
    print(f"  Detector: {detector_type}")
    print(f"  Processed: {stats['processed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Total HFF detections: {stats['total_detections']}")
    
    if stats["failed_files"]:
        print(f"\nFailed files:")
        for f in stats["failed_files"]:
            print(f"  - {f}")


if __name__ == "__main__":
    input_dir = './data/input_images'    # Directory containing input images
    output_dir = './data/output_images'  # Directory for output images
    main(input_dir=input_dir, output_dir=output_dir)
