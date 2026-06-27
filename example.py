"""
Example script to remove headers, footers, and footnotes from images in a directory.

Supports multiple detector backends:
- DocLayout-YOLO fine-tuned (default, TDLA)
- TDLA YOLO – Tibetan Document Layout Analysis (YOLO26-m)
- Eric-YOLO – tiled YOLO11-nano (640×640 tile-based inference)
- PP-DocLayout-L (PaddlePaddle)
- Surya Layout

Usage:
    python example.py
    
    Or modify settings in the main block.
"""

# Suppress noisy warnings (ccache from PaddlePaddle, CUDA driver mismatch on CPU-only runs)
import warnings
warnings.filterwarnings("ignore", message="No ccache found")
warnings.filterwarnings("ignore", message="CUDA initialization")

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

import inspect
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from tqdm import tqdm

from hff_remover.detector import (
    YoloDetector,
    TDLA_CONFIG,
    ERIC_YOLO_CONFIG,
    PPDocLayoutDetector,
    SuryaLayoutDetector,
    BaseHFFDetector,
)
from hff_remover.processor import (
    COCODatasetWriter,
    HFFProcessor,
    MaskedInferenceImageWriter,
)
from hff_remover.utils import load_image, find_images


def create_inference_writers(
    output_format: str,
    inference_dir: str,
    margin: int = 0,
) -> list:
    """Create the inference writer(s) for the requested output format.

    Args:
        output_format: One of ``"coco"``, ``"masked"``, or ``"both"``.
        inference_dir: Base directory for inference output.
        margin: Extra pixels around detected regions (masked writer only).

    Returns:
        List of writer instances (may be empty).
    """
    writers: list = []
    if output_format in {"coco", "both"}:
        writers.append(COCODatasetWriter(inference_dir))
    if output_format in {"masked", "both"}:
        writers.append(MaskedInferenceImageWriter(inference_dir, margin=margin))
    return writers


def detect_pixel_space(
    detector: BaseHFFDetector,
    image: Union[str, Path, np.ndarray],
) -> List[Dict[str, Any]]:
    """Run detection, returning bboxes in pixel coordinates.

    Some detectors (e.g. Surya, TMB-DLA) expose a ``normalize_bbox``
    parameter and return normalized ``[0, 1]`` coordinates by default;
    these are asked for pixel coordinates explicitly. Detectors without
    that parameter already return pixel coordinates.

    Args:
        detector: The detector to run.
        image: Path to an image file or a BGR numpy array.

    Returns:
        The list of detection dictionaries produced by the detector.
    """
    if "normalize_bbox" in inspect.signature(detector.detect).parameters:
        return detector.detect(image, normalize_bbox=False)
    return detector.detect(image)


def create_detector(
    detector_type: str = "docyolo",
    device: str = "cpu",
    confidence: float = 0.5,
) -> BaseHFFDetector:
    """Create a detector based on the specified type.

    Args:
        detector_type: Type of detector.  One of:
            ``"docyolo"`` – DocLayout-YOLO fine-tuned (4-class, imgsz 640),
            ``"tdla"`` – TDLA YOLO26-m (header/text-area/footnote/footer),
            ``"eric_yolo"`` – tiled YOLO11-nano (640×640 tiles),
            ``"paddle"`` – PP-DocLayout-L (PaddlePaddle, 23 classes),
            ``"surya"`` – Surya Layout (string-label based).
        device: Device for inference (``"cuda"`` or ``"cpu"``).
        confidence: Minimum confidence threshold.

    Returns:
        Detector instance.

    Raises:
        ValueError: If *detector_type* is not recognised.
    """
    if detector_type == "docyolo":
        print("Using DocLayout-YOLO fine-tuned detector (4-class, imgsz=640)")
        return YoloDetector(
            model_path="data/DocLayout-ft.pt",
            config=TDLA_CONFIG,
            device=device,
            confidence_threshold=confidence,
        )

    elif detector_type == "tdla":
        print("Using TDLA YOLO26 detector (header/text-area/footnote/footer)")
        return YoloDetector(
            model_path="data/TDLA-v10.pt",
            config=TDLA_CONFIG,
            device=device,
            confidence_threshold=confidence,
        )

    elif detector_type == "eric_yolo":
        print("Using Eric's tiled YOLO11-nano HFF detector")
        return YoloDetector(
            model_path="data/eric_yolo.pt",
            config=ERIC_YOLO_CONFIG,
            device=device,
            confidence_threshold=confidence,
        )

    elif detector_type == "paddle":
        print("Using PP-DocLayout-L detector (PaddlePaddle)")
        return PPDocLayoutDetector(
            model_name="PP-DocLayout-L",
            confidence_threshold=confidence,
            use_gpu=(device == "cuda"),
        )

    elif detector_type == "surya":
        print("Using Surya Layout detector")
        return SuryaLayoutDetector(
            confidence_threshold=confidence,
        )

    else:
        raise ValueError(f"Unknown detector type: {detector_type!r}")


def process_directory(
    input_dir: str,
    output_dir: str,
    detector_type: str = "docyolo",
    device: str = "cpu",
    confidence: float = 0.5,
    margin: int = 0,
    merge_margin: Optional[int] = None,
    output_format: str = "coco",
    inference_dir: Optional[str] = None,
) -> dict:
    """Process all images in a directory to remove headers, footers, and footnotes.

    Args:
        input_dir: Path to input directory containing images.
        output_dir: Path to output directory for processed images.
        detector_type: Type of detector (``"docyolo"``, ``"tdla"``,
            ``"eric_yolo"``, ``"paddle"``, ``"surya"``).
        device: Device for inference ('cuda' or 'cpu').
        confidence: Minimum confidence threshold for detections.
        margin: Extra pixels to add around detected regions.
        merge_margin: If set, merge nearby same-class boxes whose gap
            is within this many pixels.  ``None`` disables merging.
        output_format: One of ``"coco"``, ``"masked"``, or ``"both"``.
        inference_dir: Override output directory for inference data.

    Returns:
        Dictionary with processing statistics.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Scanning for images in: {input_path}")
    images = find_images(input_path, recursive=True)
    print(f"Found {len(images)} images")

    if not images:
        print("No images found!")
        return {"processed": 0, "failed": 0, "total_detections": 0}

    print(f"Loading model on {device}...")
    detector = create_detector(detector_type, device, confidence)

    inference_dir = inference_dir or output_dir

    output_format = (output_format or "").lower().strip()
    if output_format not in {"masked", "coco", "both"}:
        raise ValueError("output_format must be one of: masked, coco, both")

    processor = HFFProcessor(margin=margin)

    writers = create_inference_writers(output_format, inference_dir, margin=margin)
    print("Model loaded successfully!")

    stats = {
        "processed": 0,
        "failed": 0,
        "skipped": 0,
        "total_detections": 0,
        "failed_files": [],
    }

    print("\nProcessing images...")
    for image_path in tqdm(images, desc="Processing"):
        try:
            image = load_image(image_path)

            detections = detect_pixel_space(detector, image)

            if merge_margin is not None:
                detections = processor.merge_nearby_detections(
                    detections, margin=merge_margin,
                )

            stats["total_detections"] += len(detections)

            for writer in writers:
                try:
                    writer.write_sample(
                        image=image,
                        detections=detections,
                        image_rel_path=image_path.name,
                    )
                except Exception as e:
                    print(f"\nWarning: failed to write inference sample for {image_path}: {e}")
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
    #   "docyolo"    - DocLayout-YOLO fine-tuned (4-class, imgsz=640)
    #   "tdla"       - TDLA YOLO26-m (header/text-area/footnote/footer)
    #   "eric_yolo"  - Eric's tiled YOLO11-nano (640×640 tile-based inference)
    #   "paddle"     - PP-DocLayout-L (PaddlePaddle, higher precision, 23 classes)
    #   "surya"      - Surya Layout (string-label based layout analysis)
    detector_type = "paddle"
    
    # Device: "cpu" or "cuda" (for GPU)
    device = "cpu"
    
    # Confidence threshold (0.0 - 1.0)
    confidence = 0.3
    
    # Margin around detected regions in pixels
    margin = 0

    # Merge nearby same-class boxes (post-processing).
    # Set to an int (e.g. 20) to merge boxes within that pixel gap,
    # or None to disable merging entirely.
    merge_margin = None

    # Save inference as COCO dataset
    output_format = "coco"
    
    # ==========================================================================

    # Process directory
    stats = process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        detector_type=detector_type,
        device=device,
        confidence=confidence,
        margin=margin,
        merge_margin=merge_margin,
        output_format=output_format,
    )

    # Print summary
    print("\n" + "=" * 50)
    print("Processing Complete!")
    print(f"  Detector: {detector_type}")
    print(f"  Processed: {stats['processed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Total HFF detections: {stats['total_detections']}")
    
    if stats["failed_files"]:
        print("\nFailed files:")
        for f in stats["failed_files"]:
            print(f"  - {f}")


if __name__ == "__main__":
    input_dir = './data/benchmark_dataset/images'    # Directory containing input images
    output_dir = './data/paddle_new_benchmark'  # Directory for output images
    main(input_dir=input_dir, output_dir=output_dir)
