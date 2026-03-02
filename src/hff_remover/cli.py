"""Command-line interface for HFF Remover."""

import argparse
import logging
import sys
from pathlib import Path

from hff_remover import __version__
from hff_remover.batch import BatchProcessor, generate_report
from hff_remover.detector import HFFDetector
from hff_remover.processor import (
    HFFProcessor,
    YOLOInferenceDatasetWriter,
    MaskedInferenceImageWriter,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="hff-remover",
        description="Remove headers, footers, and footnotes from scanned book images using DocLayout-YOLO.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images in a directory
  hff-remover process /path/to/images --output /path/to/output

  # Process with GPU and custom batch size
  hff-remover process /path/to/images --output /path/to/output --device cuda --batch-size 32

  # Resume interrupted processing
  hff-remover process /path/to/images --output /path/to/output --resume

  # Process a single image
  hff-remover single input.jpg output.jpg

  # Detect without masking (just show coordinates)
  hff-remover detect input.jpg

  # Evaluate predictions against ground truth
  hff-remover evaluate --gt-dir /path/to/gt/labels --pred-dir /path/to/pred/labels
""",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process command (batch processing)
    process_parser = subparsers.add_parser(
        "process",
        help="Process all images in a directory",
    )
    process_parser.add_argument(
        "input",
        type=str,
        help="Input directory containing images",
    )
    process_parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for processed images",
    )
    process_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference (default: cuda)",
    )
    process_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of images to process at once (default: 8)",
    )
    process_parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold (default: 0.5)",
    )
    process_parser.add_argument(
        "--padding",
        type=int,
        default=0,
        help="Extra padding around detected regions in pixels (default: 0)",
    )
    process_parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Input size for model (default: 1024)",
    )
    process_parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="Output image quality 0-100 (default: 95)",
    )
    process_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )
    process_parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories",
    )
    process_parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save checkpoint every N images (default: 100)",
    )
    process_parser.add_argument(
        "--io-workers",
        type=int,
        default=4,
        help="Number of I/O worker threads (default: 4)",
    )
    process_parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to custom model weights (optional)",
    )
    process_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    inference_group = process_parser.add_mutually_exclusive_group()
    inference_group.add_argument(
        "--save-inference-yolo",
        action="store_true",
        help="Save inference results as a YOLO-style dataset under --inference-dir",
    )
    inference_group.add_argument(
        "--save-inference-masked",
        action="store_true",
        help="Save masked inference images under --inference-dir/images",
    )
    process_parser.add_argument(
        "--inference-dir",
        type=str,
        default="inference_data",
        help="Output directory for inference dataset (default: inference_data)",
    )

    # Single image command
    single_parser = subparsers.add_parser(
        "single",
        help="Process a single image",
    )
    single_parser.add_argument(
        "input",
        type=str,
        help="Input image path",
    )
    single_parser.add_argument(
        "output",
        type=str,
        help="Output image path",
    )
    single_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference (default: cuda)",
    )
    single_parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold (default: 0.5)",
    )
    single_parser.add_argument(
        "--padding",
        type=int,
        default=0,
        help="Extra padding around detected regions (default: 0)",
    )
    single_parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Input size for model (default: 1024)",
    )
    single_parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="Output image quality 0-100 (default: 95)",
    )
    single_parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to custom model weights (optional)",
    )
    single_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    inference_group = single_parser.add_mutually_exclusive_group()
    inference_group.add_argument(
        "--save-inference-yolo",
        action="store_true",
        help="Save inference results as a YOLO-style dataset under --inference-dir",
    )
    inference_group.add_argument(
        "--save-inference-masked",
        action="store_true",
        help="Save masked inference images under --inference-dir/images",
    )
    single_parser.add_argument(
        "--inference-dir",
        type=str,
        default="inference_data",
        help="Output directory for inference dataset (default: inference_data)",
    )

    # Detect command (detection only)
    detect_parser = subparsers.add_parser(
        "detect",
        help="Detect HFF regions without masking",
    )
    detect_parser.add_argument(
        "input",
        type=str,
        help="Input image path",
    )
    detect_parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file for detections (optional)",
    )
    detect_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference (default: cuda)",
    )
    detect_parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold (default: 0.5)",
    )
    detect_parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Input size for model (default: 1024)",
    )
    detect_parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Show all detected classes, not just HFF",
    )
    detect_parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to custom model weights (optional)",
    )
    detect_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    inference_group = detect_parser.add_mutually_exclusive_group()
    inference_group.add_argument(
        "--save-inference-yolo",
        action="store_true",
        help="Save this inference as a YOLO-style dataset under --inference-dir",
    )
    inference_group.add_argument(
        "--save-inference-masked",
        action="store_true",
        help="Save masked inference image under --inference-dir/images",
    )
    detect_parser.add_argument(
        "--inference-dir",
        type=str,
        default="inference_data",
        help="Output directory for inference dataset (default: inference_data)",
    )

    # Evaluate command (mAP evaluation)
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate predictions against ground truth (mAP)",
    )
    evaluate_parser.add_argument(
        "--gt-dir",
        type=str,
        required=True,
        help="Directory containing ground-truth YOLO label files (.txt)",
    )
    evaluate_parser.add_argument(
        "--pred-dir",
        type=str,
        required=True,
        help="Directory containing prediction YOLO label files (.txt)",
    )
    evaluate_parser.add_argument(
        "--class-names",
        type=str,
        nargs="+",
        default=None,
        help="Ordered class names (e.g. text footer header). "
             "Index position maps to class_id.",
    )
    evaluate_parser.add_argument(
        "--iou-threshold",
        type=float,
        nargs="+",
        default=None,
        help="IoU threshold(s) for evaluation. "
             "Defaults to COCO-style 0.50:0.05:0.95.",
    )
    evaluate_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    return parser


def cmd_process(args: argparse.Namespace) -> int:
    """Handle process command."""
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1

    logger.info(f"HFF Remover v{__version__}")
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")

    # Create components
    try:
        detector = HFFDetector(
            model_path=args.model_path,
            device=args.device,
            confidence_threshold=args.confidence,
        )
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        return 1

    processor = HFFProcessor(padding=args.padding)
    if getattr(args, "save_inference_yolo", False):
        inference_writer = YOLOInferenceDatasetWriter(args.inference_dir)
    elif getattr(args, "save_inference_masked", False):
        inference_writer = MaskedInferenceImageWriter(args.inference_dir)
    else:
        inference_writer = None

    batch_processor = BatchProcessor(
        detector=detector,
        processor=processor,
        inference_writer=inference_writer,
        batch_size=args.batch_size,
        num_io_workers=args.io_workers,
        checkpoint_interval=args.checkpoint_interval,
    )

    # Process
    try:
        stats = batch_processor.process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            recursive=not args.no_recursive,
            resume=args.resume,
            image_size=args.image_size,
            quality=args.quality,
        )

        # Generate report
        report_path = output_dir / "processing_report.json"
        generate_report(stats, report_path)

        # Print summary
        logger.info("=" * 50)
        logger.info("Processing Complete!")
        logger.info(f"  Total images: {stats.total_images}")
        logger.info(f"  Processed: {stats.processed_images}")
        logger.info(f"  Failed: {stats.failed_images}")
        logger.info(f"  Detections: {stats.total_detections}")
        logger.info(f"  Time: {stats.elapsed_time:.1f}s")
        logger.info(f"  Speed: {stats.images_per_second:.1f} images/sec")
        logger.info(f"  Report: {report_path}")

        return 0 if stats.failed_images == 0 else 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Progress saved to checkpoint.")
        return 130
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1


def cmd_single(args: argparse.Namespace) -> int:
    """Handle single image command."""
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        return 1

    logger.info(f"Processing: {input_path}")

    try:
        detector = HFFDetector(
            model_path=args.model_path,
            device=args.device,
            confidence_threshold=args.confidence,
        )
        processor = HFFProcessor(padding=args.padding)
        if getattr(args, "save_inference_yolo", False):
            inference_writer = YOLOInferenceDatasetWriter(args.inference_dir)
        elif getattr(args, "save_inference_masked", False):
            inference_writer = MaskedInferenceImageWriter(args.inference_dir)
        else:
            inference_writer = None

        batch_processor = BatchProcessor(
            detector=detector,
            processor=processor,
            inference_writer=inference_writer,
        )

        result = batch_processor.process_single(
            input_path=input_path,
            output_path=output_path,
            image_size=args.image_size,
            quality=args.quality,
        )

        logger.info(f"Detections: {len(result['detections'])}")
        for det in result['detections']:
            logger.info(f"  - {det['class_name']}: {det['confidence']:.2f} at {det['bbox']}")

        logger.info(f"Output saved: {output_path}")
        return 0

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1


def cmd_detect(args: argparse.Namespace) -> int:
    """Handle detect command."""
    import json
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    input_path = Path(args.input)

    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        return 1

    logger.info(f"Detecting in: {input_path}")

    try:
        detector = HFFDetector(
            model_path=args.model_path,
            device=args.device,
            confidence_threshold=args.confidence,
        )

        if args.all_classes:
            detections = detector.get_all_detections(
                input_path,
                image_size=args.image_size,
            )
        else:
            detections = detector.detect(
                input_path,
                image_size=args.image_size,
            )

        # Print detections
        logger.info(f"Found {len(detections)} detections:")
        for det in detections:
            bbox_str = ", ".join(f"{x:.1f}" for x in det['bbox'])
            logger.info(f"  - {det['class_name']}: {det['confidence']:.2f} at [{bbox_str}]")

        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump({
                    "input": str(input_path),
                    "detections": detections,
                }, f, indent=2)
            logger.info(f"Detections saved to: {output_path}")

        # Optionally save inference (either YOLO dataset or masked image)
        if getattr(args, "save_inference_yolo", False) or getattr(args, "save_inference_masked", False):
            from hff_remover.utils import load_image

            image = load_image(input_path)

            if getattr(args, "save_inference_yolo", False):
                writer = YOLOInferenceDatasetWriter(args.inference_dir)
                writer.write_sample(
                    image=image,
                    detections=detections,
                    image_rel_path=input_path.name,
                )
            else:
                # masked
                processor = HFFProcessor(padding=0)
                masked = processor.mask_regions(image, detections)
                writer = MaskedInferenceImageWriter(args.inference_dir)
                writer.write_sample(
                    image=masked,
                    detections=detections,
                    image_rel_path=input_path.name,
                )

            logger.info(f"Inference saved under: {Path(args.inference_dir).resolve()}")

        return 0

    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Handle evaluate command."""
    from hff_remover.evaluate import evaluate, print_report

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)

    # Build class_names mapping from ordered list
    class_names = None
    if args.class_names:
        class_names = {i: name for i, name in enumerate(args.class_names)}

    logger.info(f"Evaluating predictions: {pred_dir}")
    logger.info(f"Against ground truth:   {gt_dir}")

    try:
        result = evaluate(
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            class_names=class_names,
            iou_thresholds=args.iou_threshold,
        )
        print_report(result, class_names=class_names)
        return 0
    except (NotADirectoryError, FileNotFoundError) as exc:
        logger.error(str(exc))
        return 1


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "process":
        return cmd_process(args)
    elif args.command == "single":
        return cmd_single(args)
    elif args.command == "detect":
        return cmd_detect(args)
    elif args.command == "evaluate":
        return cmd_evaluate(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
