"""Standalone script to run Surya layout detection on images, PDFs, or folders."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from PIL import Image

from surya.inference import SuryaInferenceManager
from surya.input.load import load_from_file, load_from_folder
from surya.layout import LayoutPredictor
from surya.layout.schema import LayoutResult
from surya.debug.draw import draw_polys_on_image
from surya.settings import settings


def parse_page_range(range_str: str) -> List[int]:
    """Parse a page-range string like '0,5-10,20' into a sorted list of ints."""
    pages: List[int] = []
    for part in range_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            pages.extend(range(int(start), int(end) + 1))
        else:
            pages.append(int(part))
    return sorted(set(pages))


def load_inputs(
    input_path: str, page_range: Optional[List[int]] = None
) -> tuple[List[Image.Image], List[str]]:
    """Load images and names from a file or folder."""
    if os.path.isdir(input_path):
        return load_from_folder(input_path, page_range)
    return load_from_file(input_path, page_range)


def print_summary(
    results: List[LayoutResult], names: List[str]
) -> None:
    """Print a human-readable summary of layout predictions."""
    for idx, (result, name) in enumerate(zip(results, names)):
        page_label = f"[{name} / page {idx + 1}]"
        if result.error:
            print(f"{page_label}  ERROR — layout prediction failed")
            continue

        box_count = len(result.bboxes)
        labels = [b.label for b in result.bboxes]
        label_counts = defaultdict(int)
        for lbl in labels:
            label_counts[lbl] += 1
        summary_parts = [f"{lbl}: {cnt}" for lbl, cnt in sorted(label_counts.items())]

        print(f"{page_label}  {box_count} block(s)  —  {', '.join(summary_parts)}")


def save_results_json(
    results: List[LayoutResult],
    names: List[str],
    output_path: Path,
) -> Path:
    """Serialize predictions to results.json grouped by input name."""
    predictions_by_page: dict[str, list] = defaultdict(list)
    for pred, name in zip(results, names):
        out_pred = pred.model_dump()
        out_pred["page"] = len(predictions_by_page[name]) + 1
        predictions_by_page[name].append(out_pred)

    json_path = output_path / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(predictions_by_page, f, ensure_ascii=False, indent=2)
    return json_path


def save_overlay_images(
    results: List[LayoutResult],
    images: List[Image.Image],
    names: List[str],
    output_path: Path,
) -> None:
    """Draw layout bounding boxes on images and save as PNGs."""
    for idx, (image, result, name) in enumerate(zip(images, results, names)):
        if result.error or not result.bboxes:
            continue
        polygons = [box.polygon for box in result.bboxes]
        labels = [f"{box.label}-{box.position}" for box in result.bboxes]
        overlay = draw_polys_on_image(polygons, copy.deepcopy(image), labels=labels)
        overlay.save(output_path / f"{name}_{idx}_layout.png")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run Surya layout detection on images, PDFs, or folders.",
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to an image, PDF, or folder of documents.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(settings.RESULT_DIR, "surya"),
        help="Directory to save results (default: results/surya).",
    )
    parser.add_argument(
        "--page-range",
        type=str,
        default=None,
        help="Page range for PDFs, e.g. '0,5-10,20'.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save annotated images with layout bounding boxes.",
    )
    parser.add_argument(
        "--keep-server",
        action="store_true",
        help="Keep the inference server running after exit for reuse.",
    )
    return parser


def main() -> None:
    """Entry point for layout detection."""
    parser = build_parser()
    args = parser.parse_args()

    input_path = args.input_path
    if not os.path.exists(input_path):
        print(f"Error: path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.keep_server:
        settings.SURYA_INFERENCE_KEEP_ALIVE = True

    page_range = parse_page_range(args.page_range) if args.page_range else None

    print(f"Loading input from: {input_path}")
    images, names = load_inputs(input_path, page_range)
    print(f"Loaded {len(images)} page(s)")

    folder_name = (
        os.path.basename(input_path)
        if os.path.isdir(input_path)
        else os.path.basename(input_path).split(".")[0]
    )
    output_path = Path(os.path.abspath(os.path.join(args.output_dir, folder_name)))
    output_path.mkdir(parents=True, exist_ok=True)

    manager = SuryaInferenceManager()
    layout_predictor = LayoutPredictor(manager)

    print("Running layout detection...")
    start = time.time()
    results = layout_predictor(images)
    elapsed = time.time() - start
    print(f"Layout detection completed in {elapsed:.2f}s")

    print()
    print_summary(results, names)

    json_path = save_results_json(results, names, output_path)
    print(f"\nResults saved to: {json_path}")

    if args.save_images:
        save_overlay_images(results, images, names, output_path)
        print(f"Overlay images saved to: {output_path}")


if __name__ == "__main__":
    main()
