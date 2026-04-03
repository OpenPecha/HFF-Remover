"""
Example script to evaluate layout-detection predictions against ground-truth labels.

Uses the COCO-format mAP evaluation provided by ``hff_remover.evaluate``.
Writes a detailed ``report.txt`` with all abbreviations spelled out.

Usage:
    python example_evaluate.py

    Or modify the paths / settings in the ``if __name__`` block.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

from hff_remover.evaluate import evaluate, EvaluationResult

# Class-name mapping shared by both the benchmark and inference datasets.
CLASS_NAMES: Dict[int, str] = {
    0: "header",
    1: "text-area",
    2: "footnote",
    3: "footer",
}


def _build_report(
    result: EvaluationResult,
    gt_dir: Path,
    pred_dir: Path,
    class_names: Optional[Dict[int, str]] = None,
) -> str:
    """Build a human-readable evaluation report with full-form labels.

    Args:
        result: The ``EvaluationResult`` to format.
        gt_dir: Ground-truth directory (shown in the header).
        pred_dir: Prediction directory (shown in the header).
        class_names: Optional mapping ``{class_id: name}`` for display.

    Returns:
        The complete report as a string.
    """
    lines: List[str] = []
    w = lines.append  # shorthand

    w("=" * 80)
    w("LAYOUT DETECTION EVALUATION REPORT")
    w("=" * 80)
    w("")
    w(f"Ground-Truth Directory : {gt_dir}")
    w(f"Prediction Directory   : {pred_dir}")
    w(f"Total Ground-Truth Boxes  : {result.total_gt}")
    w(f"Total Prediction Boxes    : {result.total_pred}")
    w("")

    # ------------------------------------------------------------------
    # Overall mAP
    # ------------------------------------------------------------------
    w("-" * 80)
    w("OVERALL METRICS")
    w("-" * 80)
    w(f"  Mean Average Precision @ Intersection-over-Union 0.50             : {result.map_50:.4f}")
    w(f"  Mean Average Precision @ Intersection-over-Union 0.50:0.95 (COCO) : {result.map_50_95:.4f}")
    w("")

    # ------------------------------------------------------------------
    # Per-class table at IoU = 0.50
    # ------------------------------------------------------------------
    w("-" * 80)
    w("PER-CLASS METRICS  (Intersection-over-Union threshold = 0.50)")
    w("-" * 80)

    header = (
        f"{'Class':>12s}"
        f"  {'Average Precision':>18s}"
        f"  {'Precision':>10s}"
        f"  {'Recall':>10s}"
        f"  {'True Positives':>15s}"
        f"  {'False Positives':>15s}"
        f"  {'False Negatives':>15s}"
        f"  {'Ground Truth':>13s}"
        f"  {'Predictions':>12s}"
    )
    sep = "-" * len(header)
    w(header)
    w(sep)

    for cls_id, metrics in sorted(result.class_metrics_50.items()):
        name = (class_names or {}).get(cls_id, str(cls_id))
        w(
            f"{name:>12s}"
            f"  {metrics.ap:>18.4f}"
            f"  {metrics.precision:>10.4f}"
            f"  {metrics.recall:>10.4f}"
            f"  {metrics.tp:>15d}"
            f"  {metrics.fp:>15d}"
            f"  {metrics.fn:>15d}"
            f"  {metrics.num_gt:>13d}"
            f"  {metrics.num_pred:>12d}"
        )

    w(sep)
    w("")

    # ------------------------------------------------------------------
    # Per-IoU threshold breakdown
    # ------------------------------------------------------------------
    w("-" * 80)
    w("MEAN AVERAGE PRECISION AT EACH INTERSECTION-OVER-UNION THRESHOLD")
    w("-" * 80)

    iou_header = f"{'IoU Threshold':>14s}  {'Mean Average Precision':>23s}"
    w(iou_header)
    w("-" * len(iou_header))

    for iou_thr in result.iou_thresholds:
        cls_metrics = result.per_iou_class_metrics.get(iou_thr, {})
        aps = [m.ap for m in cls_metrics.values() if m.num_gt > 0]
        map_val = sum(aps) / len(aps) if aps else 0.0
        w(f"{iou_thr:>14.2f}  {map_val:>23.4f}")

    w("")

    # ------------------------------------------------------------------
    # Per-class AP at every IoU threshold
    # ------------------------------------------------------------------
    w("-" * 80)
    w("PER-CLASS AVERAGE PRECISION AT EVERY INTERSECTION-OVER-UNION THRESHOLD")
    w("-" * 80)

    sorted_class_ids = sorted(
        {cid for cm in result.per_iou_class_metrics.values() for cid in cm}
    )

    for cls_id in sorted_class_ids:
        name = (class_names or {}).get(cls_id, str(cls_id))
        w(f"\n  Class: {name} (id={cls_id})")

        cls_header = f"    {'IoU Threshold':>14s}  {'Average Precision':>18s}  {'True Positives':>15s}  {'False Positives':>15s}  {'False Negatives':>15s}"
        w(cls_header)
        w("    " + "-" * (len(cls_header) - 4))

        for iou_thr in result.iou_thresholds:
            m = result.per_iou_class_metrics.get(iou_thr, {}).get(cls_id)
            if m is None:
                continue
            w(
                f"    {iou_thr:>14.2f}"
                f"  {m.ap:>18.4f}"
                f"  {m.tp:>15d}"
                f"  {m.fp:>15d}"
                f"  {m.fn:>15d}"
            )

    w("")
    w("=" * 80)
    w("END OF REPORT")
    w("=" * 80)

    return "\n".join(lines)


def run_evaluation(
    gt_dir: str,
    pred_dir: str,
    report_path: str = "report.txt",
    class_names: Optional[Dict[int, str]] = None,
) -> EvaluationResult:
    """Run mAP evaluation and write a detailed report to a text file.

    Args:
        gt_dir: Directory containing ground-truth ``.txt`` label files.
        pred_dir: Directory containing prediction ``.txt`` label files.
        report_path: Output path for the evaluation report.
        class_names: Optional mapping ``{class_id: name}`` for display.

    Returns:
        The ``EvaluationResult`` produced by the evaluation.
    """
    gt_path = Path(gt_dir)
    pred_path = Path(pred_dir)

    if not gt_path.is_dir():
        print(f"Error: Ground-Truth directory not found: {gt_path}")
        sys.exit(1)
    if not pred_path.is_dir():
        print(f"Error: Prediction directory not found: {pred_path}")
        sys.exit(1)

    print(f"Ground-Truth directory : {gt_path}")
    print(f"Prediction directory   : {pred_path}")

    # ------------------------------------------------------------------
    # Run evaluation (default COCO IoU thresholds: 0.50, 0.55, …, 0.95)
    # ------------------------------------------------------------------
    result = evaluate(
        gt_dir=gt_path,
        pred_dir=pred_path,
        class_names=class_names,
    )

    # ------------------------------------------------------------------
    # Build and write the report
    # ------------------------------------------------------------------
    report = _build_report(result, gt_path, pred_path, class_names)

    out = Path(report_path)
    out.write_text(report, encoding="utf-8")
    print(f"\nReport written to: {out.resolve()}")

    return result


if __name__ == "__main__":
    # ======================================================================
    # CONFIGURATION — adjust these paths to match your setup
    # ======================================================================

    # Ground-truth labels (benchmark dataset shipped with the repo)
    gt_dir = "./data/benchmark_dataset/label"

    # Prediction labels produced by running example.py (COCO writer output)
    pred_dir = "./data/eric_yolo_inference/labels"

    # Where to save the evaluation report
    report_path = "./data/eric_yolo_report.txt"

    # ======================================================================

    run_evaluation(
        gt_dir=gt_dir,
        pred_dir=pred_dir,
        report_path=report_path,
        class_names=CLASS_NAMES,
    )
