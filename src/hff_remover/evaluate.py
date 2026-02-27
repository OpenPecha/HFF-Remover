"""Mean Average Precision (mAP) evaluation for YOLO-format layout detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default COCO-style IoU thresholds: 0.50, 0.55, …, 0.95
DEFAULT_IOU_THRESHOLDS: List[float] = [
    round(0.5 + 0.05 * i, 2) for i in range(10)
]


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class BoundingBox:
    """A single YOLO-format bounding box.

    Attributes:
        class_id: Integer class label.
        cx: Normalised centre-x.
        cy: Normalised centre-y.
        w: Normalised width.
        h: Normalised height.
        confidence: Detection confidence (defaults to 1.0 for GT).
    """

    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    confidence: float = 1.0

    @property
    def x1(self) -> float:
        """Left edge in normalised coordinates."""
        return self.cx - self.w / 2.0

    @property
    def y1(self) -> float:
        """Top edge in normalised coordinates."""
        return self.cy - self.h / 2.0

    @property
    def x2(self) -> float:
        """Right edge in normalised coordinates."""
        return self.cx + self.w / 2.0

    @property
    def y2(self) -> float:
        """Bottom edge in normalised coordinates."""
        return self.cy + self.h / 2.0


@dataclass
class ClassMetrics:
    """Per-class evaluation metrics at a single IoU threshold.

    Attributes:
        ap: Average Precision (area under interpolated PR curve).
        precision: Precision after processing all predictions.
        recall: Recall after processing all predictions.
        tp: Total true-positive count.
        fp: Total false-positive count.
        fn: Total false-negative (missed GT) count.
        num_gt: Total ground-truth boxes for this class.
        num_pred: Total prediction boxes for this class.
    """

    ap: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    num_gt: int = 0
    num_pred: int = 0


@dataclass
class EvaluationResult:
    """Aggregated evaluation result across all classes and IoU thresholds.

    Attributes:
        map_50: mAP at IoU = 0.50.
        map_50_95: mAP averaged over IoU 0.50 … 0.95 (COCO-style).
        class_metrics_50: Per-class ``ClassMetrics`` at IoU = 0.50.
        per_iou_class_metrics: Per-IoU, per-class metrics dict.
        iou_thresholds: IoU thresholds used.
        total_gt: Total GT boxes across all classes.
        total_pred: Total prediction boxes across all classes.
    """

    map_50: float = 0.0
    map_50_95: float = 0.0
    class_metrics_50: Dict[int, ClassMetrics] = field(default_factory=dict)
    per_iou_class_metrics: Dict[float, Dict[int, ClassMetrics]] = field(
        default_factory=dict
    )
    iou_thresholds: List[float] = field(default_factory=list)
    total_gt: int = 0
    total_pred: int = 0


# =============================================================================
# Parsing
# =============================================================================


def parse_yolo_label_file(path: Path) -> List[BoundingBox]:
    """Parse a YOLO-format label file into a list of bounding boxes.

    Each non-empty line must contain at least 5 space-separated values:
    ``class_id cx cy w h [confidence]``.  Lines that cannot be parsed are
    skipped with a warning.

    Args:
        path: Path to the ``.txt`` label file.

    Returns:
        List of parsed ``BoundingBox`` objects.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")

    boxes: List[BoundingBox] = []
    for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            logger.warning("Skipping malformed line %d in %s: %r", line_no, path, line)
            continue
        try:
            class_id = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            confidence = float(parts[5]) if len(parts) >= 6 else 1.0
        except (ValueError, IndexError) as exc:
            logger.warning(
                "Skipping malformed line %d in %s: %r (%s)", line_no, path, line, exc
            )
            continue
        boxes.append(
            BoundingBox(
                class_id=class_id,
                cx=cx,
                cy=cy,
                w=w,
                h=h,
                confidence=confidence,
            )
        )
    return boxes


# =============================================================================
# IoU computation
# =============================================================================


def compute_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Compute Intersection-over-Union between two bounding boxes.

    Both boxes are expected to be in normalised YOLO format.

    Args:
        box_a: First bounding box.
        box_b: Second bounding box.

    Returns:
        IoU value in [0, 1].
    """
    x1 = max(box_a.x1, box_b.x1)
    y1 = max(box_a.y1, box_b.y1)
    x2 = min(box_a.x2, box_b.x2)
    y2 = min(box_a.y2, box_b.y2)

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a.w * box_a.h)
    area_b = max(0.0, box_b.w * box_b.h)
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


# =============================================================================
# Prediction ↔ GT matching
# =============================================================================


def match_predictions(
    preds: List[BoundingBox],
    gts: List[BoundingBox],
    iou_threshold: float,
) -> Tuple[List[bool], int]:
    """Greedily match predictions to ground-truth boxes for a single class.

    Predictions are sorted by descending confidence, then each prediction is
    matched to the GT box with the highest IoU (above *iou_threshold*) that
    has not already been matched.

    Args:
        preds: Prediction boxes (single class).
        gts: Ground-truth boxes (single class).
        iou_threshold: Minimum IoU to count as a true positive.

    Returns:
        A tuple ``(tp_flags, fn_count)`` where *tp_flags* is a boolean list
        (one entry per prediction, ``True`` = TP, ``False`` = FP) and
        *fn_count* is the number of unmatched GT boxes.
    """
    if not preds:
        return [], len(gts)
    if not gts:
        return [False] * len(preds), 0

    # Sort predictions by confidence (descending) for deterministic order
    sorted_preds = sorted(preds, key=lambda b: b.confidence, reverse=True)

    matched_gt: set[int] = set()
    tp_flags: List[bool] = []

    for pred in sorted_preds:
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(gts):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp_flags.append(True)
            matched_gt.add(best_gt_idx)
        else:
            tp_flags.append(False)

    fn_count = len(gts) - len(matched_gt)
    return tp_flags, fn_count


# =============================================================================
# Average Precision computation
# =============================================================================


def compute_ap(tp_flags: Sequence[bool], num_gt: int) -> float:
    """Compute Average Precision using the all-points interpolation method.

    This follows the VOC/COCO convention: the precision-recall curve is
    interpolated so that precision is monotonically decreasing, and AP is the
    area under that curve.

    Args:
        tp_flags: Ordered sequence of TP (True) / FP (False) flags for each
            prediction, sorted by descending confidence.
        num_gt: Total number of ground-truth boxes for this class.

    Returns:
        AP value in [0, 1].  Returns 0.0 when *num_gt* is 0.
    """
    if num_gt == 0:
        return 0.0
    if not tp_flags:
        return 0.0

    tp_arr = np.array(tp_flags, dtype=np.float64)
    fp_arr = 1.0 - tp_arr

    tp_cumsum = np.cumsum(tp_arr)
    fp_cumsum = np.cumsum(fp_arr)

    recalls = tp_cumsum / num_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Prepend (recall=0, precision=1) sentinel
    recalls = np.concatenate(([0.0], recalls))
    precisions = np.concatenate(([1.0], precisions))

    # All-points interpolation: make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Compute area under the interpolated PR curve
    recall_diffs = np.diff(recalls)
    ap: float = float(np.sum(recall_diffs * precisions[1:]))
    return ap


# =============================================================================
# Top-level evaluation
# =============================================================================


def _collect_labels_by_class(
    boxes: List[BoundingBox],
) -> Dict[int, List[BoundingBox]]:
    """Group a flat list of boxes by their class_id."""
    by_class: Dict[int, List[BoundingBox]] = {}
    for box in boxes:
        by_class.setdefault(box.class_id, []).append(box)
    return by_class


def _compute_class_metrics(
    all_tp_flags: List[bool],
    num_gt: int,
    num_pred: int,
) -> ClassMetrics:
    """Build a ``ClassMetrics`` from accumulated TP/FP flags."""
    ap = compute_ap(all_tp_flags, num_gt)
    tp = sum(all_tp_flags)
    fp = num_pred - tp
    fn = num_gt - tp
    precision = tp / num_pred if num_pred > 0 else 0.0
    recall = tp / num_gt if num_gt > 0 else 0.0
    return ClassMetrics(
        ap=ap,
        precision=precision,
        recall=recall,
        tp=tp,
        fp=fp,
        fn=fn,
        num_gt=num_gt,
        num_pred=num_pred,
    )


def evaluate(
    gt_dir: Path,
    pred_dir: Path,
    class_names: Optional[Dict[int, str]] = None,
    iou_thresholds: Optional[List[float]] = None,
) -> EvaluationResult:
    """Evaluate predictions against ground truth in YOLO label format.

    Ground-truth and prediction directories must contain ``.txt`` files with
    matching stems.  Files present in only one directory are handled
    gracefully (missing GT → all FP; missing pred → all FN).

    Args:
        gt_dir: Directory of ground-truth ``.txt`` label files.
        pred_dir: Directory of prediction ``.txt`` label files.
        class_names: Optional mapping ``{class_id: name}`` used for logging.
        iou_thresholds: IoU thresholds to evaluate. Defaults to
            ``[0.50, 0.55, …, 0.95]``.

    Returns:
        An ``EvaluationResult`` with mAP@0.5, mAP@0.5:0.95, and per-class
        metrics.

    Raises:
        NotADirectoryError: If *gt_dir* or *pred_dir* is not a directory.
    """
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)

    if not gt_dir.is_dir():
        raise NotADirectoryError(f"GT directory not found: {gt_dir}")
    if not pred_dir.is_dir():
        raise NotADirectoryError(f"Prediction directory not found: {pred_dir}")

    if iou_thresholds is None:
        iou_thresholds = DEFAULT_IOU_THRESHOLDS

    # Discover all label stems from both directories
    gt_stems = {p.stem for p in gt_dir.glob("*.txt")}
    pred_stems = {p.stem for p in pred_dir.glob("*.txt")}
    all_stems = sorted(gt_stems | pred_stems)

    if not all_stems:
        logger.warning("No label files found in GT or prediction directories.")
        return EvaluationResult(iou_thresholds=iou_thresholds)

    # Parse all files once
    gt_boxes_per_image: Dict[str, List[BoundingBox]] = {}
    pred_boxes_per_image: Dict[str, List[BoundingBox]] = {}

    for stem in all_stems:
        gt_path = gt_dir / f"{stem}.txt"
        pred_path = pred_dir / f"{stem}.txt"

        gt_boxes_per_image[stem] = (
            parse_yolo_label_file(gt_path) if gt_path.exists() else []
        )
        pred_boxes_per_image[stem] = (
            parse_yolo_label_file(pred_path) if pred_path.exists() else []
        )

    # Determine all class IDs present
    all_class_ids: set[int] = set()
    for boxes in gt_boxes_per_image.values():
        all_class_ids.update(b.class_id for b in boxes)
    for boxes in pred_boxes_per_image.values():
        all_class_ids.update(b.class_id for b in boxes)

    if class_names is not None:
        all_class_ids.update(class_names.keys())

    sorted_class_ids = sorted(all_class_ids)

    # Evaluate at each IoU threshold
    per_iou_class_metrics: Dict[float, Dict[int, ClassMetrics]] = {}

    for iou_thr in iou_thresholds:
        class_metrics_at_thr: Dict[int, ClassMetrics] = {}

        for cls_id in sorted_class_ids:
            all_tp_flags: List[bool] = []
            total_gt = 0
            total_pred = 0

            for stem in all_stems:
                gt_by_cls = _collect_labels_by_class(gt_boxes_per_image[stem])
                pred_by_cls = _collect_labels_by_class(pred_boxes_per_image[stem])

                gt_cls = gt_by_cls.get(cls_id, [])
                pred_cls = pred_by_cls.get(cls_id, [])

                total_gt += len(gt_cls)
                total_pred += len(pred_cls)

                tp_flags, _ = match_predictions(pred_cls, gt_cls, iou_thr)
                all_tp_flags.extend(tp_flags)

            class_metrics_at_thr[cls_id] = _compute_class_metrics(
                all_tp_flags, total_gt, total_pred
            )

        per_iou_class_metrics[iou_thr] = class_metrics_at_thr

    # Aggregate mAP
    map_50 = _mean_ap(per_iou_class_metrics.get(0.5, {}))

    all_maps = [
        _mean_ap(per_iou_class_metrics[thr]) for thr in iou_thresholds
    ]
    map_50_95 = float(np.mean(all_maps)) if all_maps else 0.0

    # Summary counts (from IoU=0.5 metrics)
    total_gt = sum(
        m.num_gt for m in per_iou_class_metrics.get(0.5, {}).values()
    )
    total_pred = sum(
        m.num_pred for m in per_iou_class_metrics.get(0.5, {}).values()
    )

    return EvaluationResult(
        map_50=map_50,
        map_50_95=map_50_95,
        class_metrics_50=per_iou_class_metrics.get(0.5, {}),
        per_iou_class_metrics=per_iou_class_metrics,
        iou_thresholds=iou_thresholds,
        total_gt=total_gt,
        total_pred=total_pred,
    )


def _mean_ap(class_metrics: Dict[int, ClassMetrics]) -> float:
    """Compute mean AP across classes, ignoring classes with no GT."""
    aps = [m.ap for m in class_metrics.values() if m.num_gt > 0]
    if not aps:
        return 0.0
    return float(np.mean(aps))


# =============================================================================
# Reporting
# =============================================================================


def print_report(
    result: EvaluationResult,
    class_names: Optional[Dict[int, str]] = None,
) -> None:
    """Pretty-print the evaluation results to stdout.

    Args:
        result: The ``EvaluationResult`` to display.
        class_names: Optional mapping ``{class_id: name}`` for display.
    """
    header = (
        f"{'Class':>12s}  {'AP@.5':>7s}  {'Prec':>7s}  {'Rec':>7s}"
        f"  {'TP':>6s}  {'FP':>6s}  {'FN':>6s}  {'GT':>6s}  {'Pred':>6s}"
    )
    separator = "-" * len(header)

    print("\n" + separator)
    print(header)
    print(separator)

    for cls_id, metrics in sorted(result.class_metrics_50.items()):
        name = (class_names or {}).get(cls_id, str(cls_id))
        print(
            f"{name:>12s}  {metrics.ap:7.4f}  {metrics.precision:7.4f}"
            f"  {metrics.recall:7.4f}  {metrics.tp:6d}  {metrics.fp:6d}"
            f"  {metrics.fn:6d}  {metrics.num_gt:6d}  {metrics.num_pred:6d}"
        )

    print(separator)
    print(f"{'mAP@0.5':>12s}  {result.map_50:7.4f}")
    print(f"{'mAP@.5:.95':>12s}  {result.map_50_95:7.4f}")
    print(
        f"{'Totals':>12s}  {'':>7s}  {'':>7s}  {'':>7s}"
        f"  {'':>6s}  {'':>6s}  {'':>6s}"
        f"  {result.total_gt:6d}  {result.total_pred:6d}"
    )
    print(separator + "\n")
