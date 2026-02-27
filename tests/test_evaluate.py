"""Tests for the evaluate module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from hff_remover.evaluate import (
    BoundingBox,
    ClassMetrics,
    EvaluationResult,
    compute_ap,
    compute_iou,
    evaluate,
    match_predictions,
    parse_yolo_label_file,
    print_report,
)


# =============================================================================
# Helpers
# =============================================================================


def _write_label(directory: Path, stem: str, lines: list[str]) -> Path:
    """Write a YOLO label file and return its path."""
    path = directory / f"{stem}.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def _box(
    class_id: int = 0,
    cx: float = 0.5,
    cy: float = 0.5,
    w: float = 0.2,
    h: float = 0.2,
    confidence: float = 1.0,
) -> BoundingBox:
    """Shorthand for creating a BoundingBox."""
    return BoundingBox(
        class_id=class_id, cx=cx, cy=cy, w=w, h=h, confidence=confidence
    )


# =============================================================================
# TestParseYoloLabelFile
# =============================================================================


class TestParseYoloLabelFile:
    """Tests for parse_yolo_label_file."""

    def test_valid_file(self) -> None:
        """Parse a well-formed label file with two boxes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_label(
                Path(tmpdir),
                "img1",
                ["0 0.5 0.5 0.8 0.8", "1 0.1 0.9 0.1 0.02"],
            )
            boxes = parse_yolo_label_file(path)

            assert len(boxes) == 2
            assert boxes[0].class_id == 0
            assert boxes[0].cx == pytest.approx(0.5)
            assert boxes[1].class_id == 1
            assert boxes[1].cy == pytest.approx(0.9)

    def test_file_with_confidence(self) -> None:
        """Parse a label file where the 6th field is confidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_label(
                Path(tmpdir), "img2", ["0 0.5 0.5 0.8 0.8 0.95"]
            )
            boxes = parse_yolo_label_file(path)

            assert len(boxes) == 1
            assert boxes[0].confidence == pytest.approx(0.95)

    def test_empty_file(self) -> None:
        """An empty label file should return an empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.txt"
            path.write_text("")
            boxes = parse_yolo_label_file(path)

            assert boxes == []

    def test_malformed_lines_skipped(self) -> None:
        """Lines with fewer than 5 fields or non-numeric data are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_label(
                Path(tmpdir),
                "img3",
                [
                    "0 0.5 0.5 0.8 0.8",  # valid
                    "bad line",  # invalid
                    "1 0.1",  # too few fields
                    "2 0.3 0.3 0.2 0.2",  # valid
                ],
            )
            boxes = parse_yolo_label_file(path)

            assert len(boxes) == 2
            assert boxes[0].class_id == 0
            assert boxes[1].class_id == 2

    def test_missing_file_raises(self) -> None:
        """A FileNotFoundError is raised for a non-existent path."""
        with pytest.raises(FileNotFoundError):
            parse_yolo_label_file(Path("/nonexistent/file.txt"))

    def test_blank_lines_ignored(self) -> None:
        """Blank and whitespace-only lines are silently skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_label(
                Path(tmpdir),
                "img4",
                ["", "  ", "0 0.5 0.5 0.2 0.2", ""],
            )
            boxes = parse_yolo_label_file(path)

            assert len(boxes) == 1


# =============================================================================
# TestComputeIoU
# =============================================================================


class TestComputeIoU:
    """Tests for compute_iou."""

    def test_perfect_overlap(self) -> None:
        """Identical boxes should have IoU = 1."""
        a = _box(cx=0.5, cy=0.5, w=0.4, h=0.4)
        assert compute_iou(a, a) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        """Non-overlapping boxes should have IoU = 0."""
        a = _box(cx=0.1, cy=0.1, w=0.1, h=0.1)
        b = _box(cx=0.9, cy=0.9, w=0.1, h=0.1)
        assert compute_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """Known partial overlap should give expected IoU."""
        # Box A: [0.3, 0.3, 0.7, 0.7]  area = 0.16
        a = _box(cx=0.5, cy=0.5, w=0.4, h=0.4)
        # Box B: [0.4, 0.4, 0.8, 0.8]  area = 0.16
        b = _box(cx=0.6, cy=0.6, w=0.4, h=0.4)
        # Intersection: [0.4, 0.4, 0.7, 0.7]  area = 0.09
        # Union: 0.16 + 0.16 - 0.09 = 0.23
        expected = 0.09 / 0.23
        assert compute_iou(a, b) == pytest.approx(expected, abs=1e-6)

    def test_zero_area_box(self) -> None:
        """A box with zero width or height should yield IoU = 0."""
        a = _box(cx=0.5, cy=0.5, w=0.0, h=0.4)
        b = _box(cx=0.5, cy=0.5, w=0.4, h=0.4)
        assert compute_iou(a, b) == pytest.approx(0.0)

    def test_one_inside_other(self) -> None:
        """Small box fully inside large box."""
        big = _box(cx=0.5, cy=0.5, w=0.8, h=0.8)
        small = _box(cx=0.5, cy=0.5, w=0.2, h=0.2)
        # intersection = area(small) = 0.04,  union = 0.64
        expected = 0.04 / 0.64
        assert compute_iou(big, small) == pytest.approx(expected, abs=1e-6)

    def test_symmetry(self) -> None:
        """IoU(a, b) == IoU(b, a)."""
        a = _box(cx=0.3, cy=0.3, w=0.3, h=0.3)
        b = _box(cx=0.5, cy=0.5, w=0.3, h=0.3)
        assert compute_iou(a, b) == pytest.approx(compute_iou(b, a))


# =============================================================================
# TestMatchPredictions
# =============================================================================


class TestMatchPredictions:
    """Tests for match_predictions."""

    def test_single_match(self) -> None:
        """One prediction matching one GT should give TP."""
        preds = [_box(cx=0.5, cy=0.5, w=0.4, h=0.4)]
        gts = [_box(cx=0.5, cy=0.5, w=0.4, h=0.4)]

        tp_flags, fn = match_predictions(preds, gts, iou_threshold=0.5)

        assert tp_flags == [True]
        assert fn == 0

    def test_more_preds_than_gt(self) -> None:
        """Extra predictions become FP."""
        gt = [_box(cx=0.5, cy=0.5, w=0.4, h=0.4)]
        preds = [
            _box(cx=0.5, cy=0.5, w=0.4, h=0.4, confidence=0.9),
            _box(cx=0.1, cy=0.1, w=0.1, h=0.1, confidence=0.5),
        ]

        tp_flags, fn = match_predictions(preds, gt, iou_threshold=0.5)

        assert sum(tp_flags) == 1  # one TP
        assert tp_flags.count(False) == 1  # one FP
        assert fn == 0

    def test_more_gt_than_preds(self) -> None:
        """Unmatched GT boxes should be counted as FN."""
        preds = [_box(cx=0.5, cy=0.5, w=0.4, h=0.4)]
        gts = [
            _box(cx=0.5, cy=0.5, w=0.4, h=0.4),
            _box(cx=0.1, cy=0.1, w=0.1, h=0.1),
        ]

        tp_flags, fn = match_predictions(preds, gts, iou_threshold=0.5)

        assert tp_flags == [True]
        assert fn == 1

    def test_no_matches_below_threshold(self) -> None:
        """All predictions below IoU threshold should be FP."""
        preds = [_box(cx=0.1, cy=0.1, w=0.1, h=0.1)]
        gts = [_box(cx=0.9, cy=0.9, w=0.1, h=0.1)]

        tp_flags, fn = match_predictions(preds, gts, iou_threshold=0.5)

        assert tp_flags == [False]
        assert fn == 1

    def test_empty_predictions(self) -> None:
        """No predictions should give empty TP flags and all GT as FN."""
        gts = [_box(), _box(cx=0.2)]
        tp_flags, fn = match_predictions([], gts, iou_threshold=0.5)

        assert tp_flags == []
        assert fn == 2

    def test_empty_gt(self) -> None:
        """No GT should make all predictions FP with 0 FN."""
        preds = [_box(), _box(cx=0.2)]
        tp_flags, fn = match_predictions(preds, [], iou_threshold=0.5)

        assert tp_flags == [False, False]
        assert fn == 0

    def test_no_double_matching(self) -> None:
        """A single GT box must not be matched to two predictions."""
        gt = [_box(cx=0.5, cy=0.5, w=0.4, h=0.4)]
        preds = [
            _box(cx=0.5, cy=0.5, w=0.4, h=0.4, confidence=0.9),
            _box(cx=0.5, cy=0.5, w=0.4, h=0.4, confidence=0.8),
        ]

        tp_flags, fn = match_predictions(preds, gt, iou_threshold=0.5)

        assert sum(tp_flags) == 1  # only one TP despite two overlapping preds
        assert fn == 0


# =============================================================================
# TestComputeAP
# =============================================================================


class TestComputeAP:
    """Tests for compute_ap."""

    def test_all_tp(self) -> None:
        """All predictions are TP → AP should be 1.0."""
        assert compute_ap([True, True, True], num_gt=3) == pytest.approx(1.0)

    def test_all_fp(self) -> None:
        """All predictions are FP → AP should be 0.0."""
        assert compute_ap([False, False, False], num_gt=3) == pytest.approx(
            0.0
        )

    def test_mixed(self) -> None:
        """Mixed TP/FP should give an intermediate AP."""
        # TP, FP, TP with 3 GT
        # cumTP: [1, 1, 2]  cumFP: [0, 1, 1]
        # Prec:  [1, 0.5, 0.667]  Rec: [0.333, 0.333, 0.667]
        ap = compute_ap([True, False, True], num_gt=3)
        assert 0.0 < ap < 1.0

    def test_empty_predictions(self) -> None:
        """No predictions → AP is 0."""
        assert compute_ap([], num_gt=5) == pytest.approx(0.0)

    def test_zero_gt(self) -> None:
        """Zero GT → AP is defined as 0."""
        assert compute_ap([True, False], num_gt=0) == pytest.approx(0.0)

    def test_single_tp(self) -> None:
        """Single TP with 1 GT → AP is 1.0."""
        assert compute_ap([True], num_gt=1) == pytest.approx(1.0)

    def test_single_fp(self) -> None:
        """Single FP with 1 GT → AP is 0.0."""
        assert compute_ap([False], num_gt=1) == pytest.approx(0.0)

    def test_partial_recall(self) -> None:
        """One TP out of two GT → AP reflects partial recall."""
        # TP  → cumTP=1, cumFP=0, prec=1.0, rec=0.5
        ap = compute_ap([True], num_gt=2)
        assert ap == pytest.approx(0.5)


# =============================================================================
# TestEvaluate (end-to-end)
# =============================================================================


class TestEvaluate:
    """End-to-end tests for the evaluate function."""

    def test_perfect_predictions(self) -> None:
        """Identical GT and predictions should yield mAP = 1.0."""
        with tempfile.TemporaryDirectory() as gt_dir, tempfile.TemporaryDirectory() as pred_dir:
            gt_path = Path(gt_dir)
            pred_path = Path(pred_dir)

            lines = ["0 0.5 0.5 0.8 0.8", "1 0.1 0.9 0.1 0.02"]
            _write_label(gt_path, "img1", lines)
            _write_label(pred_path, "img1", lines)

            result = evaluate(
                gt_dir=gt_path,
                pred_dir=pred_path,
                iou_thresholds=[0.5],
            )

            assert result.map_50 == pytest.approx(1.0)
            for cls_metrics in result.class_metrics_50.values():
                assert cls_metrics.ap == pytest.approx(1.0)
                assert cls_metrics.fp == 0
                assert cls_metrics.fn == 0

    def test_no_predictions(self) -> None:
        """Empty prediction dir → mAP = 0, all GT are FN."""
        with tempfile.TemporaryDirectory() as gt_dir, tempfile.TemporaryDirectory() as pred_dir:
            gt_path = Path(gt_dir)
            pred_path = Path(pred_dir)

            _write_label(gt_path, "img1", ["0 0.5 0.5 0.8 0.8"])
            # pred_dir is empty

            result = evaluate(
                gt_dir=gt_path,
                pred_dir=pred_path,
                iou_thresholds=[0.5],
            )

            assert result.map_50 == pytest.approx(0.0)
            assert result.class_metrics_50[0].fn == 1

    def test_no_gt(self) -> None:
        """Empty GT dir → mAP = 0, all predictions are FP."""
        with tempfile.TemporaryDirectory() as gt_dir, tempfile.TemporaryDirectory() as pred_dir:
            gt_path = Path(gt_dir)
            pred_path = Path(pred_dir)

            # gt_dir is empty
            _write_label(pred_path, "img1", ["0 0.5 0.5 0.8 0.8"])

            result = evaluate(
                gt_dir=gt_path,
                pred_dir=pred_path,
                iou_thresholds=[0.5],
            )

            assert result.map_50 == pytest.approx(0.0)
            assert result.class_metrics_50[0].fp == 1

    def test_multi_class(self) -> None:
        """Multiple classes should each get their own metrics."""
        with tempfile.TemporaryDirectory() as gt_dir, tempfile.TemporaryDirectory() as pred_dir:
            gt_path = Path(gt_dir)
            pred_path = Path(pred_dir)

            _write_label(
                gt_path,
                "img1",
                [
                    "0 0.5 0.3 0.8 0.2",
                    "1 0.5 0.9 0.1 0.02",
                    "2 0.5 0.1 0.3 0.03",
                ],
            )
            # Predict class 0 and 2 correctly, miss class 1
            _write_label(
                pred_path,
                "img1",
                [
                    "0 0.5 0.3 0.8 0.2",
                    "2 0.5 0.1 0.3 0.03",
                ],
            )

            result = evaluate(
                gt_dir=gt_path,
                pred_dir=pred_path,
                iou_thresholds=[0.5],
            )

            assert result.class_metrics_50[0].ap == pytest.approx(1.0)
            assert result.class_metrics_50[2].ap == pytest.approx(1.0)
            assert result.class_metrics_50[1].ap == pytest.approx(0.0)

    def test_multiple_images(self) -> None:
        """Metrics should aggregate across multiple image files."""
        with tempfile.TemporaryDirectory() as gt_dir, tempfile.TemporaryDirectory() as pred_dir:
            gt_path = Path(gt_dir)
            pred_path = Path(pred_dir)

            _write_label(gt_path, "img1", ["0 0.5 0.5 0.4 0.4"])
            _write_label(gt_path, "img2", ["0 0.5 0.5 0.4 0.4"])

            _write_label(pred_path, "img1", ["0 0.5 0.5 0.4 0.4"])
            _write_label(pred_path, "img2", ["0 0.5 0.5 0.4 0.4"])

            result = evaluate(
                gt_dir=gt_path,
                pred_dir=pred_path,
                iou_thresholds=[0.5],
            )

            assert result.map_50 == pytest.approx(1.0)
            assert result.total_gt == 2
            assert result.total_pred == 2

    def test_coco_style_thresholds(self) -> None:
        """mAP@0.5:0.95 should be computed across multiple thresholds."""
        with tempfile.TemporaryDirectory() as gt_dir, tempfile.TemporaryDirectory() as pred_dir:
            gt_path = Path(gt_dir)
            pred_path = Path(pred_dir)

            _write_label(gt_path, "img1", ["0 0.5 0.5 0.4 0.4"])
            _write_label(pred_path, "img1", ["0 0.5 0.5 0.4 0.4"])

            result = evaluate(
                gt_dir=gt_path,
                pred_dir=pred_path,
                # defaults to COCO thresholds
            )

            # Perfect match → mAP should be 1.0 at every threshold
            assert result.map_50 == pytest.approx(1.0)
            assert result.map_50_95 == pytest.approx(1.0)

    def test_missing_gt_file(self) -> None:
        """A prediction file with no matching GT should count all as FP."""
        with tempfile.TemporaryDirectory() as gt_dir, tempfile.TemporaryDirectory() as pred_dir:
            gt_path = Path(gt_dir)
            pred_path = Path(pred_dir)

            _write_label(gt_path, "img1", ["0 0.5 0.5 0.4 0.4"])
            _write_label(pred_path, "img1", ["0 0.5 0.5 0.4 0.4"])
            # Extra prediction file with no GT counterpart
            _write_label(pred_path, "img_extra", ["0 0.2 0.2 0.1 0.1"])

            result = evaluate(
                gt_dir=gt_path,
                pred_dir=pred_path,
                iou_thresholds=[0.5],
            )

            # 1 GT matched + 1 extra FP → precision < 1, recall = 1
            metrics = result.class_metrics_50[0]
            assert metrics.tp == 1
            assert metrics.fp == 1
            assert metrics.fn == 0
            assert metrics.recall == pytest.approx(1.0)

    def test_missing_pred_file(self) -> None:
        """A GT file with no matching prediction should count all as FN."""
        with tempfile.TemporaryDirectory() as gt_dir, tempfile.TemporaryDirectory() as pred_dir:
            gt_path = Path(gt_dir)
            pred_path = Path(pred_dir)

            _write_label(gt_path, "img1", ["0 0.5 0.5 0.4 0.4"])
            _write_label(gt_path, "img_no_pred", ["0 0.2 0.2 0.1 0.1"])
            _write_label(pred_path, "img1", ["0 0.5 0.5 0.4 0.4"])

            result = evaluate(
                gt_dir=gt_path,
                pred_dir=pred_path,
                iou_thresholds=[0.5],
            )

            metrics = result.class_metrics_50[0]
            assert metrics.tp == 1
            assert metrics.fn == 1

    def test_invalid_gt_dir_raises(self) -> None:
        """Non-existent GT directory should raise NotADirectoryError."""
        with tempfile.TemporaryDirectory() as pred_dir:
            with pytest.raises(NotADirectoryError):
                evaluate(
                    gt_dir=Path("/nonexistent/gt"),
                    pred_dir=Path(pred_dir),
                )

    def test_invalid_pred_dir_raises(self) -> None:
        """Non-existent prediction directory should raise NotADirectoryError."""
        with tempfile.TemporaryDirectory() as gt_dir:
            with pytest.raises(NotADirectoryError):
                evaluate(
                    gt_dir=Path(gt_dir),
                    pred_dir=Path("/nonexistent/pred"),
                )

    def test_both_dirs_empty(self) -> None:
        """Both directories empty should return zero-valued result."""
        with tempfile.TemporaryDirectory() as gt_dir, tempfile.TemporaryDirectory() as pred_dir:
            result = evaluate(
                gt_dir=Path(gt_dir),
                pred_dir=Path(pred_dir),
                iou_thresholds=[0.5],
            )

            assert result.map_50 == pytest.approx(0.0)
            assert result.total_gt == 0
            assert result.total_pred == 0


# =============================================================================
# TestPrintReport
# =============================================================================


class TestPrintReport:
    """Tests for print_report (smoke test — just ensure no crash)."""

    def test_print_report_runs(self, capsys: pytest.CaptureFixture[str]) -> None:
        """print_report should run without errors on a valid result."""
        result = EvaluationResult(
            map_50=0.75,
            map_50_95=0.60,
            class_metrics_50={
                0: ClassMetrics(
                    ap=0.9, precision=0.8, recall=0.85,
                    tp=17, fp=4, fn=3, num_gt=20, num_pred=21,
                ),
                1: ClassMetrics(
                    ap=0.6, precision=0.7, recall=0.5,
                    tp=5, fp=2, fn=5, num_gt=10, num_pred=7,
                ),
            },
            iou_thresholds=[0.5],
            total_gt=30,
            total_pred=28,
        )

        print_report(result, class_names={0: "text", 1: "footer"})

        captured = capsys.readouterr()
        assert "text" in captured.out
        assert "footer" in captured.out
        assert "mAP@0.5" in captured.out

    def test_print_report_no_class_names(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """print_report should work when class_names is None."""
        result = EvaluationResult(
            map_50=0.5,
            map_50_95=0.4,
            class_metrics_50={
                0: ClassMetrics(ap=0.5, num_gt=10, num_pred=10),
            },
            iou_thresholds=[0.5],
            total_gt=10,
            total_pred=10,
        )

        print_report(result)

        captured = capsys.readouterr()
        assert "mAP@0.5" in captured.out
