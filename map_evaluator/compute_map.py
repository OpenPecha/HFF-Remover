import sys
import numpy as np

IOU_THRESHOLD = 0.5


def load_yolo(file):
    data = []
    with open(file) as f:
        for line in f:
            parts = list(map(float, line.split()))
            data.append(parts)
    return data


def yolo_to_box(x, y, w, h):
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return [x1, y1, x2, y2]


def iou(box1, box2):
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])

    inter = max(0, xb - xa) * max(0, yb - ya)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter

    return inter / union if union else 0


def average_precision(tp, fp, total_gt):
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    recall = tp / total_gt
    precision = tp / (tp + fp + 1e-6)

    ap = 0
    for t in np.linspace(0, 1, 11):
        p = precision[recall >= t]
        ap += max(p) if len(p) else 0

    return ap / 11


def main(gt_file, pred_file):
    gt = load_yolo(gt_file)
    pred = load_yolo(pred_file)

    classes = set([int(x[0]) for x in gt])

    aps = []

    for cls in classes:
        gt_cls = [g for g in gt if int(g[0]) == cls]
        pred_cls = [p for p in pred if int(p[0]) == cls]

        matched = set()
        tp = []
        fp = []

        for p in pred_cls:
            pbox = yolo_to_box(*p[1:])
            best_iou = 0
            best_gt = -1

            for i, g in enumerate(gt_cls):
                if i in matched:
                    continue

                gbox = yolo_to_box(*g[1:])
                score = iou(pbox, gbox)

                if score > best_iou:
                    best_iou = score
                    best_gt = i

            if best_iou >= IOU_THRESHOLD:
                tp.append(1)
                fp.append(0)
                matched.add(best_gt)
            else:
                tp.append(0)
                fp.append(1)

        ap = average_precision(tp, fp, len(gt_cls))
        aps.append(ap)

        print(f"Class {cls} AP: {ap:.4f}")

    print("\n====================")
    print(f"mAP: {np.mean(aps):.4f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])