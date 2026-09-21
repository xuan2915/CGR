"""Standalone QVHighlights evaluator - no nncore dependency.

Reports R1, R5 and mAP at IoU thresholds 0.3 / 0.5 / 0.7 plus the mean best-IoU
of the top-1 prediction, which is the mIoU column of the paper.

    python scripts/evaluate.py outputs/cgr_top5_pad025.jsonl \
        data/qvhighlights/highlight_val_release.jsonl
"""
import json, sys, os
from collections import defaultdict

def compute_iou(span_a, span_b):
    s = max(span_a[0], span_b[0])
    e = min(span_a[1], span_b[1])
    if s >= e: return 0.0
    inter = e - s
    union = (span_a[1] - span_a[0]) + (span_b[1] - span_b[0]) - inter
    return inter / union if union > 0 else 0.0

def compute_map(preds, gts, iou_thr):
    """Compute average precision for one sample."""
    if not gts: return 0.0
    tp = [0] * len(preds)
    fp = [0] * len(preds)
    gt_used = [False] * len(gts)
    for i, p in enumerate(preds):
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if gt_used[j]: continue
            iou = compute_iou(p[:2], g)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            tp[i] = 1
            gt_used[best_j] = True
        else:
            fp[i] = 1
    # Compute AP
    tp_cum, fp_cum = 0, 0
    precisions = []
    for i in range(len(preds)):
        tp_cum += tp[i]
        fp_cum += fp[i]
        precisions.append(tp_cum / max(tp_cum + fp_cum, 1))
    # AP = average of precision at each recall point
    if not precisions: return 0.0
    ap = 0.0
    for i, p in enumerate(precisions):
        if tp[i]:
            ap += p
    return ap / max(len(gts), 1)

def load_preds(path):
    preds = []
    with open(path) as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line))
    return preds

def load_gts(path):
    gts = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                key = (d["vid"], d["qid"])
                gts[key] = d.get("relevant_windows", [])
    return gts

def evaluate(pred_path, gt_path):
    preds = load_preds(pred_path)
    gts = load_gts(gt_path)

    results = {}
    for thr in [0.3, 0.5, 0.7]:
        r1_hits = 0
        r5_hits = 0
        total = 0
        maps = []

        for p in preds:
            key = (p["vid"], p["qid"])
            if key not in gts: continue
            gt_windows = gts[key]
            pred_windows = p["pred_relevant_windows"]
            total += 1

            # R1: top-1 prediction hits any GT
            if len(pred_windows) > 0:
                top1 = pred_windows[0]
                r1_hit = any(compute_iou(top1[:2], g) >= thr for g in gt_windows)
                if r1_hit: r1_hits += 1

            # R5: any of top-5 hits any GT
            if len(pred_windows) >= 5:
                top5 = pred_windows[:5]
            else:
                top5 = pred_windows
            r5_hit = False
            for pw in top5:
                if any(compute_iou(pw[:2], g) >= thr for g in gt_windows):
                    r5_hit = True
                    break
            if r5_hit: r5_hits += 1

            # mAP
            maps.append(compute_map(pred_windows[:100], gt_windows, thr))

        results[f"R1@{thr}"] = 100.0 * r1_hits / total if total > 0 else 0
        results[f"R5@{thr}"] = 100.0 * r5_hits / total if total > 0 else 0
        results[f"mAP@{thr}"] = 100.0 * sum(maps) / len(maps) if maps else 0

    # mIoU: best IoU of the top-1 prediction, independent of any threshold
    mious = []
    for p in preds:
        key = (p["vid"], p["qid"])
        if key not in gts: continue
        gt_windows = gts[key]
        pred_windows = p["pred_relevant_windows"]
        if not pred_windows or not gt_windows:
            mious.append(0.0)
            continue
        mious.append(max(compute_iou(pred_windows[0][:2], g) for g in gt_windows))
    results["mIoU"] = 100.0 * sum(mious) / len(mious) if mious else 0

    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <pred.jsonl> [gt.jsonl]")
        sys.exit(1)
    
    pred_path = sys.argv[1]
    gt_path = sys.argv[2] if len(sys.argv) > 2 else "data/qvhighlights/highlight_val_release.jsonl"
    
    results = evaluate(pred_path, gt_path)
    
    for k, v in results.items():
        print(f"MR-full-{k}: {v:.2f}")

