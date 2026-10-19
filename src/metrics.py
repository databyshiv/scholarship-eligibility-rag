"""
metrics.py - Accuracy, Precision, Recall and F1 for the Scholarship RAG.

Reads data/evaluation/evaluation_results.csv (created by evaluation.py)
and prints a classification report. No LLM re-run needed.

Usage (from the project folder):
    python src/metrics.py
    python src/metrics.py path/to/other_results.csv
"""

import csv
import os
import sys

LABELS = ["ELIGIBLE", "INELIGIBLE", "CANNOT_DETERMINE"]


def load(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    gt = [r["ground_truth"].strip().upper() for r in rows]
    pred = [r["predicted"].strip().upper() for r in rows]
    return rows, gt, pred


def per_class(gt, pred, label):
    tp = sum(1 for g, p in zip(gt, pred) if g == label and p == label)
    fp = sum(1 for g, p in zip(gt, pred) if g != label and p == label)
    fn = sum(1 for g, p in zip(gt, pred) if g == label and p != label)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    support = sum(1 for g in gt if g == label)
    return precision, recall, f1, support


def report(gt, pred, title="OVERALL"):
    n = len(gt)
    acc = sum(g == p for g, p in zip(gt, pred)) / n if n else 0.0
    labels = [l for l in LABELS if l in gt or l in pred]
    stats = {l: per_class(gt, pred, l) for l in labels}

    total_support = sum(s[3] for s in stats.values())
    macro = [sum(s[i] for s in stats.values()) / len(stats) for i in range(3)]
    weighted = [sum(s[i] * s[3] for s in stats.values()) / total_support
                if total_support else 0.0 for i in range(3)]

    print(f"\n{'-' * 70}\n  {title}  (n={n})\n{'-' * 70}")
    print(f"  Accuracy: {acc:.1%}  ({sum(g == p for g, p in zip(gt, pred))}/{n})\n")
    print(f"  {'Class':<20}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}")
    for l, (p, r, f, s) in stats.items():
        print(f"  {l:<20}{p:>10.3f}{r:>10.3f}{f:>10.3f}{s:>10}")
    print(f"  {'Macro avg':<20}{macro[0]:>10.3f}{macro[1]:>10.3f}{macro[2]:>10.3f}{total_support:>10}")
    print(f"  {'Weighted avg':<20}{weighted[0]:>10.3f}{weighted[1]:>10.3f}{weighted[2]:>10.3f}{total_support:>10}")
    return {"accuracy": acc, "macro_f1": macro[2], "weighted_f1": weighted[2]}


def confusion(gt, pred):
    labels = [l for l in LABELS if l in gt or l in pred]
    extra = sorted(set(pred) - set(labels))  # e.g. UNPARSEABLE
    cols = labels + extra
    short = {"ELIGIBLE": "ELIG", "INELIGIBLE": "INELIG", "CANNOT_DETERMINE": "CANNOT"}
    print(f"\n{'-' * 70}\n  CONFUSION MATRIX (rows = ground truth, cols = predicted)\n{'-' * 70}")
    print(f"  {'':<20}" + "".join(f"{short.get(c, c)[:8]:>10}" for c in cols))
    for g in labels:
        print(f"  {g:<20}" + "".join(
            f"{sum(1 for a, b in zip(gt, pred) if a == g and b == c):>10}" for c in cols))


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        base, "data/evaluation/evaluation_results.csv")
    rows, gt, pred = load(path)

    print("=" * 70 + "\n  SCHOLARSHIP RAG - ACCURACY & F1 REPORT\n" + "=" * 70)
    print(f"  Results file: {path}")
    report(gt, pred)
    confusion(gt, pred)

    for field in ["difficulty", "demographic_tag"]:
        for group in sorted(set(r[field] for r in rows)):
            idx = [i for i, r in enumerate(rows) if r[field] == group]
            report([gt[i] for i in idx], [pred[i] for i in idx],
                   title=f"{field.upper()}: {group}")
    print("=" * 70)


if __name__ == "__main__":
    main()
