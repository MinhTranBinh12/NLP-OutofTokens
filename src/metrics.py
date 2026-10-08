"""
Hệ thống độ đo DÙNG CHUNG cho cả 3 mô hình (Baseline, Multi-Head, CRF).

Không sửa các hàm đã có. Nếu cần metric mới (vd Span-level F1),
hãy THÊM hàm mới ở cuối file để tránh conflict.

Sentence-level (so sánh cả 3 mô hình):
  - Macro-F1 từng khía cạnh trên 4 lớp {0: None, 1: positive, 2: neutral, 3: negative}
  - Mean Aspect Macro-F1 (metric chính) = trung bình 5 Macro-F1
  - Exact Match Ratio = tỉ lệ câu đúng cả 5 khía cạnh
"""

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ASPECTS: List[str] = ["Material", "General", "Service", "Design", "Price"]
CLASS_IDS: List[int] = [0, 1, 2, 3]
CLASS_NAMES: List[str] = ["None", "positive", "neutral", "negative"]


def sentence_level_metrics(y_true: Sequence[Sequence[int]], y_pred: Sequence[Sequence[int]]) -> Dict[str, float]:
    """
    y_true, y_pred: mảng [N, 5] với giá trị 0..3.
    Trả về dict gồm F1 từng khía cạnh, Mean Aspect Macro-F1, Exact Match Ratio.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape and y_true.shape[1] == len(ASPECTS), "Shape phải là [N, 5]"

    results: Dict[str, float] = {}
    f1s = []
    for k, asp in enumerate(ASPECTS):
        f1 = f1_score(y_true[:, k], y_pred[:, k], labels=CLASS_IDS, average="macro", zero_division=0)
        results[f"{asp}_macro_f1"] = float(f1)
        results[f"{asp}_acc"] = float(accuracy_score(y_true[:, k], y_pred[:, k]))
        f1s.append(f1)

    results["mean_aspect_macro_f1"] = float(np.mean(f1s))
    results["exact_match_ratio"] = float((y_true == y_pred).all(axis=1).mean())
    return results


def per_aspect_confusion(y_true: Sequence[Sequence[int]], y_pred: Sequence[Sequence[int]]) -> Dict[str, List[List[int]]]:
    """Confusion matrix 4x4 cho từng khía cạnh (hàng = nhãn thật, cột = dự đoán)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        asp: confusion_matrix(y_true[:, k], y_pred[:, k], labels=CLASS_IDS).tolist()
        for k, asp in enumerate(ASPECTS)
    }


def format_metrics(results: Dict[str, float]) -> str:
    """In bảng kết quả gọn gàng ra console."""
    lines = [f"{'Aspect':<10} {'Macro-F1':>9} {'Acc':>8}"]
    for asp in ASPECTS:
        lines.append(f"{asp:<10} {results[f'{asp}_macro_f1'] * 100:>8.2f}% {results[f'{asp}_acc'] * 100:>7.2f}%")
    lines.append(f"{'MEAN F1':<10} {results['mean_aspect_macro_f1'] * 100:>8.2f}%")
    lines.append(f"{'EXACT':<10} {results['exact_match_ratio'] * 100:>8.2f}%")
    return "\n".join(lines)
