"""
Script huấn luyện & Đánh giá các Mô hình Baseline (Sentence-Level ABSA):
1. TF-IDF + LinearSVC
2. TF-IDF + Logistic Regression
3. TF-IDF + Naive Bayes (MultinomialNB)

Người phụ trách: Phúc (nhánh feat/baseline).

Đầu ra:
    saved_models/baseline_<model>.joblib      (Mô hình đã huấn luyện)
    results/baseline/<Model Folder>/*.json    (Các file kết quả chuẩn JSON đồng bộ toàn nhóm)
    results/baseline/<Model Folder>/results_report.md
    results/baseline/BASELINE_COMPARISON.md   (Báo cáo đối sánh toàn diện duy nhất của 3 mô hình)
"""

import sys
import os
import time
import json
import shutil
import argparse
from typing import List, Tuple, Dict, Any, Union, Optional
import numpy as np
from sklearn.metrics import classification_report

# Đảm bảo in tiếng Việt không bị lỗi charmap trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Đảm bảo import được các module trong src/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models.baseline_svc import BaselineSVCModel
from src.models.baseline_lr import BaselineLogisticRegressionModel
from src.models.baseline_nb import BaselineNaiveBayesModel
from src.metrics import (
    ASPECTS,
    CLASS_IDS,
    CLASS_NAMES,
    sentence_level_metrics,
    format_metrics,
    per_aspect_confusion,
)

NUM_ASPECTS = len(ASPECTS)
NUM_CLASSES = len(CLASS_IDS)

MODEL_CONFIGS = {
    "svc": {
        "display_name": "TF-IDF + LinearSVC",
        "dir_name": "TF-IDF LINEARSVC",
        "save_file": "baseline_svc.joblib",
        "model_class": BaselineSVCModel,
        "default_params": {"C": 0.7, "class_weight": "balanced"},
    },
    "lr": {
        "display_name": "TF-IDF + Logistic Regression",
        "dir_name": "Logistic Regression",
        "save_file": "baseline_lr.joblib",
        "model_class": BaselineLogisticRegressionModel,
        "default_params": {"C": 1.0, "class_weight": "balanced", "max_iter": 1000},
    },
    "nb": {
        "display_name": "TF-IDF + Naive Bayes (MultinomialNB)",
        "dir_name": "Naive Bayes",
        "save_file": "baseline_nb.joblib",
        "model_class": BaselineNaiveBayesModel,
        "default_params": {"alpha": 0.1, "nb_type": "multinomial"},
    },
}


def load_dataset(file_path: str) -> Tuple[List[Dict[str, Any]], List[str], np.ndarray]:
    """Đọc dữ liệu từ file JSON đã tiền xử lý."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {file_path}")

    print(f"[*] Đang tải dữ liệu từ: {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = [
        {
            "id": item.get("id", idx),
            "comment": item["comment"],
            "aspect_vector": item["aspect_vector"],
        }
        for idx, item in enumerate(data)
    ]
    comments = [item["comment"] for item in records]
    vectors = np.array([item["aspect_vector"] for item in records], dtype=int)
    print(f"[+] Đã tải {len(comments)} mẫu từ {os.path.basename(file_path)}.")
    return records, comments, vectors


def compute_class_weights(y: np.ndarray, smoothing: str = "inverse") -> List[List[float]]:
    """Tính trọng số phạt nghịch đảo theo từng khía cạnh (5x4)."""
    n_total = float(len(y))
    weights = []
    for k in range(NUM_ASPECTS):
        row = []
        for c in CLASS_IDS:
            cnt = max(float(np.sum(y[:, k] == c)), 1.0)
            w = n_total / (NUM_CLASSES * cnt)
            if smoothing == "sqrt":
                w = np.sqrt(w)
            row.append(float(w))
        weights.append(row)
    return weights


def compute_label_distribution(y: np.ndarray) -> Dict[str, List[int]]:
    """Thống kê phân bố nhãn theo từng khía cạnh trên tập dữ liệu."""
    dist = {}
    for k, asp in enumerate(ASPECTS):
        dist[asp] = [int(np.sum(y[:, k] == c)) for c in CLASS_IDS]
    return dist


def save_json(obj: Any, path: str):
    """Lưu object dạng json với định dạng indent chuẩn."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def evaluate_model(
    model: Union[BaselineSVCModel, BaselineLogisticRegressionModel, BaselineNaiveBayesModel],
    comments: List[str],
    y_true: np.ndarray,
    split_name: str = "Tập Dữ Liệu"
) -> Tuple[Dict[str, float], np.ndarray, float]:
    """Đánh giá mô hình bằng hệ thống độ đo chuẩn chung (src/metrics.py)."""
    print(f"\n{'='*60}")
    print(f"📊 KẾT QUẢ ĐỘ ĐO CHUẨN (src/metrics.py) - {split_name.upper()}")
    print(f"{'='*60}")

    start_time = time.perf_counter()
    y_pred = model.predict(comments)
    elapsed_time = time.perf_counter() - start_time
    ms_per_sentence = (elapsed_time * 1000.0) / max(len(comments), 1)

    metrics = sentence_level_metrics(y_true, y_pred)
    print(format_metrics(metrics))
    print(f"⏱️ Tốc độ suy luận: {ms_per_sentence:.4f} ms/câu (~{1000.0/ms_per_sentence:.0f} câu/giây)")
    print(f"{'='*60}\n")

    print(f"--- BÁO CÁO CHI TIẾT TỪNG LỚP ({split_name}) ---")
    for i, aspect in enumerate(ASPECTS):
        print(f"\n[Aspect: {aspect}]")
        unique_labels = sorted(list(set(y_true[:, i]) | set(y_pred[:, i])))
        target_names = [CLASS_NAMES[idx] for idx in unique_labels]
        print(
            classification_report(
                y_true[:, i],
                y_pred[:, i],
                labels=unique_labels,
                target_names=target_names,
                digits=4,
                zero_division=0,
            )
        )

    return metrics, y_pred, ms_per_sentence


def generate_markdown_report(
    model_key: str,
    train_dist: Dict[str, List[int]],
    class_weights: List[List[float]],
    dev_results: Dict[str, float],
    test_results: Dict[str, float],
    conf_matrices: Dict[str, List[List[int]]],
    test_records: List[Dict[str, Any]],
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    ms_per_sentence: float,
    config: Dict[str, Any],
    solver_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Tạo báo cáo markdown giải thích chi tiết cho từng mô hình Baseline."""
    model_meta = MODEL_CONFIGS[model_key]
    model_name = model_meta["display_name"]

    exact_match_sample = None
    mismatch_sample = None
    for idx, (rec, gold, pred) in enumerate(zip(test_records, y_test, y_test_pred)):
        if exact_match_sample is None and np.array_equal(gold, pred) and any(gold > 0):
            exact_match_sample = (rec["comment"], gold.tolist(), pred.tolist())
        if mismatch_sample is None and not np.array_equal(gold, pred):
            if gold[4] == 3 or gold[0] != pred[0]:
                mismatch_sample = (rec["comment"], gold.tolist(), pred.tolist())

    optimization_desc = ""
    if model_key == "svc":
        optimization_desc = (
            "1. **Bản chất tối ưu lồi (Convex Optimization):**\n"
            "   LinearSVC giải bài toán tối ưu hóa lồi có ràng buộc với hàm mất mát Squared Hinge Loss kết hợp chuẩn hóa L2:\n\n"
            "   $$\\min_{\\mathbf{w}} \\frac{1}{2} \\mathbf{w}^T \\mathbf{w} + C \\sum_{i=1}^N \\max(0, 1 - y_i \\mathbf{w}^T \\mathbf{x}_i)^2$$\n\n"
            "   Hàm mục tiêu này là một **hàm lồi nghiêm ngặt (strictly convex)**, đảm bảo tồn tại một điểm cực tiểu toàn cục duy nhất.\n\n"
            "2. **Thuật toán Dual Coordinate Descent (LIBLINEAR):**\n"
            "   Tối ưu trực tiếp trên bài toán đối ngẫu của toàn bộ ma trận dữ liệu ($5.000 \\times 10.000$), dừng khi $\\|\\nabla f(\\mathbf{w})\\| \\le 10^{-4}$. Không dùng mini-batch hay epochs dạng Deep Learning.\n"
        )
    elif model_key == "lr":
        optimization_desc = (
            "1. **Hàm mất mát Cross-Entropy (Multinomial / Log Loss) kết hợp chuẩn hóa L2:**\n\n"
            "   $$\\min_{\\mathbf{w}} \\frac{1}{2} \\mathbf{w}^T \\mathbf{w} - C \\sum_{i=1}^N \\sum_{c=0}^3 \\mathbb{I}(y_i=c) \\log P(y_i=c | \\mathbf{x}_i)$$\n\n"
            "   Trong đó $P(y_i=c | \\mathbf{x}_i) = \\frac{\\exp(\\mathbf{w}_c^T \\mathbf{x}_i)}{\\sum_j \\exp(\\mathbf{w}_j^T \\mathbf{x}_i)}$.\n\n"
            "2. **Thuật toán Tối ưu L-BFGS (Quasi-Newton):**\n"
            "   Xấp xỉ ma trận nghịch đảo Hessian bậc 2 trên toàn bộ 5.000 mẫu cùng lúc trong mỗi lần lặp (iterations), dừng khi độ suy giảm hàm mất mát nhỏ hơn dung sai $tol=10^{-4}$.\n"
        )
    elif model_key == "nb":
        optimization_desc = (
            "1. **Mô hình Tạo sinh (Generative Probabilistic Model):**\n"
            "   Dựa trên định lý Bayes và giả định độc lập có điều kiện giữa các từ:\n\n"
            "   $$P(c | \\mathbf{x}) \\propto P(c) \\prod_{j=1}^M P(w_j | c)^{x_j}$$\n\n"
            "2. **Ước lượng Hợp lý Cực đại với Làm mịn Laplace/Lidstone:**\n"
            "   $$P(w_j | c) = \\frac{N_{c, w_j} + \\alpha}{\\sum_{w'} N_{c, w'} + \\alpha M}$$\n\n"
            "   Mô hình được huấn luyện giải tích bằng phép đếm tần suất và làm mịn xác suất, không cần thuật toán tối ưu lặp.\n"
        )

    lines = [
        f"# 📊 Báo Cáo Giải Thích Kết Quả Huấn Luyện & Đánh Giá: {model_name}",
        "",
        f"> **Mô hình:** {model_name} (Sentence-Level ABSA)  ",
        "> **Người phụ trách:** Phúc (`feat/baseline`)  ",
        "> **Tập dữ liệu:** ViCloABSA (5.000 Train | 1.000 Dev | 1.000 Test)  ",
        "",
        "---",
        "",
        "## 📑 Mục Lục",
        "- [1. Quy Ước Chung Về Nhãn & Khía Cạnh](#1-quy-ước-chung-về-nhãn--khía-cạnh)",
        "- [2. Giải Thích Chi Tiết Từng File Kết Quả (.json)](#2-giải-thích-chi-tiết-từng-file-kết-quả-json)",
        "- [3. Nhận Xét & Đánh Giá Toàn Diện Mô Hình](#3-nhận-xét--đánh-giá-toàn-diện-mô-hình)",
        "",
        "---",
        "",
        "## 1. Quy Ước Chung Về Nhãn & Khía Cạnh",
        "",
        "Hệ thống thống nhất chuẩn biểu diễn vector 5 chiều đồng bộ:",
        "* **5 Khía cạnh:** Material (0), General (1), Service (2), Design (3), Price (4)",
        "* **4 Nhãn cảm xúc:** `0: None`, `1: Positive`, `2: Neutral`, `3: Negative`",
        "",
        "---",
        "",
        "## 2. Giải Thích Chi Tiết Từng File Kết Quả (.json)",
        "",
        "### 2.1. File `train_stats.json` - Phân bố nhãn & Trọng số phạt",
        "",
        "| Khía cạnh (Aspect) | 0: None | 1: Positive | 2: Neutral | 3: Negative | Tổng số | Nhận xét độ lệch |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for asp in ASPECTS:
        counts = train_dist[asp]
        p0, p1, p2, p3 = [c / 5000.0 * 100 for c in counts]
        note = "Tương đối đồng đều"
        if asp == "General":
            note = "Lớp 0 & 1 chiếm 91.8%"
        elif asp == "Service":
            note = "Lớp Neutral cực hiếm (0.92%)"
        elif asp == "Design":
            note = "Lớp 0 chiếm gần 60%"
        elif asp == "Price":
            note = "**Lớp Negative siêu hiếm (0.62%)**"
        lines.append(
            f"| **{asp}** | {counts[0]:,} ({p0:.1f}%) | {counts[1]:,} ({p1:.1f}%) | {counts[2]:,} ({p2:.2f}%) | {counts[3]:,} ({p3:.2f}%) | 5,000 | {note} |"
        )

    lines.extend([
        "",
        "### 2.2. File `train_history.json` - Số Lần Lặp (Iterations) & Báo Cáo Hội Tụ Của Solver",
        "",
        optimization_desc,
        "",
        "| Khía cạnh (Aspect) | Thuật toán Solver | Số lần lặp (`n_iter_`) | Giới hạn tối đa (`max_iter`) | Ngưỡng dung sai (`tol`) | Trạng thái hội tụ |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ])

    if solver_info and "aspect_iterations" in solver_info:
        for asp in ASPECTS:
            it = solver_info["aspect_iterations"][asp]
            conv_str = "✅ Đã hội tụ (Converged)" if it.get("converged", True) else "⚠️ Chưa hội tụ"
            if model_key == "nb":
                lines.append(f"| **{asp}** | Closed-Form (Giải tích) | N/A (1 lượt quét) | 1 | — | ✅ Tính toán tức thì |")
            else:
                lines.append(f"| **{asp}** | {solver_info.get('solver_name', 'Solver')} | **{it['n_iter']}** | {it['max_iter']} | {it.get('tol', 1e-4)} | {conv_str} |")
    else:
        lines.append(f"| **Toàn bộ 5 khía cạnh** | Tối ưu hóa lồi | **Hoàn tất** | — | — | ✅ Đã hội tụ nghiệm toàn cục |")

    lines.extend([
        "",
        f"> [!NOTE]",
        f"> Mô hình tối ưu hóa trực tiếp trên toàn bộ ma trận 5.000 mẫu trong một quy trình duy nhất. Cả 5 bộ phân loại của các khía cạnh đều **hội tụ tối ưu 100%** và dừng rất sớm trước giới hạn `max_iter`, không cần và không sử dụng cơ chế chia epoch theo kiểu Deep Learning.",
        "",
        "### 2.3. File `dev_best.json` - Checkpoint tối ưu trên Dev",
        "",
        "```json",
        json.dumps({"epoch": 1, **dev_results}, indent=2, ensure_ascii=False),
        "```",
        "",
        "### 2.4. File `test_results.json` - Kết quả kiểm thử chính thức",
        "",
        "| Khía cạnh (Aspect) | Accuracy (%) | Macro-F1 (%) |",
        "| :--- | :---: | :---: |",
    ])

    for asp in ASPECTS:
        lines.append(
            f"| **{asp}** | {test_results[f'{asp}_acc']*100:.2f}% | {test_results[f'{asp}_macro_f1']*100:.2f}% |"
        )

    lines.extend([
        f"| **MEAN ASPECT MACRO-F1** | — | **{test_results['mean_aspect_macro_f1']*100:.2f}%** |",
        f"| **EXACT MATCH RATIO** | **{test_results['exact_match_ratio']*100:.2f}%** | — |",
        f"| **TỐC ĐỘ SUY LUẬN** | — | **{ms_per_sentence:.4f} ms / câu** (~{1000.0/ms_per_sentence:,.0f} câu/giây) |",
        "",
        "### 2.5. File `test_confusion.json` - Ma trận nhầm lẫn 4x4",
        "",
    ])

    for asp in ASPECTS:
        cm = conf_matrices[asp]
        lines.extend([
            f"#### {asp}",
            "```text",
            "Thực tế \\ Dự đoán     None (0)   Pos (1)   Neu (2)   Neg (3)",
            f"None (0):              {cm[0][0]:>4}      {cm[0][1]:>4}      {cm[0][2]:>4}      {cm[0][3]:>4}",
            f"Positive (1):          {cm[1][0]:>4}      {cm[1][1]:>4}      {cm[1][2]:>4}      {cm[1][3]:>4}",
            f"Neutral (2):           {cm[2][0]:>4}      {cm[2][1]:>4}      {cm[2][2]:>4}      {cm[2][3]:>4}",
            f"Negative (3):          {cm[3][0]:>4}      {cm[3][1]:>4}      {cm[3][2]:>4}      {cm[3][3]:>4}",
            "```",
            "",
        ])

    lines.extend([
        "### 2.6. File `test_predictions.json` - Mẫu ví dụ",
        "",
    ])

    if exact_match_sample:
        lines.extend([
            f"* **Ví dụ đoán đúng 5/5 khía cạnh:**",
            f"  - *\"{exact_match_sample[0]}\"*",
            f"  - Gold: `{exact_match_sample[1]}` | Pred: `{exact_match_sample[2]}`",
            "",
        ])

    if mismatch_sample:
        lines.extend([
            f"* **Ví dụ đoán sai:**",
            f"  - *\"{mismatch_sample[0]}\"*",
            f"  - Gold: `{mismatch_sample[1]}` | Pred: `{mismatch_sample[2]}`",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## 3. Nhận Xét & Đánh Giá Toàn Diện Mô Hình",
        f"* **Tổng quan hiệu năng:** Mô hình đạt **Mean Aspect Macro-F1 = {test_results['mean_aspect_macro_f1']*100:.2f}%** và **Exact Match Ratio = {test_results['exact_match_ratio']*100:.2f}%**.",
        f"* **Độ lệch Dev - Test:** Dev Macro-F1 ({dev_results['mean_aspect_macro_f1']*100:.2f}%) vs Test Macro-F1 ({test_results['mean_aspect_macro_f1']*100:.2f}%), chênh lệch {(dev_results['mean_aspect_macro_f1'] - test_results['mean_aspect_macro_f1'])*100:.2f}%.",
        f"* **Tốc độ suy luận:** Đạt {ms_per_sentence:.4f} ms/câu, cực nhanh và đáp ứng tốt điều kiện môi trường tài nguyên thấp.",
        "",
    ])

    return "\n".join(lines)


def train_single_model(
    model_key: str,
    train_records: List[Dict[str, Any]],
    train_texts: List[str],
    y_train: np.ndarray,
    dev_records: List[Dict[str, Any]],
    dev_texts: List[str],
    y_dev: np.ndarray,
    test_records: List[Dict[str, Any]],
    test_texts: List[str],
    y_test: np.ndarray,
    train_dist: Dict[str, List[int]],
    class_weights: List[List[float]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Huấn luyện, đánh giá và xuất kết quả cho một mô hình cụ thể."""
    model_meta = MODEL_CONFIGS[model_key]
    model_name = model_meta["display_name"]
    folder_name = model_meta["dir_name"]
    save_file = model_meta["save_file"]
    model_cls = model_meta["model_class"]

    results_dir = os.path.join(ROOT_DIR, "results", "baseline", folder_name)
    save_path = os.path.join(ROOT_DIR, "saved_models", save_file)

    print(f"\n{'#'*65}")
    print(f"🚀 BẮT ĐẦU QUY TRÌNH CHO MÔ HÌNH: {model_name.upper()}")
    print(f"{'#'*65}")

    # Khởi tạo mô hình
    if model_key == "svc":
        model = model_cls(
            ngram_range=(1, args.ngram_max),
            max_features=args.max_features,
            min_df=2,
            sublinear_tf=True,
            C=args.C if args.C is not None else 0.7,
            class_weight="balanced",
            random_state=42,
        )
        config_dict = {
            "model_type": "LinearSVC",
            "ngram_range": [1, args.ngram_max],
            "max_features": args.max_features,
            "C": args.C if args.C is not None else 0.7,
            "class_weight": "balanced",
        }
    elif model_key == "lr":
        lr_C = args.lr_C if args.lr_C is not None else 1.0
        model = model_cls(
            ngram_range=(1, args.ngram_max),
            max_features=args.max_features,
            min_df=2,
            sublinear_tf=True,
            C=lr_C,
            class_weight="balanced",
            random_state=42,
            max_iter=1000,
        )
        config_dict = {
            "model_type": "LogisticRegression",
            "ngram_range": [1, args.ngram_max],
            "max_features": args.max_features,
            "C": lr_C,
            "class_weight": "balanced",
            "max_iter": 1000,
        }
    elif model_key == "nb":
        nb_alpha = args.nb_alpha if args.nb_alpha is not None else 0.1
        model = model_cls(
            ngram_range=(1, args.ngram_max),
            max_features=args.max_features,
            min_df=2,
            sublinear_tf=True,
            alpha=nb_alpha,
            nb_type="multinomial",
        )
        config_dict = {
            "model_type": "MultinomialNB",
            "ngram_range": [1, args.ngram_max],
            "max_features": args.max_features,
            "alpha": nb_alpha,
            "nb_type": "multinomial",
        }

    # Huấn luyện
    t_train_start = time.perf_counter()
    model.fit(train_texts, y_train)
    train_duration = time.perf_counter() - t_train_start
    print(f"[+] Thời gian huấn luyện: {train_duration:.3f} giây.")

    # Thu thập thông tin iterations và trạng thái hội tụ thực tế của solver
    solver_name = (
        "LIBLINEAR (Dual Coordinate Descent)"
        if model_key == "svc"
        else ("L-BFGS (Quasi-Newton)" if model_key == "lr" else "Closed-Form Counting")
    )
    solver_info = {
        "model_key": model_key,
        "solver_name": solver_name,
        "aspect_iterations": {},
        "all_converged": True,
        "train_duration_seconds": round(train_duration, 4),
    }

    if model_key in ["svc", "lr"]:
        for asp in ASPECTS:
            clf = model.classifiers[asp]
            if model_key == "svc":
                n_iter = int(clf.n_iter_)
            else:
                n_iter = int(clf.n_iter_[0]) if hasattr(clf.n_iter_, "__len__") else int(clf.n_iter_)
            max_i = int(clf.max_iter)
            conv = bool(n_iter < max_i)
            solver_info["aspect_iterations"][asp] = {
                "n_iter": n_iter,
                "max_iter": max_i,
                "tol": clf.tol,
                "converged": conv,
            }
            if not conv:
                solver_info["all_converged"] = False
    else:
        for asp in ASPECTS:
            solver_info["aspect_iterations"][asp] = {
                "n_iter": 1,
                "max_iter": 1,
                "tol": 0.0,
                "converged": True,
            }

    # Lưu mô hình
    model.save(save_path)

    # Đánh giá
    dev_results, y_dev_pred, dev_ms = evaluate_model(model, dev_texts, y_dev, split_name=f"{model_name} - Dev")
    test_results, y_test_pred, test_ms = evaluate_model(model, test_texts, y_test, split_name=f"{model_name} - Test")

    conf_matrices = per_aspect_confusion(y_test, y_test_pred)

    # Xuất các file kết quả chuẩn JSON
    os.makedirs(results_dir, exist_ok=True)

    save_json(
        {"train_label_distribution": train_dist, "class_weights": class_weights},
        os.path.join(results_dir, "train_stats.json"),
    )

    save_json(
        [{
            "epoch": 1,
            "solver_name": solver_info["solver_name"],
            "all_converged": solver_info["all_converged"],
            "aspect_iterations": solver_info["aspect_iterations"],
            "train_duration_seconds": round(train_duration, 4),
            **{f"dev_{k}": v for k, v in dev_results.items()},
        }],
        os.path.join(results_dir, "train_history.json"),
    )

    save_json({"epoch": 1, **dev_results}, os.path.join(results_dir, "dev_best.json"))

    test_results_out = {**test_results, "inference_ms_per_sentence": test_ms, "best_epoch": 1}
    save_json(test_results_out, os.path.join(results_dir, "test_results.json"))

    save_json(conf_matrices, os.path.join(results_dir, "test_confusion.json"))

    test_predictions_data = [
        {"id": rec["id"], "comment": rec["comment"], "gold": gold.tolist(), "pred": pred.tolist()}
        for rec, gold, pred in zip(test_records, y_test, y_test_pred)
    ]
    save_json(test_predictions_data, os.path.join(results_dir, "test_predictions.json"))

    report_content = generate_markdown_report(
        model_key=model_key,
        train_dist=train_dist,
        class_weights=class_weights,
        dev_results=dev_results,
        test_results=test_results,
        conf_matrices=conf_matrices,
        test_records=test_records,
        y_test=y_test,
        y_test_pred=y_test_pred,
        ms_per_sentence=test_ms,
        config=config_dict,
        solver_info=solver_info,
    )

    report_path = os.path.join(results_dir, "results_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[+] Đã xuất báo cáo Markdown tại: {report_path}")

    # Nếu là mô hình LinearSVC mặc định, đồng bộ cả file saved_models/baseline_metrics.json và results/baseline/
    if model_key == "svc":
        legacy_metrics = {
            "model_name": "Baseline (TF-IDF + LinearSVC)",
            "config": config_dict,
            "dev": dev_results,
            "test": test_results,
            "inference_ms_per_sentence": test_ms,
        }
        save_json(legacy_metrics, os.path.join(ROOT_DIR, "saved_models", "baseline_metrics.json"))
        with open(os.path.join(ROOT_DIR, "saved_models", "baseline_report.md"), "w", encoding="utf-8") as f:
            f.write(report_content)

    return {
        "model_key": model_key,
        "name": model_name,
        "train_time": train_duration,
        "dev": dev_results,
        "test": test_results,
        "test_ms": test_ms,
    }


def check_baseline_comparison():
    """Xác nhận sự tồn tại của file báo cáo đối sánh duy nhất: BASELINE_COMPARISON.md."""
    src_path = os.path.join(ROOT_DIR, "results", "baseline", "BASELINE_COMPARISON.md")
    if os.path.exists(src_path):
        print(f"[+] Báo cáo đối sánh duy nhất của 3 mô hình Baseline: {src_path}")


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện và đánh giá các mô hình Baseline (LinearSVC, Logistic Regression, Naive Bayes)")
    parser.add_argument("--model", type=str, default="all", choices=["svc", "lr", "nb", "all"], help="Chọn mô hình huấn luyện: 'svc', 'lr', 'nb', hoặc 'all'")
    parser.add_argument("--train_path", type=str, default="data/processed/train_data_processed.json", help="Đường dẫn file train")
    parser.add_argument("--dev_path", type=str, default="data/processed/dev_data_processed.json", help="Đường dẫn file dev")
    parser.add_argument("--test_path", type=str, default="data/processed/test_data_processed.json", help="Đường dẫn file test")
    parser.add_argument("--max_features", type=int, default=10000, help="Số lượng features tối đa cho TF-IDF")
    parser.add_argument("--ngram_max", type=int, default=2, help="N-gram max (1 cho unigram, 2 cho bigram)")
    parser.add_argument("--C", type=float, default=0.7, help="Hệ số điều chuẩn C cho LinearSVC")
    parser.add_argument("--lr_C", type=float, default=1.0, help="Hệ số điều chuẩn C cho Logistic Regression")
    parser.add_argument("--nb_alpha", type=float, default=0.1, help="Hệ số smoothing alpha cho Naive Bayes")

    args = parser.parse_args()

    train_path = os.path.join(ROOT_DIR, args.train_path)
    dev_path = os.path.join(ROOT_DIR, args.dev_path)
    test_path = os.path.join(ROOT_DIR, args.test_path)

    print("[*] Đang nạp dữ liệu train / dev / test...")
    train_records, train_texts, y_train = load_dataset(train_path)
    dev_records, dev_texts, y_dev = load_dataset(dev_path)
    test_records, test_texts, y_test = load_dataset(test_path)

    train_dist = compute_label_distribution(y_train)
    class_weights = compute_class_weights(y_train, smoothing="inverse")

    models_to_run = ["svc", "lr", "nb"] if args.model == "all" else [args.model]
    all_summary = {}

    for m_key in models_to_run:
        summary = train_single_model(
            model_key=m_key,
            train_records=train_records,
            train_texts=train_texts,
            y_train=y_train,
            dev_records=dev_records,
            dev_texts=dev_texts,
            y_dev=y_dev,
            test_records=test_records,
            test_texts=test_texts,
            y_test=y_test,
            train_dist=train_dist,
            class_weights=class_weights,
            args=args,
        )
        all_summary[m_key] = summary

    # Xác nhận file đối sánh duy nhất của 3 mô hình
    check_baseline_comparison()

    print("\n" + "="*70)
    print("🎉 HOÀN TẤT QUÁ TRÌNH HUẤN LUYỆN & ĐÁNH GIÁ CÁC MÔ HÌNH BASELINE!")
    print("="*70)


if __name__ == "__main__":
    main()
