"""
Script tổng hợp và so sánh kết quả giữa các Pipeline thí nghiệm.
Người phụ trách: Minh (nhánh feat/multihead).

Cách chạy:
    python scripts/compare_experiments.py

Đầu ra:
    - In bảng đối sánh chi tiết ra màn hình console.
    - Lưu báo cáo so sánh vào results/comparison_report.md.
    - Sinh mã LaTeX cho bảng đối sánh để dán vào report.tex.
"""

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULTS_DIR = os.path.join(ROOT, "results")
ASPECTS = ["Material", "General", "Service", "Design", "Price"]


def find_all_experiment_results():
    pattern = os.path.join(RESULTS_DIR, "**", "test_results.json")
    files = glob.glob(pattern, recursive=True)
    experiments = []

    for f in sorted(files):
        # Xác định tên thí nghiệm từ thư mục cha
        exp_dir = os.path.dirname(f)
        rel_dir = os.path.relpath(exp_dir, RESULTS_DIR).replace("\\", "/")
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            experiments.append({"path": f, "rel_dir": rel_dir, "data": data})
        except Exception as e:
            print(f"[!] Lỗi khi đọc {f}: {e}")
    return experiments


def build_markdown_table(experiments):
    if not experiments:
        return "Chưa có kết quả thí nghiệm nào trong thư mục `results/`."

    headers = [
        "Pipeline / Thí nghiệm",
        "Mô hình (Encoder)",
        "Tiền xử lý",
        "Mean F1 (%)",
        "Exact Match (%)",
        "Material F1",
        "General F1",
        "Service F1",
        "Design F1",
        "Price F1",
        "Inference (ms)",
    ]

    # Tìm giá trị cao nhất cho từng cột F1
    best_mean_f1 = max(exp["data"].get("mean_aspect_macro_f1", 0) for exp in experiments)
    best_emr = max(exp["data"].get("exact_match_ratio", 0) for exp in experiments)
    best_aspect_f1 = {
        asp: max(exp["data"].get(f"{asp}_macro_f1", 0) for exp in experiments)
        for asp in ASPECTS
    }

    rows = []
    for exp in experiments:
        d = exp["data"]
        name = d.get("exp_name") or exp["rel_dir"]
        model = d.get("model_name", "PhoBERT-base-v2").split("/")[-1]
        pre = d.get("preprocess_mode", "minimal")

        mean_f1 = d.get("mean_aspect_macro_f1", 0) * 100
        emr = d.get("exact_match_ratio", 0) * 100
        ms = d.get("inference_ms_per_sentence", 0)

        # Định dạng in đậm giá trị tốt nhất
        f1_str = f"**{mean_f1:.2f}%**" if d.get("mean_aspect_macro_f1", 0) == best_mean_f1 else f"{mean_f1:.2f}%"
        emr_str = f"**{emr:.2f}%**" if d.get("exact_match_ratio", 0) == best_emr else f"{emr:.2f}%"

        asp_strs = []
        for asp in ASPECTS:
            val = d.get(f"{asp}_macro_f1", 0)
            val_pct = val * 100
            if val == best_aspect_f1[asp]:
                asp_strs.append(f"**{val_pct:.2f}%**")
            else:
                asp_strs.append(f"{val_pct:.2f}%")

        rows.append([
            name,
            model,
            pre,
            f1_str,
            emr_str,
            asp_strs[0],
            asp_strs[1],
            asp_strs[2],
            asp_strs[3],
            asp_strs[4],
            f"{ms:.2f}",
        ])

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join([":---"] * 3 + [":---:"] * 8) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def build_latex_table(experiments):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Đối sánh hiệu năng giữa các Pipeline thí nghiệm trên tập Test}",
        r"\label{tab:pipeline_comparison}",
        r"\begin{tabular}{llcccc}",
        r"\hline",
        r"\textbf{Pipeline} & \textbf{Mô hình} & \textbf{Tiền xử lý} & \textbf{Mean Macro-F1} & \textbf{Exact Match} & \textbf{Tốc độ (ms)} \\",
        r"\hline",
    ]
    for exp in experiments:
        d = exp["data"]
        name = d.get("exp_name") or exp["rel_dir"]
        model = d.get("model_name", "PhoBERT-base-v2").split("/")[-1]
        pre = d.get("preprocess_mode", "minimal")
        mean_f1 = d.get("mean_aspect_macro_f1", 0) * 100
        emr = d.get("exact_match_ratio", 0) * 100
        ms = d.get("inference_ms_per_sentence", 0)
        lines.append(f"{name} & {model} & {pre} & {mean_f1:.2f}\\% & {emr:.2f}\\% & {ms:.2f} \\\\")
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main():
    experiments = find_all_experiment_results()
    print(f"[*] Tìm thấy {len(experiments)} kết quả thí nghiệm trong 'results/':")
    for e in experiments:
        print(f"    - {e['rel_dir']} (Mean F1: {e['data'].get('mean_aspect_macro_f1', 0)*100:.2f}%)")

    md_table = build_markdown_table(experiments)
    latex_table = build_latex_table(experiments)

    print("\n" + "=" * 80)
    print("📊 BẢNG ĐỐI SÁNH HIỆU NĂNG CÁC PIPELINE:")
    print("=" * 80)
    print(md_table)
    print("=" * 80 + "\n")

    report_content = f"""# 📈 Báo Cáo Đối Sánh Hiệu Năng Các Pipeline Thí Nghiệm
Dự án: Phân tích Cảm xúc Đa khía cạnh (ViCloABSA)
Mô hình: Multi-Head Classification (Sentence-level)

## 1. Bảng Tổng Hợp Kết Quả Thực Nghiệm Trên Tập Test
{md_table}

## 2. Mã Nguồn Bảng LaTeX (Dùng chèn vào report.tex)
```latex
{latex_table}
```
"""
    report_path = os.path.join(RESULTS_DIR, "comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[✓] Đã xuất báo cáo đối sánh tổng hợp tại: {report_path}")


if __name__ == "__main__":
    main()
