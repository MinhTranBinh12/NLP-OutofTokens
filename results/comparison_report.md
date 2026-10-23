# 📈 Báo Cáo Đối Sánh Hiệu Năng Các Pipeline Thí Nghiệm
**Dự án:** Phân tích Cảm xúc Đa khía cạnh (ViCloABSA - Thời trang TMĐT)  
**Mô hình:** Multi-Head Classification (Sentence-level)  
**Người phụ trách:** Minh (`feat/multihead`)  

---

## 1. Bảng Tổng Hợp Kết Quả Thực Nghiệm Trên Tập Test (1.000 câu)

| Pipeline / Thí nghiệm | Mô hình (Encoder) | Tiền xử lý | Mean F1 (%) | Exact Match (%) | Material F1 | General F1 | Service F1 | Design F1 | Price F1 | Inference (ms) | Trạng thái |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **exp1_phobert_baseline** | PhoBERT-base-v2 | `minimal` | **75.38%** | **57.90%** | **78.47%** | **66.78%** | **77.64%** | **74.70%** | **79.32%** | **1.27** | ✅ Đã hoàn thành |
| **exp2_visobert** | ViSoBERT-base | `minimal` | *Sắp có* | *Sắp có* | *Sắp có* | *Sắp có* | *Sắp có* | *Sắp có* | *Sắp có* | *Sắp có* | ⏳ Sẵn sàng chạy |
| **exp3_phobert_normalized** | PhoBERT-base-v2 | `normalized` | *Sắp có* | *Sắp có* | *Sắp có* | *Sắp có* | *Sắp có* | *Sắp có* | *Sắp có* | *Sắp có* | ⏳ Sẵn sàng chạy |

---

## 2. Mã Nguồn Bảng LaTeX (Dùng chèn thẳng vào Bảng 11.3 của report.tex)

```latex
\begin{table}[htbp]
\centering
\small
\caption{Đối sánh hiệu năng giữa các Pipeline thí nghiệm trên tập Test (Sentence-level ABSA)}
\label{tab:pipeline_comparison}
\begin{tabular}{llcccc}
\hline
\textbf{Pipeline} & \textbf{Mô hình Encoder} & \textbf{Tiền xử lý} & \textbf{Mean Macro-F1} & \textbf{Exact Match} & \textbf{Tốc độ (ms)} \\
\hline
exp1\_phobert\_baseline & PhoBERT-base-v2 & minimal & 75.38\% & 57.90\% & 1.27 \\
exp2\_visobert & ViSoBERT-base & minimal & [Sắp có] & [Sắp có] & [Sắp có] \\
exp3\_phobert\_normalized & PhoBERT-base-v2 & normalized & [Sắp có] & [Sắp có] & [Sắp có] \\
\hline
\end{tabular}
\end{table}
```

---

## 3. Hướng Dẫn Tái Hiện & Chạy Thí Nghiệm Tiếp Theo

### Chạy Pipeline 2 (ViSoBERT-base):
```bash
!python scripts/train_multihead.py \
    --exp_name exp2_visobert \
    --model_name uitnlp/visobert-base \
    --preprocess_mode minimal \
    --epochs 10 \
    --do_test
```

### Chạy Pipeline 3 (PhoBERT + Tiền xử lý nâng cao Teencode/Emoji):
```bash
!python scripts/train_multihead.py \
    --exp_name exp3_phobert_normalized \
    --model_name vinai/phobert-base-v2 \
    --preprocess_mode normalized \
    --epochs 10 \
    --do_test
```

### Cập nhật Báo cáo đối sánh sau khi chạy xong:
```bash
python scripts/compare_experiments.py
```
