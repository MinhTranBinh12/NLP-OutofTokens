# 📈 Báo Cáo Đối Sánh Hiệu Năng Các Pipeline Thí Nghiệm
**Dự án:** Phân tích Cảm xúc Đa khía cạnh (ViCloABSA - Thời trang TMĐT)  
**Mô hình:** Multi-Head Classification (Sentence-level)  
**Người phụ trách:** Minh (`feat/multihead`)  
**Tập dữ liệu:** 5.000 Train | 1.000 Dev | 1.000 Test  

---

## 1. Bảng Tổng Hợp Kết Quả Thực Nghiệm Trên Tập Test (1.000 câu)

| Pipeline / Thí nghiệm | Mô hình (Encoder) | Tiền xử lý | Mean F1 (%) | Exact Match (%) | Material F1 | General F1 | Service F1 | Design F1 | Price F1 | Inference (ms) | Xếp hạng |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **exp1_phobert_baseline** | PhoBERT-base-v2 | `minimal` | 75.38% | **57.90%** | **78.47%** | 66.78% | 77.64% | **74.70%** | 79.32% | **1.27** | 🥈 Hạng 2 |
| **exp2_visobert** | ViSoBERT | `minimal` | 72.63% | 52.60% | 73.04% | 65.23% | 77.97% | 69.12% | 77.81% | 1.67 | 🥉 Hạng 3 |
| **exp3_phobert_normalized** | PhoBERT-base-v2 | `normalized` | **77.35%** | 57.80% | 77.78% | **69.94%** | **82.45%** | 72.90% | **83.67%** | 1.41 | 🥇 **VÔ ĐỊCH** |

> *(Ghi chú: Giá trị in đậm là kết quả tốt nhất ở từng tiêu chí đánh giá)*

---

## 2. Phân Tích & Đánh Giá Đóng Góp Khoa Học (Ablation Study)

### 2.1. Đóng góp của Tiền xử lý Chuẩn hóa (Exp 1 vs Exp 3)
* Khi áp dụng pipeline tiền xử lý nâng cao (chuẩn hóa Teencode, giải mã Emoji, bóc tách bẫy giá tiền `k`), **Mean Aspect Macro-F1 tăng vọt từ 75.38% lên 77.35% (+1.97%)**.
* **Khía cạnh Service tăng mạnh nhất (+4.81%):** Nhờ chuẩn hóa các từ ngữ giao hàng/đóng gói (`ship`, `sop`, `tks`, `cskh`) và emoji (`👍`, `❤️`), F1 đạt **82.45%**.
* **Khía cạnh Price tăng vượt bậc (+4.35%):** Việc xử lý bẫy chữ `k` (`150k -> 150 nghìn`) giúp giải quyết triệt để sự nhập nhằng giữa chữ `k` (không) và `k` (nghìn), đẩy F1 đạt mức kỷ lục **83.67%**.
* **Khía cạnh General tăng +3.16%:** Đạt **69.94%** nhờ các tính từ cảm thán và viết tắt được quy về dạng chuẩn.

### 2.2. So sánh giữa PhoBERT và ViSoBERT (Exp 1 vs Exp 2)
* Mặc dù ViSoBERT được pretrain trên mạng xã hội, PhoBERT-base-v2 vẫn thể hiện sự vượt trội về biểu diễn ngữ pháp và ngữ cảnh thời trang trên tập dữ liệu ViCloABSA (75.38% vs 72.63%).
* Tuy nhiên, ViSoBERT xử lý khía cạnh `Service` rất tốt (77.97%), chứng minh khả năng bắt từ ngữ giao hàng trên mạng xã hội.

---

## 3. Mã Nguồn Bảng LaTeX (Dùng chèn thẳng vào report.tex)

```latex
\begin{table}[htbp]
\centering
\small
\caption{Đối sánh hiệu năng giữa các Pipeline thí nghiệm trên tập Test ViCloABSA}
\label{tab:pipeline_comparison}
\begin{tabular}{llcccc}
\hline
\textbf{Pipeline} & \textbf{Mô hình Encoder} & \textbf{Tiền xử lý} & \textbf{Mean Macro-F1} & \textbf{Exact Match} & \textbf{Tốc độ (ms)} \\
\hline
exp1\_phobert\_baseline & PhoBERT-base-v2 & minimal & 75.38\% & \textbf{57.90\%} & \textbf{1.27} \\
exp2\_visobert & ViSoBERT & minimal & 72.63\% & 52.60\% & 1.67 \\
\textbf{exp3\_phobert\_normalized} & \textbf{PhoBERT-base-v2} & \textbf{normalized} & \textbf{77.35\%} & 57.80\% & 1.41 \\
\hline
\end{tabular}
\end{table}
```
