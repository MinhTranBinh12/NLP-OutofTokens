# 📊 Hệ Thống Thí Nghiệm & Kết Quả Mô Hình 2 (PhoBERT Multi-Head)

> **Người phụ trách:** Minh (`feat/multihead`)  
> **Nhiệm vụ:** Phân tích Cảm xúc Đa khía cạnh cấp độ toàn câu (Sentence-Level ABSA)  
> **Tập dữ liệu:** ViCloABSA (5.000 Train | 1.000 Dev | 1.000 Test)  
> **Không gian nhãn:** Vector 5 chiều $\in \{0, 1, 2, 3\}^5$ (5 Khía cạnh $\times$ 4 Trạng thái)  

---

## 🏆 Bảng Tổng Hợp Kết Quả Thực Nghiệm (Tập Test)

| Pipeline | Model Pre-trained | Tiền xử lý | Mean F1 (%) | Exact Match (%) | Tốc độ (ms) | Xếp hạng |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **exp1_phobert_baseline** | `vinai/phobert-base-v2` | `minimal` | 75.38% | **57.90%** | **1.27** | 🥈 Hạng 2 |
| **exp2_visobert** | `uitnlp/visobert` | `minimal` | 72.63% | 52.60% | 1.67 | 🥉 Hạng 3 |
| **exp3_phobert_normalized** | `vinai/phobert-base-v2` | `normalized` | **77.35%** | 57.80% | 1.41 | 🥇 **VÔ ĐỊCH (+1.97%)** |

---

## 📂 Cấu Trúc Thư Mục Kết Quả Đã Chuẩn Hóa

Toàn bộ kết quả được tổ chức độc quyền trong `results/multihead/`, đảm bảo 0% xung đột khi gộp nhánh vào `master`:

```text
results/
├── comparison_report.md               # Báo cáo đối sánh chi tiết kèm mã LaTeX cho báo cáo
└── multihead/
    ├── README.md                      # Tài liệu tổng quan này
    ├── exp1_phobert_baseline/         # PhoBERT-base-v2 + Minimal Preprocessing (F1 = 75.38%)
    │   ├── test_results.json
    │   ├── dev_best.json
    │   ├── train_history.json
    │   ├── train_stats.json
    │   └── test_confusion.json
    ├── exp2_visobert/                 # ViSoBERT + Minimal Preprocessing (F1 = 72.63%)
    │   ├── test_results.json
    │   ├── dev_best.json
    │   ├── train_history.json
    │   ├── train_stats.json
    │   └── test_confusion.json
    └── exp3_phobert_normalized/       # PhoBERT-base-v2 + Full Preprocessing (F1 = 77.35%)
        ├── test_results.json
        ├── dev_best.json
        ├── train_history.json
        ├── train_stats.json
        └── test_confusion.json
```
