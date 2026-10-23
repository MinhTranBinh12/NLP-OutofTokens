# 📊 Hệ Thống Thí Nghiệm & Kết Quả Mô Hình 2 (PhoBERT Multi-Head)

> **Người phụ trách:** Minh (`feat/multihead`)  
> **Nhiệm vụ:** Phân tích Cảm xúc Đa khía cạnh cấp độ toàn câu (Sentence-Level ABSA)  
> **Không gian nhãn:** Vector 5 chiều $\in \{0, 1, 2, 3\}^5$ (5 Khía cạnh $\times$ 4 Trạng thái)  

---

## 📂 Cấu Trúc Thư Mục Kết Quả (Không Xung Đột)

Mọi file kết quả của Mô hình 2 được đặt trọn vẹn trong `results/multihead/` để tránh hoàn toàn xung đột với Thành viên 1 (`results/baseline/`) và Thành viên 3 (`results/crf/`):

```text
results/multihead/
├── README.md                          # Tài liệu hướng dẫn & giải thích tổng quan
├── exp1_phobert_baseline/             # [ĐÃ HOÀN THÀNH] PhoBERT-base-v2 + Minimal Preprocessing
│   ├── test_results.json              # Điểm số chính thức: Mean F1 = 75.38%, EMR = 57.90%
│   ├── dev_best.json                  # Best Checkpoint tại Epoch 9 (Dev Mean F1 = 75.21%)
│   ├── train_history.json             # Nhật ký Loss và F1 qua 10 Epochs
│   ├── train_stats.json               # Phân bố nhãn và trọng số phạt nghịch đảo
│   └── test_confusion.json            # Ma trận nhầm lẫn 4x4 từng khía cạnh
├── exp2_visobert/                     # [SẴN SÀNG CHẠY] ViSoBERT-base + Minimal Preprocessing
│   └── README.md                      # Hướng dẫn chạy và câu lệnh Colab
└── exp3_phobert_normalized/           # [SẴN SÀNG CHẠY] PhoBERT-base-v2 + Full Preprocessing (Teencode/Emoji)
    └── README.md                      # Hướng dẫn chạy và câu lệnh Colab
```

---

## 🏆 Bảng So Sánh Hiệu Năng Các Pipeline Thí Nghiệm

| Pipeline | Model Pre-trained | Tiền xử lý | Mean F1 (%) | Exact Match (%) | Tốc độ (ms/câu) | Trạng thái |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **exp1_phobert_baseline** | `vinai/phobert-base-v2` | `minimal` | **75.38%** | **57.90%** | **1.27** | ✅ Hoàn thành |
| **exp2_visobert** | `uitnlp/visobert-base` | `minimal` | *Sắp chạy* | *Sắp chạy* | *Sắp chạy* | ⏳ Sẵn sàng |
| **exp3_phobert_normalized** | `vinai/phobert-base-v2` | `normalized` | *Sắp chạy* | *Sắp chạy* | *Sắp chạy* | ⏳ Sẵn sàng |

---

## ⚡ Lệnh Tự Động So Sánh & Sinh Báo Cáo
Sau khi hoàn tất bất kỳ thí nghiệm nào, chỉ cần chạy từ thư mục gốc dự án:
```bash
python scripts/compare_experiments.py
```
Hệ thống sẽ tự động quét tất cả các file `test_results.json`, in bảng đối sánh ra màn hình console và sinh mã nguồn bảng LaTeX sẵn sàng để dán vào `report.tex`.
