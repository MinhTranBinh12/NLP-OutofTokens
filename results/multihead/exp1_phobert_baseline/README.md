# Thí nghiệm 1: PhoBERT-base-v2 Baseline (Tiền xử lý tối thiểu)

## 📌 Mục tiêu
Đánh giá mô hình đường cơ sở (Baseline) sử dụng kiến trúc PhoBERT Multi-Head ABSA (`vinai/phobert-base-v2`) với bộ tiền xử lý tối thiểu (`minimal`: chuẩn hóa unicode NFC, khoảng trắng, lower).

## 🚀 Câu lệnh chạy trên Google Colab GPU
```bash
!python scripts/train_multihead.py \
    --exp_name exp1_phobert_baseline \
    --model_name vinai/phobert-base-v2 \
    --preprocess_mode minimal \
    --epochs 10 \
    --do_test
```

## 📂 Các file kết quả:
- `test_results.json`: Điểm F1 từng khía cạnh, Mean Aspect Macro-F1 (75.38%), EMR (57.90%).
- `train_stats.json`: Phân bố nhãn và cấu hình siêu tham số.
- `train_history.json`: Quá trình loss và F1 qua từng epoch.
- `dev_best.json`: Thông số checkpoint đạt đỉnh trên tập Dev.
- `test_confusion.json`: Ma trận nhầm lẫn 4x4 cho từng khía cạnh.
- `test_predictions.json`: Chi tiết dự đoán trên 1.000 câu tập Test (đã chuẩn hóa UTF-8 đầy đủ).
