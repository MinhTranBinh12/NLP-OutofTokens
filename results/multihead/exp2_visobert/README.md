# Thí nghiệm 2: ViSoBERT-base (Mạng xã hội & TMĐT)

## 📌 Mục tiêu
Đánh giá năng lực của mô hình `uitnlp/visobert-base` (được huấn luyện trước trên dữ liệu bình luận mạng xã hội & Shopee/Lazada) so với PhoBERT-base-v2 (học từ báo chí tiếng Việt).

## 🚀 Câu lệnh chạy trên Google Colab GPU
```bash
!python scripts/train_multihead.py \
    --exp_name exp2_visobert \
    --model_name uitnlp/visobert-base \
    --preprocess_mode minimal \
    --epochs 10 \
    --do_test
```

## 📂 Các file kết quả dự kiến sinh ra:
- `test_results.json`: Điểm F1 từng khía cạnh, Mean Aspect Macro-F1, Exact Match Ratio, tốc độ suy luận.
- `train_stats.json`: Phân bố nhãn và cấu hình siêu tham số.
- `train_history.json`: Quá trình loss và F1 qua từng epoch.
- `dev_best.json`: Thông số checkpoint đạt đỉnh trên tập Dev.
- `test_confusion.json`: Ma trận nhầm lẫn 4x4 cho từng khía cạnh.
- `test_predictions.json`: Chi tiết dự đoán trên 1.000 câu tập Test.
