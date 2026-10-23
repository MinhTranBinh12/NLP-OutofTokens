# Thí nghiệm 3: PhoBERT-base-v2 + Tiền Xử Lý Chuẩn Hóa Toàn Diện

## 📌 Mục tiêu
Đánh giá tác động độc lập của các bước Tiền xử lý nâng cao (Text Normalization) đối với mô hình PhoBERT-base-v2:
1. Chuẩn hóa Teencode & từ viết tắt phổ biến (`sp`, `sz`, `ko/k`, `đc`, `mik`,...).
2. Bóc tách bẫy giá tiền `92k`, `150k` $\rightarrow$ `92 nghìn`, `150 nghìn`.
3. Rút gọn ký tự lặp kéo dài cảm xúc (`đẹpppp -> đẹp`, `quáaaa -> quá`).
4. Ánh xạ Emoji phổ biến sang từ ngữ biểu cảm tiếng Việt (`❤️ -> yêu thích`, `👍 -> tốt hài lòng`).

## 🚀 Câu lệnh chạy trên Google Colab GPU
```bash
!python scripts/train_multihead.py \
    --exp_name exp3_phobert_normalized \
    --model_name vinai/phobert-base-v2 \
    --preprocess_mode normalized \
    --epochs 10 \
    --do_test
```

## 📂 Các file kết quả dự kiến sinh ra:
- `test_results.json`
- `train_stats.json`
- `train_history.json`
- `dev_best.json`
- `test_confusion.json`
- `test_predictions.json`
