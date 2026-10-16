# BÁO CÁO TỔNG QUAN & ĐÁNH GIÁ KẾT QUẢ HUẤN LUYỆN (MODEL OVERVIEW)
**Dự án:** ViCloABSA - Phân tích Cảm xúc Đa khía cạnh trong Đánh giá Thời trang Tiếng Việt  
**Mô hình:** PhoBERT-base-v2 + CRF (Span-Level Extraction)  
**Nhánh thực hiện:** `feat/crf` (Thành viên 3)  
**Thời gian hoàn thành:** 12/10/2026  

---

## 📌 1. TỔNG QUAN KIẾN TRÚC MÔ HÌNH

Hệ thống giải quyết bài toán **Span-Level Aspect-Based Sentiment Analysis (ABSA)** theo hướng trích xuất chuỗi nhãn BIO liên hợp:
$$\text{Văn bản đầu vào} \xrightarrow{\text{PhoBERT}} \text{Contextual Embeddings} \xrightarrow{\text{Linear Head}} \text{Emission Scores} \xrightarrow{\text{CRF Layer (Viterbi)}} \text{Chuỗi nhãn tối ưu}$$

* **Cốt lõi ngôn ngữ (Pre-trained LM):** `vinai/phobert-base-v2` (135,023,134 tham số). Mô hình được huấn luyện đặc thù trên tiếng Việt với cơ chế từ ghép (`_`), biểu diễn ngữ cảnh vượt trội.
* **Tầng phân loại & giải mã (Classification & CRF):**
  * Tầng tuyến tính Linear chiếu từ chiều không gian ẩn ($d=768$) sang 31 nhãn.
  * Tầng **Conditional Random Field (CRF)**: Học ma trận chuyển trạng thái nhãn hợp lệ ($31 \times 31$), triệt tiêu các lỗi cú pháp gán nhãn vô lý (ví dụ: chuyển từ nhãn `O` trực tiếp sang nhãn `I-Material#positive` mà không có `B-` đi trước).
* **Không gian 31 nhãn BIO:**
  * `O`: Không thuộc khía cạnh nào.
  * 5 Khía cạnh ($\text{Aspects}$): **Material** (Chất liệu), **Design** (Thiết kế / Kiểu dáng / Form), **Price** (Giá cả), **Service** (Dịch vụ / Giao hàng / Tư vấn), **General** (Đánh giá chung).
  * 3 Cực tính cảm xúc ($\text{Sentiments}$): **positive**, **neutral**, **negative**.
  * Cặp nhãn bắt đầu và tiếp diễn: $B\text{-Aspect\Sentiment}$ và $I\text{-Aspect\Sentiment}$ ($2 \times 5 \times 3 = 30$ nhãn + 1 nhãn $O$).

---

## 📊 2. THỐNG KÊ TẬP DỮ LIỆU & TỈ LỆ PHÂN CHIA

Bộ dữ liệu gồm **7,000 mẫu** đánh giá thực tế thu thập từ các nền tảng thương mại điện tử ngành hàng thời trang:

| Tập dữ liệu | Số lượng mẫu (Samples) | Tỉ lệ (%) | Mục đích sử dụng |
| :--- | :---: | :---: | :--- |
| **Train** | **5,000** | 71.43% | Cập nhật trọng số PhoBERT và tầng CRF |
| **Dev (Validation)** | **1,000** | 14.29% | Đánh giá sau mỗi epoch, chọn checkpoint tốt nhất |
| **Test** | **1,000** | 14.29% | Đánh giá độc lập chất lượng mô hình sau khi train xong |
| **Tổng cộng** | **7,000** | 100.0% | |

---

## ⚙️ 3. THIẾT LẬP THỰC NGHIỆM & SIÊU THAM SỐ (HYPERPARAMETERS)

* **Thiết bị phần cứng:** NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB VRAM).
* **Chế độ tính toán:** Mixed Precision FP16 (`torch.cuda.amp`), tăng tốc độ và tối ưu dung lượng VRAM.
* **Số Epochs:** 15.
* **Batch size:** 16 (train), 32 (eval).
* **Độ dài chuỗi tối đa (`max_length`):** 256 tokens.
* **Chiến lược Tốc độ học vi sai (Differential Learning Rates):**
  * **PhoBERT Encoder LR:** `2e-5` (giữ ổn định các tầng tiền huấn luyện, chống catastrophic forgetting).
  * **Classifier Head & CRF LR:** `1e-4` (tốc độ học cao hơn 5 lần để tầng mới hội tụ nhanh).
* **Bộ tối ưu & Điều hòa:** AdamW, Weight Decay = 0.01, Gradient Clipping = 1.0.
* **Lịch trình học (Scheduler):** Cosine Annealing với Warmup 10% (469 warmup steps trên tổng 4,695 training steps).
* **Tổng thời gian huấn luyện:** **70.62 phút** (~4.7 phút/epoch).

---

## 📈 4. DIỄN BIẾN HUẤN LUYỆN QUA 15 EPOCHS

Checkpoint tốt nhất được lưu tự động tại **Epoch 12** với **Dev Exact F1 đạt 0.4823** và **Dev Proportional F1 đạt 0.7190**.

| Epoch | Train Loss | Dev Loss | Dev Exact F1 | Dev Prop. F1 | Dev Precision | Dev Recall | Thời gian (s) | Ghi chú |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | 81.8582 | 53.3834 | 0.1658 | 0.4824 | 0.1634 | 0.1682 | 277s | Giai đoạn Warmup ban đầu |
| 2 | 43.3133 | 32.8845 | 0.3974 | 0.6424 | 0.3550 | 0.4513 | 282s | Mô hình bắt đầu học ranh giới span |
| 3 | 28.7150 | 28.0565 | 0.3971 | 0.6520 | 0.3482 | 0.4619 | 282s | |
| 4 | 22.5283 | 26.7063 | 0.4141 | 0.6831 | 0.3696 | 0.4709 | 283s | Dev Loss chạm đáy thấp nhất |
| 5 | 18.1733 | 26.7695 | 0.4189 | 0.6755 | 0.3684 | 0.4855 | 277s | |
| 6 | 15.1560 | 27.3161 | 0.4377 | 0.6847 | 0.3955 | 0.4900 | 281s | |
| 7 | 13.3410 | 29.6913 | 0.4549 | 0.6949 | 0.4093 | 0.5120 | 279s | |
| 8 | 11.0411 | 29.4115 | 0.4641 | 0.6974 | 0.4164 | 0.5242 | 281s | |
| 9 | 9.5297 | 30.4398 | 0.4591 | 0.7025 | 0.4143 | 0.5149 | 285s | |
| 10 | 8.3323 | 30.6413 | 0.4784 | 0.7145 | 0.4371 | 0.5283 | 280s | |
| 11 | 7.3627 | 31.3992 | 0.4798 | 0.7138 | 0.4369 | 0.5320 | 279s | |
| **12** | **6.6496** | **31.1911** | **0.4823** | **0.7190** | **0.4406** | **0.5328** | **282s** | **★ Checkpoint Tối Ưu Nhất (Best Model)** |
| 13 | 6.3160 | 32.1643 | 0.4791 | 0.7161 | 0.4364 | 0.5312 | 282s | |
| 14 | 6.0203 | 32.0387 | 0.4817 | 0.7195 | 0.4404 | 0.5316 | 284s | |
| 15 | 5.8847 | 32.1108 | 0.4811 | 0.7176 | 0.4396 | 0.5312 | 287s | Kết thúc 15 epochs |

> Biểu đồ đồ thị trực quan tiến trình huấn luyện được lưu tại file ảnh: `training_curves.png`.

---

## 🎯 5. KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (TEST SET RESULTS)

Đánh giá được thực hiện độc lập trên 1,000 mẫu của tập Test bằng trọng số tốt nhất (`best_model.pt` từ Epoch 12).

### 5.1. Các độ đo tổng thể

* **Span-Level Exact Match (Khớp chính xác cả ranh giới từ & nhãn):**
  * **Precision:** 44.38%
  * **Recall:** 53.56%
  * **Micro F1:** **48.54%**
  * Số lượng thống kê: **True Positives (TP) = 1,323**, False Positives (FP) = 1,658, False Negatives (FN) = 1,147.
* **Span-Level Proportional Match / IoU $\ge$ 0.5 (Khớp mềm / Gần đúng ranh giới):**
  * **Proportional Precision:** 65.25%
  * **Proportional Recall:** 78.74%
  * **Proportional F1:** **71.36%**
* **Macro F1 (Trung bình cộng 15 class):** **34.32%**

---

### 5.2. Kết quả chi tiết theo từng khía cạnh & cực tính (Per-category Breakdown)

| Khía cạnh & Cảm xúc (Aspect#Sentiment) | Precision (%) | Recall (%) | F1-Score (%) | Support (Số lượng mẫu thực tế) | Đánh giá mức độ hiệu quả |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `Material#positive` | **59.96%** | **67.67%** | **63.58%** | 467 | Xuất sắc (Aspect trọng tâm, nhiều dữ liệu) |
| `Service#positive` | **51.46%** | **57.83%** | **54.46%** | 396 | Rất tốt (Từ khóa dịch vụ rõ ràng) |
| `Material#negative` | **46.31%** | **56.63%** | **50.95%** | 166 | Khá tốt |
| `General#positive` | **43.33%** | **55.20%** | **48.55%** | 500 | Tốt (Mẫu nhiều nhưng ngữ cảnh rất đa dạng) |
| `Design#positive` | **43.33%** | **50.19%** | **46.51%** | 259 | Tốt |
| `Price#positive` | **40.78%** | **49.52%** | **44.73%** | 210 | Tốt ("giá rẻ", "hợp túi tiền") |
| `Design#negative` | **36.99%** | **54.00%** | **43.90%** | 150 | Trung bình khá |
| `Price#neutral` | **31.75%** | **43.48%** | **36.70%** | 46 | Trung bình ("giá tạm ổn", "tương đương giá") |
| `General#negative` | **28.00%** | **43.75%** | **34.15%** | 64 | Trung bình |
| `Material#neutral` | **29.85%** | **29.85%** | **29.85%** | 67 | Trung bình |
| `Service#negative` | **26.19%** | **26.83%** | **26.51%** | 41 | Thấp ("giao chậm", "hộp móp") |
| `General#neutral` | **24.24%** | **26.67%** | **25.40%** | 30 | Thấp |
| `Design#neutral` | **8.33%** | **11.11%** | **9.52%** | 54 | Rất thấp |
| `Price#negative` | **0.00%** | **0.00%** | **0.00%** | 11 | Không nhận diện được (Data quá ít) |
| `Service#neutral` | **0.00%** | **0.00%** | **0.00%** | 9 | Không nhận diện được (Data quá ít) |

---

## 🔍 6. PHÂN TÍCH CHUYÊN SÂU & BÀI HỌC KINH NGHIỆM (INSIGHTS)

### 6.1. Tại sao có sự chênh lệch lớn giữa Exact F1 (48.54%) và Proportional F1 (71.36%)?
* **Định nghĩa Exact Match:** Yêu cầu mô hình phải dự đoán chính xác tuyệt đối từng token từ vị trí bắt đầu đến kết thúc của span nhãn.
* **Thực tế văn bản đánh giá:** Khách hàng bình luận rất đa dạng về từ đệm và từ bổ nghĩa. Ví dụ:
  * Nhãn nhãn thực tế (Ground Truth): `[chất vải]` $\rightarrow$ `Material#positive`
  * Mô hình dự đoán: `[chất vải dày dặn mát mẻ]` $\rightarrow$ `Material#positive`
  * Ở trường hợp này, **Exact Match bị tính là False Positive và False Negative (F1 = 0%)**, trong khi về mặt ngữ nghĩa và ứng dụng thực tế mô hình đã bắt trúng 100% cảm xúc và khía cạnh.
* **Proportional F1 (71.36%)** và **Recall (78.74%)** phản ánh năng lực nhận biết khía cạnh thực tế của mô hình: mô hình phát hiện và gán đúng khía cạnh-cảm xúc cho gần 80% trường hợp người dùng đề cập.

### 6.2. Vấn đề mất cân bằng dữ liệu (Class Imbalance)
* Dữ liệu thương mại điện tử có thiên lệch tự nhiên: Đánh giá tích cực (Positive) chiếm đa số (> 70%), trong khi trung tính (Neutral) và tiêu cực (Negative) chiếm tỉ lệ rất nhỏ.
* Đặc biệt hai lớp `Price#negative` (chỉ 11 mẫu trong tập Test) và `Service#neutral` (chỉ 9 mẫu trong tập Test) có số lượng quá ít trong tập Train để tầng CRF và Linear Head học được trọng số phân tách. Đây là nguyên nhân kéo điểm **Macro F1 xuống còn 34.32%**, dù Micro F1 và Proportional F1 đạt mức tốt.

### 6.3. Vai trò của tầng CRF
* Giúp triệt tiêu hoàn toàn các lỗi cú pháp BIO (không bao giờ sinh ra nhãn lẻ loi dạng `I-` mà không có `B-`).
* Khóa chặt sự liên kết giữa các token liền kề có cùng cảm xúc trong cùng một cụm từ tiếng Việt.

---

## 📁 7. CẤU TRÚC THƯ MỤC OUTPUT VÀ HƯỚNG DẪN SỬ DỤNG

### 7.1. Các tệp tin trong thư mục `output/`
```text
output/
├── best_model.pt         # Trọng số PyTorch tốt nhất (lưu tại Epoch 12, ~1.6 GB)
├── config.json           # Cấu hình siêu tham số và từ điển id2label (31 nhãn)
├── results.json          # Toàn bộ chỉ số test, per-category và log 15 epoch
├── train.log             # Nhật ký chi tiết tiến trình huấn luyện và đánh giá
├── training_curves.png   # Biểu đồ đường cong Loss và F1 qua các epoch
└── OVERVIEW.md           # Báo cáo tổng quan và phân tích chi tiết (tệp tin này)
```

### 7.2. Hướng dẫn chạy thử nghiệm mô hình (Inference)
Sử dụng script `predict.py` để chạy suy luận với mô hình đã lưu:

```bash
# 1. Dự đoán trên một câu văn bản bất kỳ
python predict.py --text "Áo đẹp, vải mịn mát, nhưng giao hàng hơi lâu và giá hơi chát."

# 2. Chạy chế độ tương tác nhập liệu từ bàn phím (Interactive Mode)
python predict.py --interactive

# 3. Dự đoán hàng loạt từ file dữ liệu JSON
python predict.py --input_file data/test_data_processed.json --output_file output/predictions.json
```
