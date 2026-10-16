# BÁO CÁO TỔNG QUAN & ĐÁNH GIÁ KẾT QUẢ HUẤN LUYỆN (MODEL OVERVIEW)
**Dự án:** ViCloABSA - Phân tích Cảm xúc Đa khía cạnh trong Đánh giá Thời trang Tiếng Việt  
**Mô hình:** PhoBERT-base-v2 + CRF (Span-Level Extraction & Sentence-Level Aggregation)  
**Nhánh thực hiện:** `feat/crf` (Thành viên 3)  
**Cập nhật mới nhất:** 16/10/2026  

---

## 🏆 TỔNG HỢP KẾT QUẢ CỐT LÕI TRÊN TẬP TEST (1,000 MẪU)

Mô hình PhoBERT + CRF giải quyết đồng thời hai cấp độ: **Trích xuất vị trí cụm từ (Span-Level)** và **Phân loại cảm xúc khía cạnh câu (Sentence-Level)**.

```text
========================================================================================
1. THƯỚC ĐO ĐỐI SÁNH VỚI NHÓM (SENTENCE-LEVEL - Thước đo chung của cả 3 mô hình):
   ★ MEAN ASPECT MACRO-F1 : 75.09%  (Trung bình cộng Macro-F1 của 5 khía cạnh)
   ★ EXACT MATCH RATIO    : 59.20%  (Tỉ lệ câu dự đoán chính xác cả 5 khía cạnh cùng lúc)

2. THƯỚC ĐO TRÍCH XUẤT CỤM TỪ ĐẶC THÙ CỦA CRF (SPAN-LEVEL):
   ★ PROPORTIONAL SPAN F1 : 71.36%  (Khớp mềm ranh giới từ với IoU >= 0.5)
   ★ EXACT SPAN F1        : 48.54%  (Khớp chính xác tuyệt đối từng ký tự bắt đầu và kết thúc)
========================================================================================
```

---

## 📌 1. TỔNG QUAN KIẾN TRÚC MÔ HÌNH

Hệ thống giải quyết bài toán theo kiến trúc lai kết hợp giữa biểu diễn ngữ cảnh sâu và tối ưu chuỗi nhãn toàn cục:
$$\text{Văn bản đầu vào} \xrightarrow{\text{PhoBERT}} \text{Contextual Embeddings} \xrightarrow{\text{Linear Head}} \text{Emission Scores} \xrightarrow{\text{CRF Layer (Viterbi)}} \text{Chuỗi nhãn tối ưu}$$

* **Cốt lõi ngôn ngữ (Pre-trained LM):** `vinai/phobert-base-v2` (135,023,134 tham số). Mô hình được tiền huấn luyện đặc thù trên tiếng Việt với cơ chế từ ghép phân tách bằng dấu gạch dưới (`_`), mang lại khả năng nắm bắt ngữ nghĩa từ vựng thương mại điện tử vượt trội.
* **Tầng phân loại & giải mã (Classification & CRF):**
  * Tầng Linear chiếu từ không gian ẩn ($d=768$) sang 31 chiều nhãn.
  * Tầng **Conditional Random Field (CRF)**: Học ma trận chuyển trạng thái nhãn ($31 \times 31$), triệt tiêu các lỗi gán nhãn cú pháp phi lý (ví dụ: cấm chuyển trực tiếp từ `O` sang `I-Material#positive` mà không có nhãn `B-` mở đầu).
* **Không gian 31 nhãn BIO:**
  * `O`: Không thuộc khía cạnh nào.
  * 5 Khía cạnh ($\text{Aspects}$): **Material** (Chất liệu), **Design** (Thiết kế / Form dáng), **Price** (Giá cả), **Service** (Dịch vụ / Giao hàng / Tư vấn), **General** (Đánh giá chung).
  * 3 Cực tính cảm xúc ($\text{Sentiments}$): **positive**, **neutral**, **negative**.
  * Cặp nhãn BIO: $B\text{-Aspect-Sentiment}$ và $I\text{-Aspect-Sentiment}$ ($2 \times 5 \times 3 = 30$ nhãn + 1 nhãn $O$).

---

## 📊 2. DỮ LIỆU & TỈ LỆ PHÂN CHIA (DATASET)

Bộ dữ liệu gồm **7,000 mẫu** đánh giá thực tế từ các sàn thương mại điện tử ngành thời trang:

| Tập dữ liệu | Số lượng mẫu (Samples) | Tỉ lệ (%) | Mục đích sử dụng |
| :--- | :---: | :---: | :--- |
| **Train** | **5,000** | 71.43% | Tinh chỉnh tham số PhoBERT và huấn luyện tầng CRF |
| **Dev (Validation)** | **1,000** | 14.29% | Theo dõi và chọn checkpoint tối ưu nhất qua từng epoch |
| **Test** | **1,000** | 14.29% | Đánh giá độc lập chất lượng mô hình sau khi train xong |
| **Tổng cộng** | **7,000** | 100.0% | |

---

## ⚙️ 3. THIẾT LẬP THỰC NGHIỆM & SIÊU THAM SỐ (HYPERPARAMETERS)

* **Thiết bị phần cứng:** NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB VRAM).
* **Tăng tốc huấn luyện:** Mixed Precision FP16 (`torch.cuda.amp`).
* **Số Epochs:** 15 epochs.
* **Batch size:** 16 (train), 32 (eval).
* **Chiều dài tối đa (`max_length`):** 256 tokens.
* **Tốc độ học vi sai (Differential Learning Rates):**
  * **PhoBERT Encoder LR:** `2e-5` (giữ ổn định tri thức tiền huấn luyện).
  * **Classifier Head & CRF LR:** `1e-4` (tốc độ học nhanh hơn để tầng giải mã hội tụ tốt).
* **Bộ tối ưu:** AdamW, Weight Decay = 0.01, Gradient Clipping = 1.0.
* **Lịch trình học (Scheduler):** Cosine Annealing với Warmup 10% (469 warmup steps trên 4,695 total steps).
* **Tổng thời gian huấn luyện:** **70.62 phút** (~4.7 phút/epoch).

---

## 📈 4. DIỄN BIẾN HUẤN LUYỆN QUA 15 EPOCHS

Checkpoint tốt nhất được hệ thống tự động lưu tại **Epoch 12** với **Dev Exact F1 đạt 0.4823** và **Dev Proportional F1 đạt 0.7190**.

| Epoch | Train Loss | Dev Loss | Dev Exact F1 | Dev Prop. F1 | Dev Precision | Dev Recall | Thời gian (s) | Ghi chú |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | 81.8582 | 53.3834 | 0.1658 | 0.4824 | 0.1634 | 0.1682 | 277s | Warmup ban đầu |
| 2 | 43.3133 | 32.8845 | 0.3974 | 0.6424 | 0.3550 | 0.4513 | 282s | Bắt đầu học ranh giới span |
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

---

## 🎯 5. CHI TIẾT KẾT QUẢ TRÊN TẬP TEST (1,000 MẪU)

Đánh giá bằng checkpoint `best_model.pt` (Epoch 12) trên 2 cấp độ:

### 5.1. CẤP ĐỘ CÂU (SENTENCE-LEVEL) — THƯỚC ĐO CHUNG CỦA DỰ ÁN

Đây là thước đo chuẩn hóa được quy định trong `src/metrics.py` để **đối sánh công bằng giữa cả 3 mô hình của nhóm**:
* Quy đổi: Tổng hợp các spans trích xuất được từ câu thành vector 5 chiều `[Material, General, Service, Design, Price]`. Mỗi khía cạnh nhận 1 trong 4 nhãn: `{0: None, 1: positive, 2: neutral, 3: negative}`.

| Khía cạnh (Aspect) | Macro-F1 (%) | Accuracy (%) | Đánh giá năng lực |
| :--- | :---: | :---: | :--- |
| **Material** (Chất liệu) | **79.64%** | 89.20% | Xuất sắc, nhận diện cực kỳ chính xác cảm xúc về vải |
| **Service** (Dịch vụ) | **76.64%** | 95.30% | Rất cao, phân biệt rõ đánh giá giao hàng / đóng gói |
| **Price** (Giá cả) | **76.62%** | 94.20% | Rất cao, nắm bắt tốt từ vựng về giá rẻ, giá hợp lý |
| **General** (Đánh giá chung) | **72.16%** | 82.70% | Tốt, xử lý ngữ cảnh đa dạng của câu khen tổng thể |
| **Design** (Thiết kế / Form dáng) | **70.38%** | 83.90% | Tốt, nhận biết chuẩn form áo, kích cỡ, màu sắc |
| ------------------------------------ | ------------ | ------------ | ------------------------------------------------ |
| **MEAN ASPECT MACRO-F1 (Metric chính)** | **75.09%** | — | **Độ đo trung bình đối sánh với 2 mô hình kia** |
| **EXACT MATCH RATIO** | **59.20%** | — | **Tỉ lệ đúng hoàn toàn cả 5 khía cạnh trong câu** |

---

### 5.2. CẤP ĐỘ CỤM TỪ (SPAN-LEVEL) — ĐẶC TRÙ MÔ HÌNH CRF

* **Span-Level Exact Match (Khớp chính xác cả vị trí ký tự & nhãn):**
  * Precision: 44.38% | Recall: 53.56% | **Micro F1: 48.54%**
  * TP = 1,323 | FP = 1,658 | FN = 1,147
* **Span-Level Proportional Match (IoU $\ge$ 0.5):**
  * Precision: 65.25% | Recall: 78.74% | **Proportional F1: 71.36%**
* **Macro F1 (Trung bình 15 cặp Aspect#Sentiment):** **34.32%**

#### Chi tiết theo từng cặp nhãn Span (Aspect#Sentiment):

| Cặp nhãn (Category) | Prec (%) | Rec (%) | F1 (%) | Support | Ghi chú |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `Material#positive` | 59.96% | 67.67% | **63.58%** | 467 | Khía cạnh mạnh nhất, dữ liệu dồi dào |
| `Service#positive` | 51.46% | 57.83% | **54.46%** | 396 | Rất tốt |
| `Material#negative` | 46.31% | 56.63% | **50.95%** | 166 | Khá tốt |
| `General#positive` | 43.33% | 55.20% | **48.55%** | 500 | Tốt |
| `Design#positive` | 43.33% | 50.19% | **46.51%** | 259 | Tốt |
| `Price#positive` | 40.78% | 49.52% | **44.73%** | 210 | Tốt |
| `Design#negative` | 36.99% | 54.00% | **43.90%** | 150 | Trung bình khá |
| `Price#neutral` | 31.75% | 43.48% | **36.70%** | 46 | Trung bình |
| `General#negative` | 28.00% | 43.75% | **34.15%** | 64 | Trung bình |
| `Material#neutral` | 29.85% | 29.85% | **29.85%** | 67 | Trung bình |
| `Service#negative` | 26.19% | 26.83% | **26.51%** | 41 | Thấp |
| `General#neutral` | 24.24% | 26.67% | **25.40%** | 30 | Thấp |
| `Design#neutral` | 8.33% | 11.11% | **9.52%** | 54 | Rất thấp |
| `Price#negative` | 0.00% | 0.00% | **0.00%** | 11 | Không bắt được (Dữ liệu test chỉ có 11 mẫu) |
| `Service#neutral` | 0.00% | 0.00% | **0.00%** | 9 | Không bắt được (Dữ liệu test chỉ có 9 mẫu) |

---

## 🔍 6. PHÂN TÍCH CHUYÊN SÂU & GIẢI THÍCH KẾT QUẢ

### 6.1. Mối liên hệ giữa Mean Aspect Macro-F1 (75.09%) và Span-Level F1 (48.54%)
* **Tại sao Span-Level Exact F1 là 48.54% nhưng Sentence Macro-F1 lại đạt 75.09%?**
  * Trong đánh giá Span-level Exact, mô hình phải chỉ ra chính xác tuyệt đối từng ký tự bắt đầu và kết thúc của cụm từ. Nếu nhãn gốc là `"vải mềm"` mà mô hình cắt thành `"chất vải mềm"` thì bị tính là 1 FP và 1 FN (F1 = 0%), mặc dù về mặt nhận biết ngữ nghĩa câu, mô hình đã hiểu đúng 100% người dùng khen chất liệu tích cực.
  * Khi tổng hợp các spans về cấp độ câu (Sentence-Level), các sai lệch ranh giới nhỏ này không còn ảnh hưởng đến việc phân loại cảm xúc tổng thể. Nhờ đó, **Mean Aspect Macro-F1 đạt 75.09%** và **Exact Match Ratio đạt 59.20%**.
  * **Proportional Span F1 (IoU $\ge$ 0.5) đạt 71.36%**, tương thích cao với điểm số câu, chứng minh mô hình định vị rất sát vị trí xuất hiện khía cạnh.

### 6.2. Ưu thế vượt trội của mô hình CRF khi đối sánh với Baseline và Multi-Head
* **Mô hình Baseline (TF-IDF + LinearSVC)** và **PhoBERT Multi-Head** chỉ dự đoán được nhãn ở cấp độ câu ("câu này có khen chất liệu hay không"), nhưng **hoàn toàn "mù" về vị trí** (không biết từ nào, cụm từ nào mang sắc thái đó).
* **Mô hình PhoBERT + CRF** không những đạt độ chính xác câu rất cao (**75.09% Mean Aspect Macro-F1**), mà còn **trích xuất và highlight được trực tiếp cụm từ bằng chứng** trong câu cho người dùng.

---

## 🖼️ 7. BIỂU ĐỒ TRỰC QUAN HÓA TỔNG HỢP (`training_curves.png`)

Tệp đồ thị `output/training_curves.png` đã được thiết kế lại thành **lưới 4 biểu đồ toàn diện (2x2)**:
1. **Ô 1 (Góc trên - trái):** Tiến trình Dev F1 / Precision / Recall qua 15 epochs, đánh dấu epoch 12 tối ưu nhất.
2. **Ô 2 (Góc trên - phải):** Đường cong hội tụ Train Loss và Dev Loss qua 15 epochs.
3. **Ô 3 (Góc dưới - trái):** Biểu đồ cột thể hiện Macro-F1 và Accuracy của từng Aspect ở cấp độ câu, cùng đường tham chiếu **Mean Aspect Macro-F1 (75.09%)**.
4. **Ô 4 (Góc dưới - phải):** Cột so sánh trực quan toàn diện 5 độ đo hiệu năng của mô hình trên tập Test.

---

## 📁 8. CẤU TRÚC THƯ MỤC OUTPUT VÀ HƯỚNG DẪN SỬ DỤNG

```text
output/
├── best_model.pt         # Trọng số mô hình tối ưu nhất (Epoch 12)
├── config.json           # Cấu hình siêu tham số và bộ 31 nhãn BIO
├── results.json          # Toàn bộ chỉ số test (Sentence-level & Span-level) và lịch sử 15 epochs
├── train.log             # Nhật ký chi tiết tiến trình huấn luyện
├── training_curves.png   # Biểu đồ lưới 4 ô trực quan hóa toàn bộ kết quả
└── OVERVIEW.md           # Báo cáo tổng quan và giải thích kết quả (tệp tin này)
```

### Lệnh chạy dự đoán (Inference):
```bash
# Dự đoán 1 câu bất kỳ (hiển thị cả nhãn câu và cụm từ span)
python predict.py --text "Áo mặc rất mát, form rộng đẹp, giá rẻ nhưng giao hơi lâu."

# Chế độ nhập liệu tương tác liên tục
python predict.py --interactive
```
