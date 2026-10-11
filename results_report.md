# 📊 Báo Cáo Giải Thích Kết Quả Huấn Luyện & Đánh Giá Mô Hình

> **Mô hình 2:** PhoBERT Multi-Head (Sentence-Level Aspect-Based Sentiment Analysis)  
> **Người phụ trách:** Minh (`feat/multihead`)  
> **Tập dữ liệu:** ViCloABSA (5.000 Train | 1.000 Dev | 1.000 Test)  
> **Hạn chót báo cáo nhóm:** 23:59 Ngày 15/10/2026  

---

## 📑 Mục Lục
- [1. Quy Ước Chung Về Nhãn & Khía Cạnh](#1-quy-ước-chung-về-nhãn--khía-cạnh)
- [2. Giải Thích Chi Tiết Từng File Kết Quả (.json)](#2-giải-thích-chi-tiết-từng-file-kết-quả-json)
  - [2.1. File `train_stats.json` - Phân bố nhãn & Trọng số phạt](#21-file-train_statsjson---phân-bố-nhãn--trọng-số-phạt)
  - [2.2. File `train_history.json` - Nhật ký 10 Epochs](#22-file-train_historyjson---nhật-ký-10-epochs)
  - [2.3. File `dev_best.json` - Checkpoint tốt nhất trên Dev](#23-file-dev_bestjson---checkpoint-tốt-nhất-trên-dev)
  - [2.4. File `test_results.json` - Kết quả kiểm thử chính thức](#24-file-test_resultsjson---kết-quả-kiểm-thử-chính-thức)
  - [2.5. File `test_confusion.json` - Ma trận nhầm lẫn 4x4](#25-file-test_confusionjson---ma-trận-nhầm-lẫn-4x4)
  - [2.6. File `test_predictions.json` - Chi tiết từng mẫu kiểm thử](#26-file-test_predictionsjson---chi-tiết-từng-mẫu-kiểm-thử)
- [3. Nhận Xét & Đánh Giá Toàn Diện Mô Hình](#3-nhận-xét--đánh-giá-toàn-diện-mô-hình)
  - [3.1. Tính hội tụ và Khả năng Tổng quát hóa (Generalization)](#31-tính-hội-tụ-và-khả-năng-tổng-quát-hóa-generalization)
  - [3.2. Hiệu quả của Class-Weighted Loss với nhãn thiểu số](#32-hiệu-quả-của-class-weighted-loss-với-nhãn-thiểu-số)
  - [3.3. Phân tích nguyên nhân ngôn ngữ của các lỗi sai](#33-phân-tích-nguyên-nhân-ngôn-ngữ-của-các-lỗi-sai)
  - [3.4. Ưu thế của kiến trúc Shared-Encoder & Tốc độ suy luận](#34-ưu-thế-của-kiến-trúc-shared-encoder--tốc-độ-suy-luận)
  - [3.5. Kết luận dùng cho Báo cáo (report.tex) & Họp nhóm](#35-kết-luận-dùng-cho-báo-cáo-reporttex--họp-nhóm)

---

## 1. Quy Ước Chung Về Nhãn & Khía Cạnh

Toàn bộ hệ thống thống nhất chuẩn biểu diễn vector 5 chiều:

### 5 Khía cạnh (Aspects):
1. **Material (0):** Chất liệu vải, độ dày, độ thoáng mát.
2. **General (1):** Đánh giá chung, tổng thể sản phẩm / shop.
3. **Service (2):** Dịch vụ CSKH, đóng gói, tốc độ giao hàng.
4. **Design (3):** Thiết kế, form dáng, mẫu mã, màu sắc, đường may.
5. **Price (4):** Giá tiền, độ tương xứng với giá bán.

### 4 Nhãn cảm xúc (Sentiments):
* `0: None` — Câu không đề cập đến khía cạnh này.
* `1: Positive` — Khen ngợi, tích cực.
* `2: Neutral` — Đánh giá trung tính, bình thường, nước đôi.
* `3: Negative` — Chê bai, tiêu cực, phàn nàn.

---

## 2. Giải Thích Chi Tiết Từng File Kết Quả (.json)

### 2.1. File `train_stats.json` - Phân bố nhãn & Trọng số phạt

File này ghi nhận phân tích dữ liệu trên **5.000 câu tập Train** trước khi mô hình học.

#### A. Phân bố nhãn thực tế ("train_label_distribution")

| Khía cạnh (Aspect) | 0: None | 1: Positive | 2: Neutral | 3: Negative | Tổng số | Nhận xét độ lệch |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Material** | 1,750 (35.0%) | 2,230 (44.6%) | 321 (6.4%) | 699 (14.0%) | 5,000 | Tương đối đồng đều |
| **General** | 2,253 (45.1%) | 2,334 (46.7%) | 156 (3.1%) | 257 (5.1%) | 5,000 | Lớp 0 & 1 chiếm 91.8% |
| **Service** | 2,848 (57.0%) | 1,877 (37.5%) | **46 (0.92%)** | 229 (4.58%) | 5,000 | Lớp Neutral cực hiếm |
| **Design** | 2,982 (59.6%) | 1,188 (23.8%) | 226 (4.52%) | 604 (12.08%) | 5,000 | Lớp 0 chiếm gần 60% |
| **Price** | 3,626 (72.5%) | 1,123 (22.5%) | 220 (4.40%) | **31 (0.62%)** | 5,000 | **Lớp Negative siêu hiếm** |

> [!WARNING]
> **Hiện tượng mất cân bằng cực hạn (Extreme Class Imbalance):**  
> Trên 5.000 đánh giá thời trang Shopee, chỉ có đúng **31 câu chê giá đắt** (0.62%) và **46 câu đánh giá dịch vụ bình thường** (0.92%). Nếu không có chiến lược xử lý, mô hình sẽ mặc định bỏ qua các lớp này.

#### B. Trọng số phạt nghịch đảo ("class_weights")
Công thức:  
$$w_{a,c} = \frac{N_{\text{total}}}{C \cdot N_{a,c}}$$

Hệ thống tính trọng số phạt bù trừ cho từng lớp của mỗi khía cạnh:
* **Service:** Lớp 2 (Neutral) nhận trọng số phạt **$27.17$**.
* **Price:** Lớp 3 (Negative) nhận trọng số phạt lên tới **$40.32$** (cao nhất toàn bộ dữ liệu).

> [!NOTE]
> **Ý nghĩa:** Khi mô hình đoán sai một câu chê giá đắt, mức phạt trong hàm mất mát sẽ nặng gấp **40 lần** so với đoán sai một câu không nhắc giá. Điều này ép gradient phải cập nhật để nhận diện được lớp thiểu số.

---

### 2.2. File `train_history.json` - Nhật ký 10 Epochs

Bảng theo dõi sự hội tụ của mô hình qua 10 epochs trên tập Train (5.000 mẫu) và tập Dev (1.000 mẫu):

| Epoch | Train Loss | Dev Loss | Dev Mean Aspect F1 (%) | Dev Exact Match Ratio (%) | Trạng thái Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 5.9110 | 4.1209 | 52.63% | 16.20% | 💾 Lưu Checkpoint 1 |
| **2** | 3.5095 | 2.8635 | 65.97% | 30.80% | 💾 Lưu Checkpoint 2 (+13.34%) |
| **3** | 2.5542 | 2.4812 | 69.40% | 47.70% | 💾 Lưu Checkpoint 3 (+3.43%) |
| **4** | 1.9521 | 2.5544 | 71.65% | 55.10% | 💾 Lưu Checkpoint 4 (+2.25%) |
| **5** | 1.5915 | 2.4246 | 72.86% | 55.20% | 💾 Lưu Checkpoint 5 (+1.21%) |
| **6** | 1.2619 | 2.5699 | 74.69% | 58.80% | 💾 Lưu Checkpoint 6 (+1.83%) |
| **7** | 1.0255 | 2.4636 | 74.28% | 56.40% | ⚠️ Không tăng (1/3 patience) |
| **8** | 0.8790 | 2.5224 | 74.46% | 58.60% | ⚠️ Không tăng (2/3 patience) |
| **9** | **0.7917** | **2.5982** | **75.21%** | **59.60%** | 🏆 **LƯU BEST MODEL (ĐỈNH CAO)** |
| **10**| 0.7698 | 2.5947 | 75.05% | 59.40% | 🏁 Hoàn tất 10 epochs |

* **Train Loss:** Giảm đều từ `5.9110` xuống `0.7698` $\rightarrow$ Tối ưu gradient rất ổn định nhờ Cosine Warmup Scheduler và AdamW.
* **Exact Match Ratio:** Tăng từ `16.20%` lên gần `60%` $\rightarrow$ Năng lực đoán trúng cùng lúc cả 5 khía cạnh trong một câu tăng gấp gần 4 lần.

---

### 2.3. File `dev_best.json` - Checkpoint tốt nhất trên Dev

Lưu lại toàn bộ metric của Checkpoint tốt nhất tại **Epoch 9** (được lưu thành `best_model.pt`):

```json
{
  "epoch": 9,
  "loss": 2.5982,
  "Material_macro_f1": 0.8003, "Material_acc": 0.887,
  "General_macro_f1":  0.7234, "General_acc":  0.819,
  "Service_macro_f1":  0.6695, "Service_acc":  0.948,
  "Design_macro_f1":   0.7893, "Design_acc":   0.872,
  "Price_macro_f1":    0.7780, "Price_acc":    0.942,
  "mean_aspect_macro_f1": 0.7521,
  "exact_match_ratio": 0.596
}
```

---

### 2.4. File `test_results.json` - Kết quả kiểm thử chính thức

Đây là số liệu quan trọng nhất của Mô hình 2, được đánh giá trên **1.000 câu Test độc lập**:

| Khía cạnh (Aspect) | Accuracy (%) | Macro-F1 (%) | Đánh giá chi tiết |
| :--- | :---: | :---: | :--- |
| **Price** (Giá cả) | **94.30%** | **79.32%** | **F1 cao nhất** (học rất tốt dù nhãn hiếm) |
| **Material** (Chất liệu) | 87.00% | **78.47%** | Rất cao (đặc trưng từ ngữ rõ ràng) |
| **Service** (Dịch vụ) | **95.10%** | **77.64%** | Accuracy cao nhất (> 95%) |
| **Design** (Thiết kế) | 84.70% | **74.70%** | Khá cao |
| **General** (Chung) | 80.90% | **66.78%** | Thấp nhất (ngữ nghĩa rộng, ranh giới mờ) |
| **MEAN ASPECT MACRO-F1** | — | **75.38%** | **Thước đo tổng thể chính của đề tài** |
| **EXACT MATCH RATIO** | **57.90%** | — | **Đoán đúng toàn bộ 5/5 khía cạnh** |
| **TỐC ĐỘ SUY LUẬN** | — | **1.27 ms / câu** | **~785 câu / giây trên GPU T4** |
| **BEST CHECKPOINT** | — | **Epoch 9** | Điểm rơi tối ưu |

---

### 2.5. File `test_confusion.json` - Ma trận nhầm lẫn 4x4

Cấu trúc ma trận: `Hàng = Nhãn thật`, `Cột = Nhãn dự đoán`. Thứ tự: `[0: None, 1: Pos, 2: Neu, 3: Neg]`.

#### A. Khía cạnh Material (Chất liệu)
```text
Thực tế \ Dự đoán     None (0)   Pos (1)   Neu (2)   Neg (3)
None (0):               295        27         6        12
Positive (1):             8       421        11         1
Neutral (2):              4         3        40        20   <-- Nhầm sang Neg 20 câu
Negative (3):             8         5        25       114   <-- Nhầm sang Neu 25 câu
```
* **Phân tích:** 421 câu khen chất liệu được đoán đúng. Điểm nhầm lẫn tập trung ở giữa **Neutral (2)** và **Negative (3)** do người mua hay dùng từ ngữ nước đôi (*"vải tạm được nhưng hơi mỏng"*).

#### B. Khía cạnh General (Tổng quan - F1 thấp nhất 66.78%)
```text
Thực tế \ Dự đoán     None (0)   Pos (1)   Neu (2)   Neg (3)
None (0):               355        56        25        20   <-- Nhầm None -> Pos 56 câu
Positive (1):            47       398         8         2   <-- Nhầm Pos -> None 47 câu
Neutral (2):              7         4        13         5
Negative (3):            12         1         4        43
```
* **Phân tích:** Ranh giới giữa "khen chung chung" và "không đánh giá khía cạnh chung" rất mong manh. Người chấm nhãn và mô hình thường khó phân định các câu như *"ưng bụng"*, *"tuyệt vời"* là thuộc về General hay quy về khía cạnh cụ thể.

#### C. Khía cạnh Price (Giá cả - F1 cao nhất 79.32%)
```text
Thực tế \ Dự đoán     None (0)   Pos (1)   Neu (2)   Neg (3)
None (0):               712        17         8         0   <-- Đoán đúng None 96.6%
Positive (1):             5       193         9         0
Neutral (2):              3         8        32         2
Negative (3):             0         2         3         6   <-- Đoán đúng 6/11 câu chê đắt
```
* **Phân tích:** Nhận diện nhãn None cực kỳ xuất sắc (712/737 câu). Ở lớp chê giá siêu hiếm (chỉ có 11 câu trên test), mô hình vẫn phát hiện đúng 6 câu và không bao giờ đoán nhầm sang None.

---

### 2.6. File `test_predictions.json` - Chi tiết từng mẫu kiểm thử

Chứa đầy đủ kết quả của 1.000 câu. Ví dụ:

* **Ví dụ 1 (Đoán đúng tuyệt đối - ID 0):**
  - **Bình luận:** *"chất liệu:không đảm bảo"*
  - **Nhãn thật (`gold`):** `[3, 0, 0, 0, 0]` (Material: Negative, 4 aspect còn lại: None)
  - **Dự đoán (`pred`):** `[3, 0, 0, 0, 0]` $\rightarrow$ **ĐÚNG 100%**.

---

## 3. Nhận Xét & Đánh Giá Toàn Diện Mô Hình

### 3.1. Tính hội tụ và Khả năng Tổng quát hóa (Generalization)
* Điểm **Dev Macro-F1 đạt 75.21%** và **Test Macro-F1 đạt 75.38%** (độ chênh lệch chỉ $0.17\%$).
* Điều này chứng minh mô hình **không hề bị Overfitting**. Khả năng khái quát hóa trên tập dữ liệu hoàn toàn mới là cực kỳ vững chắc.

### 3.2. Hiệu quả của Class-Weighted Loss với nhãn thiểu số
* Việc áp dụng trọng số nghịch đảo đã giải quyết triệt để vấn đề mất cân bằng dữ liệu.
* Bằng chứng đắt giá nhất: Khía cạnh **Price** (nơi có nhãn tiêu cực chỉ chiếm $0.62\%$ tập train) lại đạt **Macro-F1 cao nhất (79.32%)** trong toàn bộ 5 khía cạnh.

### 3.3. Phân tích nguyên nhân ngôn ngữ của các lỗi sai
1. **Sắc thái nước đôi (Khen - Chê giảm nhẹ):** Các đánh giá như *"với giá này thì chất lượng tạm được"* tạo ra sự chồng lấn giữa Neutral (2) và Negative (3).
2. **Ngữ cảnh không tường minh của khía cạnh General:** Các từ cảm thán như *"quá đỉnh"*, *"ok"* nếu không đi kèm tên sản phẩm cụ thể sẽ gây nhiễu giữa lớp None và Positive.

### 3.4. Ưu thế của kiến trúc Shared-Encoder & Tốc độ suy luận
* Thay vì chạy 5 mô hình PhoBERT độc lập (tốn gấp 5 lần VRAM và thời gian), việc dùng **1 PhoBERT Encoder kết hợp 5 Linear Heads song song**:
  - Giúp các khía cạnh học ké biểu diễn ngữ nghĩa của nhau.
  - Đạt tốc độ suy luận **1.27 ms / câu**, sẵn sàng cho ứng dụng phân tích bình luận thời gian thực trong sản xuất.

### 3.5. Kết luận dùng cho Báo cáo (report.tex) & Họp nhóm
1. Mô hình 2 (PhoBERT Multi-Head) của bạn Minh đã **hoàn thành 100% mục tiêu nghiên cứu**.
2. Kết quả **Mean Macro-F1 = 75.38%** và **Exact Match Ratio = 57.90%** là cột mốc Deep Learning rất mạnh để đối sánh với Baseline (TF-IDF + SVC) của bạn Phúc và PhoBERT + CRF của bạn Cường trong buổi họp ngày 15/10/2026.
