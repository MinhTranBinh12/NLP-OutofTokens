# 📊 Báo Cáo Giải Thích Kết Quả Huấn Luyện & Đánh Giá Mô Hình

> **Mô hình 1:** Baseline (TF-IDF + LinearSVC Multi-Output)  
> **Người phụ trách:** Phúc (`feat/baseline`)  
> **Tập dữ liệu:** ViCloABSA (5.000 Train | 1.000 Dev | 1.000 Test)  
> **Hạn chót báo cáo nhóm:** 23:59 Ngày 15/10/2026  

---

## 📑 Mục Lục
- [1. Quy Ước Chung Về Nhãn & Khía Cạnh](#1-quy-ước-chung-về-nhãn--khía-cạnh)
- [2. Giải Thích Chi Tiết Từng File Kết Quả (.json)](#2-giải-thích-chi-tiết-từng-file-kết-quả-json)
  - [2.1. File `train_stats.json` - Phân bố nhãn & Trọng số phạt](#21-file-train_statsjson---phân-bố-nhãn--trọng-số-phạt)
  - [2.2. File `train_history.json` - Cơ chế tối ưu Coordinate Descent & Lý do không chia Epoch](#22-file-train_historyjson---cơ-chế-tối-ưu-coordinate-descent--lý-do-không-chia-epoch)
  - [2.3. File `dev_best.json` - Checkpoint tối ưu trên Dev](#23-file-dev_bestjson---checkpoint-tối-ưu-trên-dev)
  - [2.4. File `test_results.json` - Kết quả kiểm thử chính thức](#24-file-test_resultsjson---kết-quả-kiểm-thử-chính-thức)
  - [2.5. File `test_confusion.json` - Ma trận nhầm lẫn 4x4](#25-file-test_confusionjson---ma-trận-nhầm-lẫn-4x4)
  - [2.6. File `test_predictions.json` - Chi tiết từng mẫu kiểm thử](#26-file-test_predictionsjson---chi-tiết-từng-mẫu-kiểm-thử)
- [3. Nhận Xét & Đánh Giá Toàn Diện Mô Hình](#3-nhận-xét--đánh-giá-toàn-diện-mô-hình)
  - [3.1. Tính hội tụ và Khả năng Tổng quát hóa (Generalization)](#31-tính-hội-tụ-và-khả-năng-tổng-quát-hóa-generalization)
  - [3.2. Hiệu quả và Hạn chế của `class_weight='balanced'` trong LinearSVC](#32-hiệu-quả-và-hạn-chế-của-class_weightbalanced-trong-linearsvc)
  - [3.3. Phân tích nguyên nhân ngôn ngữ của các lỗi sai](#33-phân-tích-nguyên-nhân-ngôn-ngữ-của-các-lỗi-sai)
  - [3.4. Ưu thế của kiến trúc TF-IDF + LinearSVC & Tốc độ suy luận](#34-ưu-thế-của-kiến-trúc-tf-idf--linearsvc--tốc-độ-suy-luận)
  - [3.5. Kết luận dùng cho Báo cáo (report.tex) & Họp nhóm](#35-kết-luận-dùng-cho-báo-cáo-reporttex--họp-nhóm)

---

## 1. Quy Ước Chung Về Nhãn & Khía Cạnh

Toàn bộ hệ thống thống nhất chuẩn biểu diễn vector 5 chiều đồng bộ cả 3 mô hình:

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
| **Material** | 1,750 (35.0%) | 2,230 (44.6%) | 321 (6.42%) | 699 (13.98%) | 5,000 | Tương đối đồng đều |
| **General** | 2,253 (45.1%) | 2,334 (46.7%) | 156 (3.12%) | 257 (5.14%) | 5,000 | Lớp 0 & 1 chiếm 91.8% |
| **Service** | 2,848 (57.0%) | 1,877 (37.5%) | 46 (0.92%) | 229 (4.58%) | 5,000 | Lớp Neutral cực hiếm (0.92%) |
| **Design** | 2,982 (59.6%) | 1,188 (23.8%) | 226 (4.52%) | 604 (12.08%) | 5,000 | Lớp 0 chiếm gần 60% |
| **Price** | 3,626 (72.5%) | 1,123 (22.5%) | 220 (4.40%) | 31 (0.62%) | 5,000 | **Lớp Negative siêu hiếm (0.62%)** |

> [!WARNING]
> **Hiện tượng mất cân bằng cực hạn (Extreme Class Imbalance):**  
> Trên 5.000 đánh giá thời trang Shopee, chỉ có đúng **31 câu chê giá đắt** (0.62%) và **46 câu đánh giá dịch vụ bình thường** (0.92%). Sự mất cân bằng này là thách thức lớn nhất đối với các mô hình Machine Learning truyền thống dựa trên tần số từ vựng.

#### B. Trọng số phạt nghịch đảo ("class_weights")
Công thức:  
$$w_{a,c} = \frac{N_{\text{total}}}{C \cdot N_{a,c}}$$

Hệ thống tính trọng số phạt bù trừ cho từng lớp của mỗi khía cạnh:
* **Service:** Lớp 2 (Neutral) nhận trọng số phạt **$27.17$**.
* **Price:** Lớp 3 (Negative) nhận trọng số phạt lên tới **$40.32$** (cao nhất toàn bộ dữ liệu).

> [!NOTE]
> **Ý nghĩa trong Baseline:** Trong mô hình LinearSVC, cờ `class_weight='balanced'` tự động sử dụng trọng số nghịch đảo này để nhân vào hàm phạt Hinge Loss của từng mẫu, ép siêu phẳng phân cách dịch chuyển để không bỏ sót các lớp thiểu số.

---

### 2.2. File `train_history.json` - Cơ chế tối ưu Coordinate Descent & Lý do không chia Epoch

Mô hình Baseline (TF-IDF + LinearSVC) không sử dụng vòng lặp Epochs như Deep Learning. Lý do xuất phát từ bản chất toán học của bài toán tối ưu:

1. **Bản chất tối ưu lồi (Convex Optimization):**  
   LinearSVC (dựa trên thư viện **LIBLINEAR**) giải bài toán tối ưu hóa lồi có ràng buộc với hàm mất mát Squared Hinge Loss kết hợp chuẩn hóa L2:
   $$\min_{\mathbf{w}} \frac{1}{2} \mathbf{w}^T \mathbf{w} + C \sum_{i=1}^N \max(0, 1 - y_i \mathbf{w}^T \mathbf{x}_i)^2$$
   Hàm mục tiêu này là một **hàm lồi nghiêm ngặt (strictly convex)**, đảm bảo tồn tại một điểm cực tiểu toàn cục (Global Optimum) duy nhất.

2. **Thuật toán Coordinate Descent:**  
   Thay vì dùng Stochastic Gradient Descent (SGD) chia mini-batch qua nhiều Epochs như PhoBERT, LinearSVC sử dụng thuật toán Coordinate Descent duyệt qua ma trận đặc trưng toàn cục $X$ và tối ưu trực tiếp từng biến đối ngẫu (dual variable) cho đến khi sai số gradient nhỏ hơn ngưỡng $\epsilon = 10^{-4}$ (hoặc chạm `max_iter=2000`).

3. **Tính đầy đủ của quá trình huấn luyện:**  
   Khi thuật toán đạt điểm dừng hội tụ, toàn bộ 5 bộ phân loại LinearSVC đã đạt nghiệm tối ưu toàn cục. Việc gọi `fit()` một lần trên 5.000 mẫu đã đảm bảo **100% tính hội tụ và hoàn tất huấn luyện**, không cần và không thể chia Epochs theo kiểu Deep Learning.

Bảng ghi nhận trạng thái hội tụ tại `train_history.json`:

| Bước tối ưu | Thuật toán | Dev Mean Aspect F1 (%) | Dev Exact Match Ratio (%) | Trạng thái |
| :---: | :---: | :---: | :---: | :--- |
| **Hội tụ toàn cục** | Dual Coordinate Descent (LIBLINEAR) | **67.34%** | **48.10%** | 🏆 **LƯU CHECKPOINT BASELINE TỐI ƯU** |

---

### 2.3. File `dev_best.json` - Checkpoint tối ưu trên Dev

Lưu lại toàn bộ metric của mô hình Baseline tốt nhất trên tập Dev (1.000 mẫu):

```json
{
  "epoch": 1,
  "Material_macro_f1": 0.7574, "Material_acc": 0.849,
  "General_macro_f1":  0.6162, "General_acc":  0.757,
  "Service_macro_f1":  0.6510, "Service_acc":  0.934,
  "Design_macro_f1":   0.6324, "Design_acc":   0.805,
  "Price_macro_f1":    0.7102, "Price_acc":    0.933,
  "mean_aspect_macro_f1": 0.6734,
  "exact_match_ratio": 0.4810
}
```

---

### 2.4. File `test_results.json` - Kết quả kiểm thử chính thức

Đây là số liệu quan trọng nhất của Mô hình Baseline, được đánh giá trên **1.000 câu Test độc lập**:

| Khía cạnh (Aspect) | Accuracy (%) | Macro-F1 (%) | Đánh giá chi tiết |
| :--- | :---: | :---: | :--- |
| **Service** (Dịch vụ) | **92.90%** | **71.27%** | **F1 cao nhất** (từ khóa CSKH, ship hàng rất đặc trưng) |
| **Material** (Chất liệu) | 81.50% | **69.15%** | Khá cao (từ vựng vải, dày, mát phân tách rõ) |
| **Design** (Thiết kế) | 78.10% | **64.14%** | Trung bình khá |
| **Price** (Giá cả) | **92.70%** | **61.68%** | Độ chính xác cao nhưng F1 thấp do nhãn chê giá hiếm |
| **General** (Chung) | 74.60% | **57.65%** | Thấp nhất (từ ngữ chung chung, dễ nhầm với None) |
| **MEAN ASPECT MACRO-F1** | — | **64.78%** | **Thước đo tổng thể chính của đề tài** |
| **EXACT MATCH RATIO** | **43.20%** | — | **Đoán đúng cùng lúc toàn bộ 5/5 khía cạnh** |
| **TỐC ĐỘ SUY LUẬN** | — | **0.0237 ms / câu** | **~42,154 câu / giây trên CPU (Siêu tốc)** |
| **DUNG LƯỢNG MÔ HÌNH** | — | **~2.0 MB** | Cực kỳ gọn nhẹ, không yêu cầu GPU |

---

### 2.5. File `test_confusion.json` - Ma trận nhầm lẫn 4x4

Cấu trúc ma trận: `Hàng = Nhãn thật`, `Cột = Nhãn dự đoán`. Thứ tự: `[0: None, 1: Pos, 2: Neu, 3: Neg]`.

#### Material (Khía cạnh: Material)
```text
Thực tế \ Dự đoán     None (0)   Pos (1)   Neu (2)   Neg (3)
None (0):               295        28         4        13
Positive (1):            37       393         8         3
Neutral (2):              8         9        18        32
Negative (3):            19        10        14       109
```

#### General (Khía cạnh: General)
```text
Thực tế \ Dự đoán     None (0)   Pos (1)   Neu (2)   Neg (3)
None (0):               341        84        15        16
Positive (1):            79       371         4         1
Neutral (2):             16         5         7         1
Negative (3):            28         2         3        27
```
* **Nhận xét về General:** Nhầm lẫn 84 câu None sang Pos và 79 câu Pos sang None do đặc trưng Bag-of-Words không phân biệt được khen chung hay khen sản phẩm cụ thể.

#### Service (Khía cạnh: Service)
```text
Thực tế \ Dự đoán     None (0)   Pos (1)   Neu (2)   Neg (3)
None (0):               568        11         0         1
Positive (1):            28       343         0         2
Neutral (2):              0         3         2         2
Negative (3):            14        10         0        16
```

#### Design (Khía cạnh: Design)
```text
Thực tế \ Dự đoán     None (0)   Pos (1)   Neu (2)   Neg (3)
None (0):               508        29         5        21
Positive (1):            65       166         9         4
Neutral (2):             12        12        14        14
Negative (3):            21        14        13        93
```

#### Price (Khía cạnh: Price)
```text
Thực tế \ Dự đoán     None (0)   Pos (1)   Neu (2)   Neg (3)
None (0):               720        10         7         0
Positive (1):            21       179         7         0
Neutral (2):              5        12        28         0
Negative (3):             6         3         2         0
```
* **Nhận xét quan trọng về Price:** Trên 11 câu thực tế chê giá đắt (Negative), mô hình Baseline **đoán đúng 0 câu** (bị nhầm sang None 6 câu, Pos 3 câu, Neu 2 câu). Điều này phản ánh rõ giới hạn của mô hình truyền thống khi gặp lớp siêu hiếm.

---

### 2.6. File `test_predictions.json` - Chi tiết từng mẫu kiểm thử

Chứa đầy đủ kết quả của 1.000 câu kiểm thử độc lập.

* **Ví dụ đoán đúng tuyệt đối 5/5 khía cạnh:**
  - **Bình luận:** *"hình ảnh mang tính chất nhận xu,giáo hàng lâu,giáo nhầm nhà."*
  - **Nhãn thật (`gold`):** `[0, 0, 3, 0, 0]`
  - **Dự đoán (`pred`):** `[0, 0, 3, 0, 0]` $\rightarrow$ **ĐÚNG 100%**.

* **Ví dụ đoán sai do giới hạn của mô hình n-gram / Bag-of-Words:**
  - **Bình luận:** *"chất liệu:không đảm bảo"*
  - **Nhãn thật (`gold`):** `[3, 0, 0, 0, 0]`
  - **Dự đoán (`pred`):** `[0, 0, 0, 0, 0]` $\rightarrow$ **Lệch nhãn** do thiếu hiểu biết ngữ cảnh sâu.

---

## 3. Nhận Xét & Đánh Giá Toàn Diện Mô Hình

### 3.1. Tính hội tụ và Khả năng Tổng quát hóa (Generalization)
* **Dev Macro-F1 = 67.34%** so với **Test Macro-F1 = 64.78%** (độ lệch chỉ **2.56%**).
* **Dev Exact Match = 48.10%** so với **Test Exact Match = 43.20%**.
* Khoảng cách giữa Dev và Test rất nhỏ, chứng minh mô hình **không hề bị Overfitting**. Nhờ chuẩn hóa L2 với $C=0.7$, siêu phẳng phân cách của LinearSVC đạt độ tổng quát hóa rất ổn định trên dữ liệu mới.

### 3.2. Hiệu quả và Hạn chế của `class_weight='balanced'` trong LinearSVC
* Việc bật `class_weight='balanced'` giúp mô hình nâng cao đáng kể Recall ở các khía cạnh có nhãn tương đối phân bố như Material và Service.
* Tuy nhiên, đối với **Price (Negative chỉ chiếm 0.62%)**, mô hình đạt F1 = 0.00% trên lớp tiêu cực. Nguyên nhân là vì không gian TF-IDF là không gian thưa (sparse vector), các từ chê đắt xuất hiện quá ít (chỉ 31 lần / 5.000 câu) nên trọng số phân lớp tuyến tính không đủ biên phân cách vững chắc như cơ chế nhúng ngữ cảnh sâu (Contextualized Embeddings) của PhoBERT.

### 3.3. Phân tích nguyên nhân ngôn ngữ của các lỗi sai
1. **Mất mát ngữ cảnh từ xa (Long-range Dependencies):** Do chỉ dùng n-gram (1, 2), các câu đánh giá phức tạp như *"áo mặc mát nhưng giao hàng quá chậm"* thường bị trộn lẫn đặc trưng giữa chất liệu và dịch vụ.
2. **Hiện tượng phủ định và chuyển hướng ngữ nghĩa:** Các cụm từ như *"không đắt"*, *"chẳng có gì để chê"* chứa từ *"đắt"*, *"chê"* khiến mô hình Bag-of-Words dễ bị phạt nhầm sang nhãn Negative.
3. **Từ đồng nghĩa ngoài tập từ điển (OOV - Out of Vocabulary):** Khi khách hàng viết tắt (*k*, *ko*, *hok*, *bth*, *okela*), TF-IDF coi chúng là các từ hoàn toàn xa lạ nếu không có mặt đầy đủ trong từ điển 10.000 đặc trưng.

### 3.4. Ưu thế của kiến trúc TF-IDF + LinearSVC & Tốc độ suy luận
* **Tốc độ suy luận siêu tốc:** Đạt **0.0237 ms / câu** (~42,154 câu / giây), nhanh gấp **~45 lần** so với PhoBERT Multi-Head (1.27 ms/câu trên GPU T4).
* **Tài nguyên siêu nhẹ:** Dung lượng chỉ **~2 MB**, chạy trực tiếp trên CPU mà không cần GPU hay môi trường CUDA phức tạp.
* Rất phù hợp triển khai làm tầng lọc nhanh (fast heuristic filter) hoặc ứng dụng biên (edge computing) với tài nguyên phần cứng hạn chế.

### 3.5. Kết luận dùng cho Báo cáo (report.tex) & Họp nhóm
1. Mô hình 1 (Baseline TF-IDF + LinearSVC) của bạn Phúc đã **hoàn thành 100% nhiệm vụ đặt ra**, cung cấp cột mốc tham chiếu truyền thống vững chắc.
2. Kết quả **Mean Macro-F1 = 64.78%** và **Exact Match Ratio = 43.20%** là cơ sở đối chứng thực nghiệm hoàn hảo để làm nổi bật sự vượt trội của PhoBERT Multi-Head (**75.38% F1**, vượt trội **+10.60%**) trong buổi họp nhóm ngày 15/10/2026 và Báo cáo tổng kết (`report.tex`).
