# 📊 Báo Cáo Giải Thích Kết Quả Huấn Luyện & Đánh Giá: TF-IDF + Naive Bayes (MultinomialNB)

> **Mô hình:** TF-IDF + Naive Bayes (MultinomialNB) (Sentence-Level ABSA)  
> **Người phụ trách:** Phúc (`feat/baseline`)  
> **Tập dữ liệu:** ViCloABSA (5.000 Train | 1.000 Dev | 1.000 Test)  

---

## 📑 Mục Lục
- [1. Quy Ước Chung Về Nhãn & Khía Cạnh](#1-quy-ước-chung-về-nhãn--khía-cạnh)
- [2. Giải Thích Chi Tiết Từng File Kết Quả (.json)](#2-giải-thích-chi-tiết-từng-file-kết-quả-json)
- [3. Nhận Xét & Đánh Giá Toàn Diện Mô Hình](#3-nhận-xét--đánh-giá-toàn-diện-mô-hình)

---

## 1. Quy Ước Chung Về Nhãn & Khía Cạnh

Hệ thống thống nhất chuẩn biểu diễn vector 5 chiều đồng bộ:
* **5 Khía cạnh:** Material (0), General (1), Service (2), Design (3), Price (4)
* **4 Nhãn cảm xúc:** `0: None`, `1: Positive`, `2: Neutral`, `3: Negative`

---

## 2. Giải Thích Chi Tiết Từng File Kết Quả (.json)

### 2.1. File `train_stats.json` - Phân bố nhãn & Trọng số phạt

| Khía cạnh (Aspect) | 0: None | 1: Positive | 2: Neutral | 3: Negative | Tổng số | Nhận xét độ lệch |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Material** | 1,750 (35.0%) | 2,230 (44.6%) | 321 (6.42%) | 699 (13.98%) | 5,000 | Tương đối đồng đều |
| **General** | 2,253 (45.1%) | 2,334 (46.7%) | 156 (3.12%) | 257 (5.14%) | 5,000 | Lớp 0 & 1 chiếm 91.8% |
| **Service** | 2,848 (57.0%) | 1,877 (37.5%) | 46 (0.92%) | 229 (4.58%) | 5,000 | Lớp Neutral cực hiếm (0.92%) |
| **Design** | 2,982 (59.6%) | 1,188 (23.8%) | 226 (4.52%) | 604 (12.08%) | 5,000 | Lớp 0 chiếm gần 60% |
| **Price** | 3,626 (72.5%) | 1,123 (22.5%) | 220 (4.40%) | 31 (0.62%) | 5,000 | **Lớp Negative siêu hiếm (0.62%)** |

### 2.2. File `train_history.json` - Số Lần Lặp (Iterations) & Báo Cáo Hội Tụ Của Solver

1. **Mô hình Tạo sinh (Generative Probabilistic Model):**
   Dựa trên định lý Bayes và giả định độc lập có điều kiện giữa các từ:

   $$P(c | \mathbf{x}) \propto P(c) \prod_{j=1}^M P(w_j | c)^{x_j}$$

2. **Ước lượng Hợp lý Cực đại với Làm mịn Laplace/Lidstone:**
   $$P(w_j | c) = \frac{N_{c, w_j} + \alpha}{\sum_{w'} N_{c, w'} + \alpha M}$$

   Mô hình được huấn luyện giải tích bằng phép đếm tần suất và làm mịn xác suất, không cần thuật toán tối ưu lặp.


| Khía cạnh (Aspect) | Thuật toán Solver | Số lần lặp (`n_iter_`) | Giới hạn tối đa (`max_iter`) | Ngưỡng dung sai (`tol`) | Trạng thái hội tụ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Material** | Closed-Form (Giải tích) | N/A (1 lượt quét) | 1 | — | ✅ Tính toán tức thì |
| **General** | Closed-Form (Giải tích) | N/A (1 lượt quét) | 1 | — | ✅ Tính toán tức thì |
| **Service** | Closed-Form (Giải tích) | N/A (1 lượt quét) | 1 | — | ✅ Tính toán tức thì |
| **Design** | Closed-Form (Giải tích) | N/A (1 lượt quét) | 1 | — | ✅ Tính toán tức thì |
| **Price** | Closed-Form (Giải tích) | N/A (1 lượt quét) | 1 | — | ✅ Tính toán tức thì |

> [!NOTE]
> Mô hình tối ưu hóa trực tiếp trên toàn bộ ma trận 5.000 mẫu trong một quy trình duy nhất. Cả 5 bộ phân loại của các khía cạnh đều **hội tụ tối ưu 100%** và dừng rất sớm trước giới hạn `max_iter`, không cần và không sử dụng cơ chế chia epoch theo kiểu Deep Learning.

### 2.3. File `dev_best.json` - Checkpoint tối ưu trên Dev

```json
{
  "epoch": 1,
  "Material_macro_f1": 0.5542605973392861,
  "Material_acc": 0.704,
  "General_macro_f1": 0.4118796248772716,
  "General_acc": 0.682,
  "Service_macro_f1": 0.5713592571976929,
  "Service_acc": 0.872,
  "Design_macro_f1": 0.5244759348548353,
  "Design_acc": 0.756,
  "Price_macro_f1": 0.5684194011711412,
  "Price_acc": 0.878,
  "mean_aspect_macro_f1": 0.5260789630880455,
  "exact_match_ratio": 0.287
}
```

### 2.4. File `test_results.json` - Kết quả kiểm thử chính thức

| Khía cạnh (Aspect) | Accuracy (%) | Macro-F1 (%) |
| :--- | :---: | :---: |
| **Material** | 68.10% | 53.11% |
| **General** | 70.00% | 47.47% |
| **Service** | 86.10% | 52.46% |
| **Design** | 71.00% | 51.98% |
| **Price** | 88.10% | 54.31% |
| **MEAN ASPECT MACRO-F1** | — | **51.87%** |
| **EXACT MATCH RATIO** | **26.70%** | — |
| **TỐC ĐỘ SUY LUẬN** | — | **0.1638 ms / câu** (~6,106 câu/giây) |

### 2.5. File `test_confusion.json` - Ma trận nhầm lẫn 4x4

#### Material
```text
Thực tế \ Dự đoán     None (0)   Pos (1)   Neu (2)   Neg (3)
None (0):               177       114         2        47
Positive (1):            39       386         1        15
Neutral (2):              7        15         4        41
Negative (3):            15        18         5       114
```

#### General
```text
Thực tế \ Dự đoán     None (0)   Pos (1)   Neu (2)   Neg (3)
None (0):               322       127         2         5
Positive (1):            91       363         1         0
Neutral (2):             26         0         2         1
Negative (3):            47         0         0        13
```

#### Service
```text
Thực tế \ Dự đoán     None (0)   Pos (1)   Neu (2)   Neg (3)
None (0):               520        58         0         2
Positive (1):            41       332         0         0
Neutral (2):              1         6         0         0
Negative (3):            18        13         0         9
```

#### Design
```text
Thực tế \ Dự đoán     None (0)   Pos (1)   Neu (2)   Neg (3)
None (0):               487        36         1        39
Positive (1):           107       127         1         9
Neutral (2):             16        10         2        24
Negative (3):            31        16         0        94
```

#### Price
```text
Thực tế \ Dự đoán     None (0)   Pos (1)   Neu (2)   Neg (3)
None (0):               718        14         5         0
Positive (1):            54       145         8         0
Neutral (2):             19         8        18         0
Negative (3):             7         3         1         0
```

### 2.6. File `test_predictions.json` - Mẫu ví dụ

* **Ví dụ đoán đúng 5/5 khía cạnh:**
  - *"áo đẹp như mô tả, giao hàng nhanh đong goi cẩn thận, đẹp và sang"*
  - Gold: `[0, 1, 1, 0, 0]` | Pred: `[0, 1, 1, 0, 0]`

* **Ví dụ đoán sai:**
  - *"chất liệu:không đảm bảo"*
  - Gold: `[3, 0, 0, 0, 0]` | Pred: `[0, 1, 0, 0, 0]`

---

## 3. Nhận Xét & Đánh Giá Toàn Diện Mô Hình
* **Tổng quan hiệu năng:** Mô hình đạt **Mean Aspect Macro-F1 = 51.87%** và **Exact Match Ratio = 26.70%**.
* **Độ lệch Dev - Test:** Dev Macro-F1 (52.61%) vs Test Macro-F1 (51.87%), chênh lệch 0.74%.
* **Tốc độ suy luận:** Đạt 0.1638 ms/câu, cực nhanh và đáp ứng tốt điều kiện môi trường tài nguyên thấp.
