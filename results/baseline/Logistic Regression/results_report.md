# 📊 Báo Cáo Giải Thích Kết Quả Huấn Luyện & Đánh Giá: TF-IDF + Logistic Regression

> **Mô hình:** TF-IDF + Logistic Regression (Sentence-Level ABSA)  
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

1. **Hàm mất mát Cross-Entropy (Multinomial / Log Loss) kết hợp chuẩn hóa L2:**

   $$\min_{\mathbf{w}} \frac{1}{2} \mathbf{w}^T \mathbf{w} - C \sum_{i=1}^N \sum_{c=0}^3 \mathbb{I}(y_i=c) \log P(y_i=c | \mathbf{x}_i)$$

   Trong đó $P(y_i=c | \mathbf{x}_i) = \frac{\exp(\mathbf{w}_c^T \mathbf{x}_i)}{\sum_j \exp(\mathbf{w}_j^T \mathbf{x}_i)}$.

2. **Thuật toán Tối ưu L-BFGS (Quasi-Newton):**
   Xấp xỉ ma trận nghịch đảo Hessian bậc 2 trên toàn bộ 5.000 mẫu cùng lúc trong mỗi lần lặp (iterations), dừng khi độ suy giảm hàm mất mát nhỏ hơn dung sai $tol=10^{-4}$.


| Khía cạnh (Aspect) | Thuật toán Solver | Số lần lặp (`n_iter_`) | Giới hạn tối đa (`max_iter`) | Ngưỡng dung sai (`tol`) | Trạng thái hội tụ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Material** | L-BFGS (Quasi-Newton) | **68** | 1000 | 0.0001 | ✅ Đã hội tụ (Converged) |
| **General** | L-BFGS (Quasi-Newton) | **68** | 1000 | 0.0001 | ✅ Đã hội tụ (Converged) |
| **Service** | L-BFGS (Quasi-Newton) | **42** | 1000 | 0.0001 | ✅ Đã hội tụ (Converged) |
| **Design** | L-BFGS (Quasi-Newton) | **66** | 1000 | 0.0001 | ✅ Đã hội tụ (Converged) |
| **Price** | L-BFGS (Quasi-Newton) | **54** | 1000 | 0.0001 | ✅ Đã hội tụ (Converged) |

> [!NOTE]
> Mô hình tối ưu hóa trực tiếp trên toàn bộ ma trận 5.000 mẫu trong một quy trình duy nhất. Cả 5 bộ phân loại của các khía cạnh đều **hội tụ tối ưu 100%** và dừng rất sớm trước giới hạn `max_iter`, không cần và không sử dụng cơ chế chia epoch theo kiểu Deep Learning.

### 2.3. File `dev_best.json` - Checkpoint tối ưu trên Dev

```json
{
  "epoch": 1,
  "Material_macro_f1": 0.7458927989197492,
  "Material_acc": 0.827,
  "General_macro_f1": 0.6410597674151267,
  "General_acc": 0.746,
  "Service_macro_f1": 0.6278800836983558,
  "Service_acc": 0.919,
  "Design_macro_f1": 0.6814830795192343,
  "Design_acc": 0.795,
  "Price_macro_f1": 0.679384350879045,
  "Price_acc": 0.921,
  "mean_aspect_macro_f1": 0.6751400160863023,
  "exact_match_ratio": 0.439
}
```

### 2.4. File `test_results.json` - Kết quả kiểm thử chính thức

| Khía cạnh (Aspect) | Accuracy (%) | Macro-F1 (%) |
| :--- | :---: | :---: |
| **Material** | 78.60% | 68.61% |
| **General** | 73.40% | 58.94% |
| **Service** | 92.30% | 74.31% |
| **Design** | 75.80% | 64.98% |
| **Price** | 92.20% | 62.33% |
| **MEAN ASPECT MACRO-F1** | — | **65.84%** |
| **EXACT MATCH RATIO** | **40.90%** | — |
| **TỐC ĐỘ SUY LUẬN** | — | **0.0778 ms / câu** (~12,856 câu/giây) |

### 2.5. File `test_confusion.json` - Ma trận nhầm lẫn 4x4

#### Material
```text
Thực tế \ Dự đoán     None (0)   Pos (1)   Neu (2)   Neg (3)
None (0):               279        32         5        24
Positive (1):            41       373        19         8
Neutral (2):              6         7        27        27
Negative (3):            15         8        22       107
```

#### General
```text
Thực tế \ Dự đoán     None (0)   Pos (1)   Neu (2)   Neg (3)
None (0):               312        77        32        35
Positive (1):            70       371        11         3
Neutral (2):             12         3        11         3
Negative (3):            14         1         5        40
```

#### Service
```text
Thực tế \ Dự đoán     None (0)   Pos (1)   Neu (2)   Neg (3)
None (0):               560        13         0         7
Positive (1):            31       338         1         3
Neutral (2):              0         3         3         1
Negative (3):            11         6         1        22
```

#### Design
```text
Thực tế \ Dự đoán     None (0)   Pos (1)   Neu (2)   Neg (3)
None (0):               454        45        14        50
Positive (1):            46       179        13         6
Neutral (2):              7         6        22        17
Negative (3):             9        11        18       103
```

#### Price
```text
Thực tế \ Dự đoán     None (0)   Pos (1)   Neu (2)   Neg (3)
None (0):               710        16        11         0
Positive (1):            16       175        16         0
Neutral (2):              4         3        37         1
Negative (3):             6         2         3         0
```

### 2.6. File `test_predictions.json` - Mẫu ví dụ

* **Ví dụ đoán đúng 5/5 khía cạnh:**
  - *"chất liệu:không đảm bảo"*
  - Gold: `[3, 0, 0, 0, 0]` | Pred: `[3, 0, 0, 0, 0]`

* **Ví dụ đoán sai:**
  - *"Chất liệu:vải Màu sắc:trắng Đúng với mô tả:80% hàng bị lỗi 1 xíu shop nhiệt tình tư vấn đổi trả vải đẹp hàng giống mẫu mng nên ủng hộ"*
  - Gold: `[1, 0, 1, 0, 0]` | Pred: `[0, 0, 1, 0, 0]`

---

## 3. Nhận Xét & Đánh Giá Toàn Diện Mô Hình
* **Tổng quan hiệu năng:** Mô hình đạt **Mean Aspect Macro-F1 = 65.84%** và **Exact Match Ratio = 40.90%**.
* **Độ lệch Dev - Test:** Dev Macro-F1 (67.51%) vs Test Macro-F1 (65.84%), chênh lệch 1.68%.
* **Tốc độ suy luận:** Đạt 0.0778 ms/câu, cực nhanh và đáp ứng tốt điều kiện môi trường tài nguyên thấp.
