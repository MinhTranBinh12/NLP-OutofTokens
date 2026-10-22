# 📊 Báo Cáo Giải Thích Kết Quả Huấn Luyện & Đánh Giá: TF-IDF + LinearSVC

> **Mô hình:** TF-IDF + LinearSVC (Sentence-Level ABSA)  
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

1. **Bản chất tối ưu lồi (Convex Optimization):**
   LinearSVC giải bài toán tối ưu hóa lồi có ràng buộc với hàm mất mát Squared Hinge Loss kết hợp chuẩn hóa L2:

   $$\min_{\mathbf{w}} \frac{1}{2} \mathbf{w}^T \mathbf{w} + C \sum_{i=1}^N \max(0, 1 - y_i \mathbf{w}^T \mathbf{x}_i)^2$$

   Hàm mục tiêu này là một **hàm lồi nghiêm ngặt (strictly convex)**, đảm bảo tồn tại một điểm cực tiểu toàn cục duy nhất.

2. **Thuật toán Dual Coordinate Descent (LIBLINEAR):**
   Tối ưu trực tiếp trên bài toán đối ngẫu của toàn bộ ma trận dữ liệu ($5.000 \times 10.000$), dừng khi $\|\nabla f(\mathbf{w})\| \le 10^{-4}$. Không dùng mini-batch hay epochs dạng Deep Learning.


| Khía cạnh (Aspect) | Thuật toán Solver | Số lần lặp (`n_iter_`) | Giới hạn tối đa (`max_iter`) | Ngưỡng dung sai (`tol`) | Trạng thái hội tụ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Material** | LIBLINEAR (Dual Coordinate Descent) | **74** | 2000 | 0.0001 | ✅ Đã hội tụ (Converged) |
| **General** | LIBLINEAR (Dual Coordinate Descent) | **113** | 2000 | 0.0001 | ✅ Đã hội tụ (Converged) |
| **Service** | LIBLINEAR (Dual Coordinate Descent) | **60** | 2000 | 0.0001 | ✅ Đã hội tụ (Converged) |
| **Design** | LIBLINEAR (Dual Coordinate Descent) | **58** | 2000 | 0.0001 | ✅ Đã hội tụ (Converged) |
| **Price** | LIBLINEAR (Dual Coordinate Descent) | **50** | 2000 | 0.0001 | ✅ Đã hội tụ (Converged) |

> [!NOTE]
> Mô hình tối ưu hóa trực tiếp trên toàn bộ ma trận 5.000 mẫu trong một quy trình duy nhất. Cả 5 bộ phân loại của các khía cạnh đều **hội tụ tối ưu 100%** và dừng rất sớm trước giới hạn `max_iter`, không cần và không sử dụng cơ chế chia epoch theo kiểu Deep Learning.

### 2.3. File `dev_best.json` - Checkpoint tối ưu trên Dev

```json
{
  "epoch": 1,
  "Material_macro_f1": 0.7573618735810014,
  "Material_acc": 0.849,
  "General_macro_f1": 0.6161581699772769,
  "General_acc": 0.757,
  "Service_macro_f1": 0.6509708409496074,
  "Service_acc": 0.934,
  "Design_macro_f1": 0.6323780550952776,
  "Design_acc": 0.805,
  "Price_macro_f1": 0.7102483095000555,
  "Price_acc": 0.933,
  "mean_aspect_macro_f1": 0.6734234498206437,
  "exact_match_ratio": 0.481
}
```

### 2.4. File `test_results.json` - Kết quả kiểm thử chính thức

| Khía cạnh (Aspect) | Accuracy (%) | Macro-F1 (%) |
| :--- | :---: | :---: |
| **Material** | 81.50% | 69.15% |
| **General** | 74.60% | 57.65% |
| **Service** | 92.90% | 71.27% |
| **Design** | 78.10% | 64.14% |
| **Price** | 92.70% | 61.68% |
| **MEAN ASPECT MACRO-F1** | — | **64.78%** |
| **EXACT MATCH RATIO** | **43.20%** | — |
| **TỐC ĐỘ SUY LUẬN** | — | **0.0242 ms / câu** (~41,387 câu/giây) |

### 2.5. File `test_confusion.json` - Ma trận nhầm lẫn 4x4

#### Material
```text
Thực tế \ Dự đoán     None (0)   Pos (1)   Neu (2)   Neg (3)
None (0):               295        28         4        13
Positive (1):            37       393         8         3
Neutral (2):              8         9        18        32
Negative (3):            19        10        14       109
```

#### General
```text
Thực tế \ Dự đoán     None (0)   Pos (1)   Neu (2)   Neg (3)
None (0):               341        84        15        16
Positive (1):            79       371         4         1
Neutral (2):             16         5         7         1
Negative (3):            28         2         3        27
```

#### Service
```text
Thực tế \ Dự đoán     None (0)   Pos (1)   Neu (2)   Neg (3)
None (0):               568        11         0         1
Positive (1):            28       343         0         2
Neutral (2):              0         3         2         2
Negative (3):            14        10         0        16
```

#### Design
```text
Thực tế \ Dự đoán     None (0)   Pos (1)   Neu (2)   Neg (3)
None (0):               508        29         5        21
Positive (1):            65       166         9         4
Neutral (2):             12        12        14        14
Negative (3):            21        14        13        93
```

#### Price
```text
Thực tế \ Dự đoán     None (0)   Pos (1)   Neu (2)   Neg (3)
None (0):               720        10         7         0
Positive (1):            21       179         7         0
Neutral (2):              5        12        28         0
Negative (3):             6         3         2         0
```

### 2.6. File `test_predictions.json` - Mẫu ví dụ

* **Ví dụ đoán đúng 5/5 khía cạnh:**
  - *"hình ảnh mang tính chất nhận xu,giáo hàng lâu,giáo nhầm nhà."*
  - Gold: `[0, 0, 3, 0, 0]` | Pred: `[0, 0, 3, 0, 0]`

* **Ví dụ đoán sai:**
  - *"chất liệu:không đảm bảo"*
  - Gold: `[3, 0, 0, 0, 0]` | Pred: `[0, 0, 0, 0, 0]`

---

## 3. Nhận Xét & Đánh Giá Toàn Diện Mô Hình
* **Tổng quan hiệu năng:** Mô hình đạt **Mean Aspect Macro-F1 = 64.78%** và **Exact Match Ratio = 43.20%**.
* **Độ lệch Dev - Test:** Dev Macro-F1 (67.34%) vs Test Macro-F1 (64.78%), chênh lệch 2.56%.
* **Tốc độ suy luận:** Đạt 0.0242 ms/câu, cực nhanh và đáp ứng tốt điều kiện môi trường tài nguyên thấp.
