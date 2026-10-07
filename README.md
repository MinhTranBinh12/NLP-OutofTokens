# NLP-OutofTokens: Phân tích Cảm xúc Đa khía cạnh (ViCloABSA)

*(Lịch sử Data Crawling: Đã hoàn thành thu thập dữ liệu thô vào 30/09 và lưu tại `data/raw/`)*

---

## 🎯 GIAI ĐOẠN 2: HUẤN LUYỆN MÔ HÌNH (MODEL TRAINING)

> **🚨 DEADLINE HOÀN THÀNH: 23:59 Ngày 15/10/2026 (1 tuần kể từ hôm nay).**
> **📅 LỊCH TRÌNH TIẾP THEO:** Sau khi kết thúc deadline, nhóm sẽ họp để báo cáo chéo và đối sánh kết quả độ chính xác (F1-score) của cả 3 mô hình.

Giai đoạn này nhóm sẽ chia ra làm 3 mô hình độc lập. Để không bị **Conflict (xung đột code)** khi làm việc nhóm trên GitHub, toàn bộ nhóm **bắt buộc phải tuân thủ nghiêm ngặt** cấu trúc thư mục và quy tắc phân nhánh (branching) dưới đây.

---

## 📂 1. Cấu trúc Cây Thư Mục Làm Việc Mới
Mỗi thành viên chỉ được tạo và sửa file đúng với phần việc của mình trong thư mục `src/` và `scripts/`. **Tuyệt đối không sửa file của người khác.**

```text
NLP-OutofTokens/
├── data/
│   ├── processed/            # CHỈ ĐỌC: Chứa dữ liệu JSON đã tiền xử lý
│   └── raw/                  # Chứa data crawl đợt trước
├── src/
│   ├── label_processing.py   # Code tiền xử lý (đã làm xong)
│   ├── preprocessing.py      # Code tiền xử lý (đã làm xong)
│   ├── datasets/             # NƠI TẠO FILE: Định nghĩa PyTorch Dataset
│   ├── models/               # NƠI TẠO FILE: Định nghĩa Kiến trúc Mô hình (AI/ML)
│   └── metrics.py            # NƠI TẠO FILE: Hàm tính toán F1 chung (Minh)
├── scripts/                  # NƠI TẠO FILE: Chứa file code chạy huấn luyện độc lập
├── .gitignore                # Chặn các file rác và trọng số nặng (model weights)
└── requirements.txt          # (Sắp tạo) File chốt chung version các thư viện
```

---

## 👨‍💻 2. Phân công Nhiệm vụ & Vùng Hoạt động Độc quyền

### 🥇 Thành viên 1 (Phuc): Mô hình Baseline (TF-IDF + LinearSVC)
* **Tên nhánh làm việc (Branch):** `feat/baseline`
* **Vùng tạo file độc quyền (Người khác không được vào):**
  - `src/models/baseline_svc.py` *(Viết class thuật toán ML truyền thống)*
  - `scripts/train_baseline.py` *(Viết code đọc data và thực thi training)*

### 🥈 Thành viên 2 (Minh): Mô hình PhoBERT Multi-Head (Sentence-Level)
* **Tên nhánh làm việc (Branch):** `feat/multihead`
* **Vùng tạo file độc quyền (Người khác không được vào):**
  - `src/datasets/dataset_multihead.py` *(Viết class Dataset đọc nhãn vector 5 chiều)*
  - `src/models/phobert_multihead.py` *(Viết kiến trúc mạng PhoBERT + 5 Linear Heads)*
  - `scripts/train_multihead.py` *(Viết vòng lặp training & eval)*

### 🥉 Thành viên 3 (Cuong): Mô hình PhoBERT + CRF (Span-Level)
* **Tên nhánh làm việc (Branch):** `feat/crf`
* **Vùng tạo file độc quyền (Người khác không được vào):**
  - `src/datasets/dataset_crf.py` *(Viết class Dataset đọc chuỗi 31 nhãn BIO)*
  - `src/models/phobert_crf.py` *(Viết kiến trúc mạng PhoBERT nối với tầng CRF)*
  - `scripts/train_crf.py` *(Viết vòng lặp training & giải mã Viterbi)*

---

## 🚀 3. Quy trình Đẩy Code (Git Workflow) 

**Bước 1: Cập nhật code mới nhất từ nhánh `master` về máy**
```bash
git checkout master
git pull origin master
```

**Bước 2: Tạo nhánh riêng theo phần việc được giao**
*(Ví dụ Thành viên 2 làm MultiHead chạy lệnh này)*
```bash
git checkout -b feat/multihead
```

**Bước 3: Code và Test trên máy cá nhân.** 
*Nhắc lại: Chỉ tạo và làm việc trên các file thuộc phân quyền của mình như mục 2.*

**Bước 4: Commit và Đẩy lên GitHub**
```bash
git add .
git commit -m "Tạo khung sườn cho PhoBERT Multihead"
git push -u origin feat/multihead
```

**Bước 5: Lên trang web GitHub tạo Pull Request (PR)** để yêu cầu hợp nhất nhánh của bạn vào `master`.

---

## ⚠️ 4. NHỮNG ĐIỀU CẤM KỴ ĐỂ TRÁNH SẬP GIT
1. **Không code trực tiếp trên nhánh `master`.**
3. **Không dùng code Python để ghi đè (overwrite) lên các file trong thư mục `data/processed/`.** Nếu cần biến đổi data, hãy thực hiện trên RAM (biến trong file code).
