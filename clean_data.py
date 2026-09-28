import csv
import os
import shutil


def clean_and_organize(
    input_tsv: str = "output/reviews_raw.tsv",
    target_tsv: str = "data/raw/reviews_minh.tsv",
):
    if not os.path.exists(input_tsv):
        print(f"[!] Không tìm thấy tệp: {input_tsv}")
        return

    print(f"[*] Đang đọc và làm sạch dữ liệu từ: {input_tsv}")
    with open(input_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames
        rows = list(reader)

    total_rows = len(rows)
    seen_ids = set()
    unique_rows = []

    for r in rows:
        rev_id = r.get("review_id", "").strip()
        if not rev_id:
            continue
        if rev_id not in seen_ids:
            seen_ids.add(rev_id)
            unique_rows.append(r)

    # 1. Lưu lại file output sạch
    with open(input_tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(unique_rows)

    # 2. Xuất sang thư mục data/raw/ cho thành viên Minh
    os.makedirs(os.path.dirname(target_tsv), exist_ok=True)
    with open(target_tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(unique_rows)

    print(f"[✓] ĐÃ LỌC TRÙNG VÀ ĐÓNG GÓI THÀNH CÔNG!")
    print(f"  - Tổng số mẫu ban đầu: {total_rows}")
    print(f"  - Số bản ghi trùng đã xoá: {total_rows - len(unique_rows)}")
    print(f"  - Số đánh giá chuẩn duy nhất: {len(unique_rows)}")
    print(f"  - Đã xuất file để push lên GitHub: {target_tsv}")


if __name__ == "__main__":
    clean_and_organize()
