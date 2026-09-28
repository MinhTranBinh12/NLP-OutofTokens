import csv
import glob
import os


def merge_all_reviews(raw_dir: str = "data/raw", output_path: str = "data/merged_reviews.tsv"):
    """
    Tự động đọc tất cả các file data/raw/reviews_*.tsv của các thành viên,
    lọc trùng lặp theo review_id và gộp thành một file duy nhất.
    """
    if not os.path.exists(raw_dir):
        print(f"[!] Thư mục {raw_dir} không tồn tại!")
        return

    tsv_files = glob.glob(os.path.join(raw_dir, "reviews_*.tsv"))
    if not tsv_files:
        print(f"[!] Không tìm thấy file 'reviews_*.tsv' nào trong '{raw_dir}'!")
        return

    print(f"[*] Tìm thấy {len(tsv_files)} file dữ liệu thành viên:")
    for f in tsv_files:
        print(f"    - {os.path.basename(f)}")

    seen_ids = set()
    combined_rows = []
    fieldnames = None
    stats_per_file = {}

    for file_path in tsv_files:
        file_name = os.path.basename(file_path)
        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if not fieldnames:
                fieldnames = reader.fieldnames
            for row in reader:
                rev_id = row.get("review_id", "").strip()
                if rev_id and rev_id not in seen_ids:
                    seen_ids.add(rev_id)
                    combined_rows.append(row)
                    count += 1
        stats_per_file[file_name] = count

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(combined_rows)

    print("\n=== THỐNG KÊ ĐÓNG GÓP THEO THÀNH VIÊN ===")
    for fname, count in stats_per_file.items():
        print(f"  • {fname}: {count} đánh giá đóng góp")
    print(f"---------------------------------------------")
    print(f"🎉 TỔNG SỐ ĐÁNH GIÁ DUY NHẤT SAU KHI GỘP: {len(combined_rows)}")
    print(f"💾 File tổng hợp đã lưu tại: {output_path}")


if __name__ == "__main__":
    merge_all_reviews()
