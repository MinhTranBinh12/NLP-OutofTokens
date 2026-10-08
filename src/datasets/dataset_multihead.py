"""
Dataset cho Mô hình 2: PhoBERT Multi-Head (Sentence-level).
Người phụ trách: Minh (nhánh feat/multihead).

- Đọc dữ liệu đã tiền xử lý tại data/processed/*_processed.json
  (dùng trường "comment" và "aspect_vector" [5 phần tử, giá trị 0..3]).
- Dynamic Padding: chỉ pad theo câu dài nhất trong từng batch.
- Tính class weights (tần suất nghịch đảo) cho từng khía cạnh từ tập Train.
"""

import json
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

ASPECTS: List[str] = ["Material", "General", "Service", "Design", "Price"]
NUM_ASPECTS: int = 5
NUM_CLASSES: int = 4  # 0: None, 1: positive, 2: neutral, 3: negative


def load_processed_json(path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Đọc file JSON đã xử lý, chỉ giữ lại các trường cần cho Multi-Head."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if max_samples is not None:
        data = data[:max_samples]
    return [
        {"id": item.get("id", i), "comment": item["comment"], "aspect_vector": item["aspect_vector"]}
        for i, item in enumerate(data)
    ]


class MultiHeadDataset(Dataset):
    """Mỗi mẫu trả về input_ids, attention_mask (chưa pad) và labels [5]."""

    def __init__(self, records: List[Dict[str, Any]], tokenizer, max_len: int = 256):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        enc = self.tokenizer(
            rec["comment"],
            truncation=True,
            max_length=self.max_len,
            padding=False,  # Không pad ở đây -> để collate_fn pad động
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": rec["aspect_vector"],
        }


class MultiHeadCollator:
    """Dynamic Padding: pad input theo câu dài nhất trong batch, stack labels thành [B, 5]."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
        features = [{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]} for b in batch]
        padded = self.tokenizer.pad(features, padding="longest", return_tensors="pt")
        padded["labels"] = labels
        return padded


def compute_class_weights(records: List[Dict[str, Any]], smoothing: str = "inverse") -> torch.Tensor:
    """
    Tính trọng số lớp cho từng khía cạnh từ tập Train.
    Trả về tensor [5, 4].

    smoothing:
      - "inverse": w_c = N / (C * n_c)          (tần suất nghịch đảo chuẩn)
      - "sqrt":    w_c = sqrt(N / (C * n_c))    (mềm hơn, tránh trọng số quá lớn)
    """
    counts = torch.zeros(NUM_ASPECTS, NUM_CLASSES, dtype=torch.float)
    for rec in records:
        for k, y in enumerate(rec["aspect_vector"]):
            counts[k, y] += 1

    n_total = float(len(records))
    counts = counts.clamp(min=1.0)  # tránh chia cho 0 nếu một lớp không xuất hiện
    weights = n_total / (NUM_CLASSES * counts)
    if smoothing == "sqrt":
        weights = weights.sqrt()
    return weights


def label_distribution(records: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """Thống kê số mẫu mỗi lớp theo từng khía cạnh (phục vụ báo cáo)."""
    dist = {asp: [0] * NUM_CLASSES for asp in ASPECTS}
    for rec in records:
        for k, y in enumerate(rec["aspect_vector"]):
            dist[ASPECTS[k]][y] += 1
    return dist
