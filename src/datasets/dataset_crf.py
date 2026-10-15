"""
PyTorch Dataset cho mô hình PhoBERT + CRF (Span-Level ABSA).

Quản lý dữ liệu chuỗi 31 nhãn BIO:
  - Đọc dữ liệu từ file JSON tiền xử lý (data/processed/*_processed.json)
  - Tokenize văn bản với PhoBERT Fast Tokenizer bảo tồn offset_mapping
  - Căn chỉnh các span thực thể (Aspect#Sentiment) thành chuỗi nhãn BIO tương ứng
  - Cung cấp hàm crf_collate_fn gom batch an toàn cho DataLoader
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download

from src.label_processing import BIO_TAG2ID, ID2BIO_TAG, BIO_TAGS, ASPECTS

logger = logging.getLogger(__name__)


def get_tokenizer(model_name: str = "vinai/phobert-base-v2"):
    """
    Load PhoBERT Fast Tokenizer hỗ trợ offset_mapping.
    Tự động nạp backend tokenizer.json thông qua PreTrainedTokenizerFast.
    """
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if getattr(tok, "is_fast", False):
            return tok
    except Exception:
        pass

    if "phobert" in model_name.lower():
        try:
            tokenizer_file = hf_hub_download(model_name, "tokenizer.json")
            return PreTrainedTokenizerFast(
                tokenizer_file=tokenizer_file,
                bos_token="<s>",
                eos_token="</s>",
                unk_token="<unk>",
                pad_token="<pad>",
                mask_token="<mask>",
            )
        except Exception as e:
            logger.warning(f"Không thể tải tokenizer.json cho {model_name}: {e}")

    return AutoTokenizer.from_pretrained(model_name)


class CRFDataset(Dataset):
    """
    Dataset cho bài toán Span-level ABSA sử dụng biểu diễn BIO Sequence Labeling.
    
    Không gian 31 nhãn BIO:
        O: Ngoài thực thể
        B-Aspect#sentiment: Bắt đầu một khía cạnh mang cảm xúc cụ thể
        I-Aspect#sentiment: Các token tiếp theo trong cụm từ khía cạnh
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[Any] = None,
        max_length: int = 256,
        label2id: Optional[Dict[str, int]] = None,
    ):
        self.data_path = data_path
        self.max_length = max_length
        self.label2id = label2id or BIO_TAG2ID
        self.id2label = {v: k for k, v in self.label2id.items()}

        # Khởi tạo tokenizer nếu chưa truyền vào
        if tokenizer is None:
            self.tokenizer = get_tokenizer("vinai/phobert-base-v2")
        elif isinstance(tokenizer, str):
            self.tokenizer = get_tokenizer(tokenizer)
        else:
            self.tokenizer = tokenizer

        # Đọc dữ liệu từ file JSON
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        logger.info(f"Đã nạp {len(self.data)} mẫu từ {data_path}")

    def __len__(self) -> int:
        return len(self.data)

    def _create_bio_labels(
        self,
        offset_mapping: torch.Tensor,
        attention_mask: torch.Tensor,
        spans: List,
        comment_len: int,
    ) -> torch.Tensor:
        """
        Gán nhãn BIO token-level dựa trên character offset mapping.
        Special tokens (CLS, SEP, PAD) được gán nhãn 0 (O tag).
        """
        seq_len = offset_mapping.size(0)
        labels = torch.zeros(seq_len, dtype=torch.long)

        valid_spans = []
        for s in spans:
            start, end, label = int(s[0]), int(s[1]), s[2]
            start = max(0, min(start, comment_len))
            end = max(0, min(end, comment_len))
            if start < end:
                valid_spans.append((start, end, label))
        valid_spans.sort(key=lambda x: (x[0], x[1]))

        for start, end, label in valid_spans:
            is_first = True
            for token_idx in range(seq_len):
                if attention_mask[token_idx] == 0:
                    continue
                token_start = offset_mapping[token_idx, 0].item()
                token_end = offset_mapping[token_idx, 1].item()
                if token_start == token_end:  # Special token
                    continue
                # Giao thoa giữa token và span
                if max(token_start, start) < min(token_end, end):
                    prefix = "B-" if is_first else "I-"
                    tag_name = f"{prefix}{label}"
                    tag_id = self.label2id.get(tag_name, 0)
                    labels[token_idx] = tag_id
                    is_first = False

        return labels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        comment = item.get("comment", "")
        spans = item.get("spans") or item.get("aspect-sentiment") or []
        aspect_vector = item.get("aspect_vector", [0, 0, 0, 0, 0])

        encoding = self.tokenizer(
            comment,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offset_mapping = encoding["offset_mapping"].squeeze(0)

        # Bảo vệ tràn từ điển
        if hasattr(self.tokenizer, "__len__"):
            tok_size = len(self.tokenizer)
            if (input_ids >= tok_size).any():
                unk_id = getattr(self.tokenizer, "unk_token_id", 3)
                input_ids = torch.where(input_ids >= tok_size, torch.tensor(unk_id, dtype=input_ids.dtype), input_ids)

        labels = self._create_bio_labels(offset_mapping, attention_mask, spans, len(comment))

        comment_len = len(comment)
        original_spans = [
            (max(0, int(s[0])), min(comment_len, int(s[1])), s[2])
            for s in spans
            if int(s[0]) < comment_len and int(s[1]) > int(s[0])
        ]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mask": attention_mask.bool(),
            "labels": labels,
            "tag_ids": labels,
            "aspect_vector": torch.tensor(aspect_vector, dtype=torch.long),
            "offset_mapping": offset_mapping,
            "original_spans": original_spans,
            "comment": comment,
        }


def crf_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function cho DataLoader hỗ trợ gom batch an toàn.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    mask = torch.stack([item["mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    tag_ids = torch.stack([item["tag_ids"] for item in batch])
    aspect_vector = torch.stack([item["aspect_vector"] for item in batch])
    offset_mapping = torch.stack([item["offset_mapping"] for item in batch])

    original_spans = [item["original_spans"] for item in batch]
    comments = [item["comment"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mask": mask,
        "labels": labels,
        "tag_ids": tag_ids,
        "aspect_vector": aspect_vector,
        "offset_mapping": offset_mapping,
        "original_spans": original_spans,
        "comments": comments,
    }
