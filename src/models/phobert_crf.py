"""
Mô hình PhoBERT + CRF cho bài toán Gán nhãn Chuỗi Cấp độ Span (Span-Level ABSA).

Kiến trúc mạng:
    PhoBERT Contextual Encoder -> Dropout -> Linear Classifier -> CRF Layer

CRF (Conditional Random Field) đóng vai trò:
  - Học ma trận chuyển đổi nhãn (Transition Matrix) giữa 31 nhãn BIO.
  - Ngăn chặn các chuỗi nhãn bất hợp lệ (ví dụ: I-X không thể đứng trước B-X, hoặc I-Aspect1 theo sau B-Aspect2).
  - Thuật toán Viterbi giải mã chuỗi nhãn tối ưu toàn cục (Globally Optimal Sequence).
"""

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel

try:
    from torchcrf import CRF
except ImportError:
    raise ImportError(
        "Thư viện 'pytorch-crf' chưa được cài đặt. Vui lòng chạy: pip install pytorch-crf"
    )

logger = logging.getLogger(__name__)


class PhoBERTCRF(nn.Module):
    """
    Kiến trúc kết hợp PhoBERT và tầng CRF cho bài toán Aspect-Based Sentiment Analysis.
    """

    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
        num_labels: int = 31,
        dropout: float = 0.1,
        vocab_size: Optional[int] = None,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.model_name = model_name

        # PhoBERT backbone encoder
        self.encoder = AutoModel.from_pretrained(model_name)

        # Mở rộng embedding table nếu kích thước tokenizer lớn hơn checkpoint mặc định
        if vocab_size is not None and vocab_size > self.encoder.config.vocab_size:
            logger.info(
                f"Resize token embeddings: {self.encoder.config.vocab_size} -> {vocab_size}"
            )
            self.encoder.resize_token_embeddings(vocab_size)

        hidden_size = self.encoder.config.hidden_size

        # Tuyến tính phân loại (Emission scores)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Tầng CRF mô hình hóa phụ thuộc chuyển trạng thái BIO
        self.crf = CRF(num_labels, batch_first=True)

        logger.info(
            f"PhoBERTCRF initialized | Backbone: {model_name} | Hidden: {hidden_size} | "
            f"Labels: {num_labels} | Vocab: {self.encoder.config.vocab_size}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        tag_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass của mô hình:
        - Khi huấn luyện (có labels hoặc tag_ids): tính Negative Log-Likelihood loss qua CRF.
        - Khi suy luận (không có labels): giải mã Viterbi trả về chuỗi nhãn dự đoán tối ưu.
        """
        target_labels = labels if labels is not None else tag_ids

        # 1. Trích xuất đặc trưng ngữ cảnh từ PhoBERT
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # 2. Chiếu qua Classifier để lấy điểm phát xạ (emissions)
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)  # [batch_size, seq_len, num_labels]

        crf_mask = attention_mask.bool()
        result: Dict[str, Any] = {"emissions": emissions}

        if target_labels is not None:
            # Giới hạn nhãn an toàn trong khoảng [0, num_labels - 1]
            clamped_labels = target_labels.clamp(min=0, max=self.num_labels - 1)
            # CRF tính log-likelihood, loss = -log_likelihood
            loss = -self.crf(emissions, clamped_labels, mask=crf_mask, reduction="mean")
            result["loss"] = loss
        else:
            predictions = self.crf.decode(emissions, mask=crf_mask)
            result["predictions"] = predictions

        return result

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        """
        Giải mã chuỗi nhãn BIO tối ưu qua thuật toán Viterbi.
        """
        return self.crf.decode(emissions, mask=mask.bool())

    def predict(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> List[List[int]]:
        """
        Hàm tiện ích suy luận nhanh từ tensor đầu vào.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            if "predictions" in outputs:
                return outputs["predictions"]
            return self.decode(outputs["emissions"], attention_mask)
