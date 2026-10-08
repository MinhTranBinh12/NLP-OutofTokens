"""
Mô hình 2: PhoBERT Multi-Head (Sentence-level).
Người phụ trách: Minh (nhánh feat/multihead).

Kiến trúc:
    Review Text -> PhoBERT-base-v2 Encoder -> [CLS] (768d) -> Dropout(0.3)
                -> 5 Linear Heads độc lập (768 -> 4) -> Softmax
Loss:
    L_total = sum_{k=1..5} CrossEntropy(Z_k, y_k; w_k)   (Weighted CE cho từng head)
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel

ASPECTS: List[str] = ["Material", "General", "Service", "Design", "Price"]


class PhoBERTMultiHead(nn.Module):
    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
        num_aspects: int = 5,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_aspects = num_aspects
        self.num_classes = num_classes

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size  # 768 với base

        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden, num_classes) for _ in range(num_aspects)])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Trả về logits có shape [batch_size, 5, 4]."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # Shared Representation H = vector [CLS]
        h = self.dropout(cls)
        return torch.stack([head(h) for head in self.heads], dim=1)

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Trả về nhãn dự đoán [batch_size, 5] với giá trị trong {0,1,2,3}."""
        return self.forward(input_ids, attention_mask).argmax(dim=-1)

    def param_groups(self, encoder_lr: float, head_lr: float, weight_decay: float = 0.01) -> List[Dict]:
        """
        Differential Learning Rates:
          - Thân PhoBERT: LR nhỏ (giữ tri thức pre-train).
          - 5 Heads: LR lớn (thích nghi nhanh với miền thời trang).
        Không áp dụng weight decay cho bias và LayerNorm.
        """
        no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")

        def split(module: nn.Module, lr: float) -> List[Dict]:
            decay, nodecay = [], []
            for n, p in module.named_parameters():
                if not p.requires_grad:
                    continue
                (nodecay if any(nd in n for nd in no_decay) else decay).append(p)
            return [
                {"params": decay, "lr": lr, "weight_decay": weight_decay},
                {"params": nodecay, "lr": lr, "weight_decay": 0.0},
            ]

        return split(self.encoder, encoder_lr) + split(self.heads, head_lr)


class MultiHeadWeightedLoss(nn.Module):
    """Tổng Weighted Cross-Entropy của 5 heads, mỗi head có bộ trọng số w_k riêng."""

    def __init__(self, class_weights: Optional[torch.Tensor] = None, num_aspects: int = 5):
        super().__init__()
        self.num_aspects = num_aspects
        if class_weights is not None:
            # register_buffer để weights tự động chuyển theo .to(device)
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = logits.float()  # tính loss ở fp32 cho ổn định khi dùng AMP
        total = logits.new_zeros(())
        for k in range(self.num_aspects):
            w = self.class_weights[k] if self.class_weights is not None else None
            total = total + nn.functional.cross_entropy(logits[:, k, :], labels[:, k], weight=w)
        return total
