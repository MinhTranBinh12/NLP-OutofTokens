"""
Kịch bản huấn luyện mô hình PhoBERT + CRF (Span-Level ABSA).

Quy trình:
  1. Nạp dữ liệu tiền xử lý (train, dev, test) với CRFDataset và Fast Tokenizer.
  2. Khởi tạo mô hình PhoBERTCRF với tầng CRF giải mã Viterbi.
  3. Huấn luyện với Differential Learning Rate (tốc độ học riêng cho PhoBERT và CRF)
     kết hợp Mixed Precision (AMP) và Gradient Clipping.
  4. Đánh giá đa chiều sau mỗi Epoch:
     - Cấp độ Span: Exact Span F1, Proportional Span F1, Per-category F1.
     - Cấp độ Câu: Quy đổi Span sang Vector 5 chiều và tính Macro-F1 theo chuẩn nhóm (src.metrics).
  5. Lưu mô hình tốt nhất (best_model.pt) và báo cáo kết quả (results.json).

Cách chạy:
  python scripts/train_crf.py --epochs 15 --batch_size 16 --lr 2e-5 --crf_lr 1e-3
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# Thêm đường dẫn gốc của repository vào sys.path để import đồng bộ
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.datasets.dataset_crf import CRFDataset, crf_collate_fn, get_tokenizer
from src.label_processing import (
    ASPECTS,
    BIO_TAG2ID,
    BIO_TAGS,
    ID2BIO_TAG,
    extracted_spans_to_vector,
)
from src.metrics import format_metrics, sentence_level_metrics
from src.models.phobert_crf import PhoBERTCRF

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Cố định seed để đảm bảo tính tái lập (reproducibility)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_spans_from_bio_predictions(
    predictions: List[List[int]],
    offset_mappings: torch.Tensor,
    comments: List[str],
    id2tag: Dict[int, str],
) -> List[List[Tuple[int, int, str]]]:
    """
    Chuyển đổi chuỗi ID nhãn BIO dự đoán thành danh sách các spans ký tự:
    [(start_char, end_char, 'Aspect#sentiment'), ...]
    """
    batch_spans = []
    batch_size = len(predictions)

    for i in range(batch_size):
        pred_ids = predictions[i]
        offsets = offset_mappings[i].tolist()
        comment_len = len(comments[i])

        spans = []
        curr_label = None
        curr_start = None
        curr_end = None

        seq_len = min(len(pred_ids), len(offsets))
        for t_idx in range(seq_len):
            t_start, t_end = offsets[t_idx]
            if t_start == t_end:  # Special token
                continue

            tag_id = pred_ids[t_idx]
            tag_name = id2tag.get(tag_id, "O")

            if tag_name.startswith("B-"):
                if curr_label is not None and curr_start is not None and curr_end is not None:
                    spans.append((curr_start, curr_end, curr_label))
                curr_label = tag_name[2:]
                curr_start = t_start
                curr_end = t_end
            elif tag_name.startswith("I-"):
                label = tag_name[2:]
                if curr_label == label and curr_end is not None:
                    curr_end = t_end
                else:
                    if curr_label is not None and curr_start is not None and curr_end is not None:
                        spans.append((curr_start, curr_end, curr_label))
                    curr_label = label
                    curr_start = t_start
                    curr_end = t_end
            else:  # "O"
                if curr_label is not None and curr_start is not None and curr_end is not None:
                    spans.append((curr_start, curr_end, curr_label))
                curr_label = None
                curr_start = None
                curr_end = None

        if curr_label is not None and curr_start is not None and curr_end is not None:
            spans.append((curr_start, curr_end, curr_label))

        # Khử trùng lặp và lọc spans hợp lệ
        unique_spans = []
        for s in spans:
            s_tuple = (max(0, s[0]), min(comment_len, s[1]), s[2])
            if s_tuple[1] > s_tuple[0] and s_tuple not in unique_spans:
                unique_spans.append(s_tuple)

        batch_spans.append(unique_spans)

    return batch_spans


def compute_span_f1(
    gold_spans_list: List[List[Tuple[int, int, str]]],
    pred_spans_list: List[List[Tuple[int, int, str]]],
) -> Dict[str, Any]:
    """
    Tính Exact Match F1 và Proportional F1 cho tập các spans.
    """
    total_gold = 0
    total_pred = 0
    exact_tp = 0
    proportional_tp = 0.0

    per_category: Dict[str, Dict[str, float]] = {}

    for gold_spans, pred_spans in zip(gold_spans_list, pred_spans_list):
        gold_set = set(gold_spans)
        pred_set = set(pred_spans)

        total_gold += len(gold_set)
        total_pred += len(pred_set)

        # Exact match TP
        exact_tp += len(gold_set & pred_set)

        # Per category support and TP
        for g in gold_set:
            cat = g[2]
            if cat not in per_category:
                per_category[cat] = {"tp": 0, "fp": 0, "fn": 0, "support": 0}
            per_category[cat]["support"] += 1
            if g in pred_set:
                per_category[cat]["tp"] += 1
            else:
                per_category[cat]["fn"] += 1

        for p in pred_set:
            cat = p[2]
            if cat not in per_category:
                per_category[cat] = {"tp": 0, "fp": 0, "fn": 0, "support": 0}
            if p not in gold_set:
                per_category[cat]["fp"] += 1

        # Proportional overlap match
        for p in pred_set:
            p_start, p_end, p_label = p
            best_overlap = 0.0
            for g in gold_set:
                g_start, g_end, g_label = g
                if p_label == g_label:
                    intersection = max(0, min(p_end, g_end) - max(p_start, g_start))
                    if intersection > 0:
                        overlap = 2.0 * intersection / ((p_end - p_start) + (g_end - g_start))
                        if overlap > best_overlap:
                            best_overlap = overlap
            proportional_tp += best_overlap

    # Exact metrics
    precision = exact_tp / total_pred if total_pred > 0 else 0.0
    recall = exact_tp / total_gold if total_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Proportional metrics
    prop_prec = proportional_tp / total_pred if total_pred > 0 else 0.0
    prop_rec = proportional_tp / total_gold if total_gold > 0 else 0.0
    prop_f1 = (
        2 * prop_prec * prop_rec / (prop_prec + prop_rec)
        if (prop_prec + prop_rec) > 0
        else 0.0
    )

    # Per-category F1
    cat_results = {}
    for cat, stat in sorted(per_category.items()):
        c_tp = stat["tp"]
        c_fp = stat["fp"]
        c_fn = stat["fn"]
        c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        c_rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        c_f1 = 2 * c_prec * c_rec / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0.0
        cat_results[cat] = {
            "precision": c_prec,
            "recall": c_rec,
            "f1": c_f1,
            "support": stat["support"],
        }

    return {
        "precision": precision,
        "recall": recall,
        "exact_f1": f1,
        "proportional_f1": prop_f1,
        "total_gold": total_gold,
        "total_pred": total_pred,
        "per_category": cat_results,
    }


def evaluate(
    model: PhoBERTCRF,
    data_loader: DataLoader,
    device: torch.device,
    id2tag: Dict[int, str],
) -> Dict[str, Any]:
    """
    Đánh giá mô hình trên tập Dev hoặc Test với giải mã Viterbi.
    Tính toán cả Span-level F1 và Sentence-level Macro-F1.
    """
    model.eval()
    total_loss = 0.0
    all_gold_spans = []
    all_pred_spans = []
    all_gold_aspect_vectors = []
    all_pred_aspect_vectors = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            offset_mapping = batch["offset_mapping"]
            gold_spans = batch["original_spans"]
            comments = batch["comments"]
            aspect_vectors = batch["aspect_vector"].tolist()

            # 1. Tính loss
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs["loss"].item()

            # 2. Giải mã Viterbi chuỗi BIO dự đoán
            emissions = outputs["emissions"]
            predictions = model.decode(emissions, mask=attention_mask)

            # 3. Chuyển BIO thành Spans
            batch_pred_spans = extract_spans_from_bio_predictions(
                predictions, offset_mapping, comments, id2tag
            )

            all_gold_spans.extend(gold_spans)
            all_pred_spans.extend(batch_pred_spans)
            all_gold_aspect_vectors.extend(aspect_vectors)

            # 4. Quy đổi Span dự đoán thành Vector 5 chiều cho đánh giá cấp độ câu
            for spans in batch_pred_spans:
                dict_spans = [
                    {"aspect": s[2].split("#")[0], "sentiment": s[2].split("#")[1]}
                    for s in spans
                    if "#" in s[2]
                ]
                pred_vec = extracted_spans_to_vector(dict_spans)
                all_pred_aspect_vectors.append(pred_vec)

    avg_loss = total_loss / max(len(data_loader), 1)

    # Tính độ đo Span-Level
    span_metrics = compute_span_f1(all_gold_spans, all_pred_spans)

    # Tính độ đo Sentence-Level dùng chung cho nhóm
    sentence_metrics = sentence_level_metrics(
        all_gold_aspect_vectors, all_pred_aspect_vectors
    )

    return {
        "loss": avg_loss,
        "span_metrics": span_metrics,
        "sentence_metrics": sentence_metrics,
    }


def get_param_groups(
    model: PhoBERTCRF, lr: float, crf_lr: float, weight_decay: float = 0.01
) -> List[Dict]:
    """Phân tách tham số để áp dụng Differential Learning Rate."""
    no_decay = ["bias", "LayerNorm.weight"]
    crf_params = list(model.crf.parameters()) + list(model.classifier.parameters())
    crf_param_ids = set(id(p) for p in crf_params)

    encoder_params = [p for p in model.parameters() if id(p) not in crf_param_ids]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for p in encoder_params
                if not any(nd in p.name if hasattr(p, "name") else "" for nd in no_decay)
            ],
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": [
                p
                for p in encoder_params
                if any(nd in p.name if hasattr(p, "name") else "" for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": crf_params,
            "weight_decay": weight_decay,
            "lr": crf_lr,
        },
    ]
    return optimizer_grouped_parameters


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình PhoBERT + CRF cho ABSA")
    parser.add_argument(
        "--train_data",
        default="data/processed/train_data_processed.json",
        help="Đường dẫn file train JSON",
    )
    parser.add_argument(
        "--dev_data",
        default="data/processed/dev_data_processed.json",
        help="Đường dẫn file dev JSON",
    )
    parser.add_argument(
        "--test_data",
        default="data/processed/test_data_processed.json",
        help="Đường dẫn file test JSON",
    )
    parser.add_argument("--model_name", default="vinai/phobert-base-v2", help="PhoBERT checkpoint")
    parser.add_argument("--output_dir", default="output/crf", help="Thư mục lưu checkpoints")
    parser.add_argument("--max_length", type=int, default=256, help="Chiều dài tối đa token")
    parser.add_argument("--batch_size", type=int, default=16, help="Kích thước batch")
    parser.add_argument("--epochs", type=int, default=15, help="Số lượng epoch")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate cho PhoBERT")
    parser.add_argument("--crf_lr", type=float, default=1e-3, help="Learning rate cho CRF layer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Dùng Mixed Precision")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Khởi tạo huấn luyện PhoBERT + CRF | Thiết bị: {device}")

    # 1. Khởi tạo Tokenizer và Datasets
    tokenizer = get_tokenizer(args.model_name)
    train_dataset = CRFDataset(args.train_data, tokenizer=tokenizer, max_length=args.max_length)
    dev_dataset = CRFDataset(args.dev_data, tokenizer=tokenizer, max_length=args.max_length)
    test_dataset = (
        CRFDataset(args.test_data, tokenizer=tokenizer, max_length=args.max_length)
        if os.path.isfile(args.test_data)
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=crf_collate_fn,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=crf_collate_fn,
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=crf_collate_fn,
        )
        if test_dataset
        else None
    )

    # 2. Khởi tạo mô hình PhoBERTCRF
    vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else None
    model = PhoBERTCRF(
        model_name=args.model_name,
        num_labels=len(BIO_TAGS),
        dropout=0.1,
        vocab_size=vocab_size,
    ).to(device)

    # 3. Tối ưu hóa & Lịch trình học
    param_groups = get_param_groups(model, lr=args.lr, crf_lr=args.crf_lr)
    optimizer = torch.optim.AdamW(param_groups)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(args.use_amp and device.type == "cuda"))

    best_dev_f1 = -1.0
    best_epoch = -1
    best_model_path = os.path.join(args.output_dir, "best_model.pt")
    history = []

    logger.info("Bắt đầu vòng lặp huấn luyện...")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(args.use_amp and device.type == "cuda")):
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(len(train_loader), 1)

        # Đánh giá sau mỗi epoch trên tập dev
        eval_res = evaluate(model, dev_loader, device, ID2BIO_TAG)
        span_f1 = eval_res["span_metrics"]["exact_f1"]
        prop_f1 = eval_res["span_metrics"]["proportional_f1"]
        mean_macro_f1 = eval_res["sentence_metrics"]["mean_aspect_macro_f1"]
        duration = time.time() - start_time

        logger.info(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | Dev Loss: {eval_res['loss']:.4f} | "
            f"Exact F1: {span_f1:.4f} | Prop F1: {prop_f1:.4f} | "
            f"Sentence Mean F1: {mean_macro_f1:.4f} | Time: {duration:.0f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "dev_loss": eval_res["loss"],
            "dev_exact_f1": span_f1,
            "dev_proportional_f1": prop_f1,
            "dev_sentence_macro_f1": mean_macro_f1,
            "time_seconds": duration,
        })

        # Lưu checkpoint tốt nhất dựa trên Span Exact F1
        if span_f1 > best_dev_f1:
            best_dev_f1 = span_f1
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_dev_f1": best_dev_f1,
                    "args": vars(args),
                    "vocab_size": vocab_size,
                },
                best_model_path,
            )
            logger.info(f"  -> Đã lưu mô hình tốt nhất (Epoch {epoch}) với Dev F1: {best_dev_f1:.4f}")

    logger.info(f"Huấn luyện hoàn tất! Best Epoch: {best_epoch} (Dev F1: {best_dev_f1:.4f})")

    # 4. Đánh giá kiểm thử với Checkpoint tốt nhất
    if test_loader is not None and os.path.exists(best_model_path):
        logger.info("Đang nạp mô hình tốt nhất để đánh giá trên tập Test...")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_res = evaluate(model, test_loader, device, ID2BIO_TAG)

        print("\n" + "=" * 55)
        print("  KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (MÔ HÌNH CRF)")
        print("=" * 55)
        print(f"Span Exact F1      : {test_res['span_metrics']['exact_f1'] * 100:.2f}%")
        print(f"Span Precision     : {test_res['span_metrics']['precision'] * 100:.2f}%")
        print(f"Span Recall        : {test_res['span_metrics']['recall'] * 100:.2f}%")
        print(f"Span Proportional F1: {test_res['span_metrics']['proportional_f1'] * 100:.2f}%")
        print("-" * 55)
        print("Sentence-level Aspect Macro-F1 (So sánh với nhóm):")
        print(format_metrics(test_res["sentence_metrics"]))
        print("=" * 55 + "\n")

        # Lưu kết quả đầy đủ vào JSON
        results_file = os.path.join(args.output_dir, "results.json")
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_epoch": best_epoch,
                    "best_dev_f1": best_dev_f1,
                    "test_span_metrics": test_res["span_metrics"],
                    "test_sentence_metrics": test_res["sentence_metrics"],
                    "training_history": history,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info(f"Đã lưu kết quả tại: {results_file}")


if __name__ == "__main__":
    main()
