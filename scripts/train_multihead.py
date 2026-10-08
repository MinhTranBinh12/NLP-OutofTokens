"""
Script huấn luyện Mô hình 2: PhoBERT Multi-Head (Sentence-level).
Người phụ trách: Minh (nhánh feat/multihead).

Cách chạy (từ thư mục gốc dự án):
    # Smoke test nhanh trên CPU
    python scripts/train_multihead.py --max_samples 64 --epochs 1 --batch_size 8
    # Huấn luyện đầy đủ (Colab GPU) + test 1 lần ở cuối
    python scripts/train_multihead.py --epochs 10 --do_test
    # Chỉ chạy Test bằng best checkpoint đã lưu
    python scripts/train_multihead.py --test_only

Đầu ra:
    saved_models/multihead/best_model.pt   (KHÔNG push - đã chặn bởi .gitignore)
    results/multihead/*.json               (nhẹ - commit để báo cáo nhóm)
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Thêm thư mục gốc dự án vào sys.path để import "src.*"
# (Không thêm "src/" trực tiếp vì "src/datasets" sẽ trùng tên thư viện HuggingFace "datasets")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.datasets.dataset_multihead import (  # noqa: E402
    MultiHeadCollator,
    MultiHeadDataset,
    compute_class_weights,
    label_distribution,
    load_processed_json,
)
from src.metrics import format_metrics, per_aspect_confusion, sentence_level_metrics  # noqa: E402
from src.models.phobert_multihead import MultiHeadWeightedLoss, PhoBERTMultiHead  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Train PhoBERT Multi-Head (Sentence-level ABSA)")
    # Dữ liệu & đường dẫn
    p.add_argument("--data_dir", default=os.path.join(ROOT, "data", "processed"))
    p.add_argument("--output_dir", default=os.path.join(ROOT, "saved_models", "multihead"))
    p.add_argument("--results_dir", default=os.path.join(ROOT, "results", "multihead"))
    p.add_argument("--max_samples", type=int, default=None, help="Giới hạn số mẫu (debug)")
    # Mô hình
    p.add_argument("--model_name", default="vinai/phobert-base-v2")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--weight_smoothing", choices=["inverse", "sqrt", "none"], default="inverse")
    # Huấn luyện
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--encoder_lr", type=float, default=2e-5)
    p.add_argument("--head_lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=3, help="Early stopping (số epoch không cải thiện)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--no_amp", action="store_true", help="Tắt Mixed Precision fp16")
    # Chế độ
    p.add_argument("--do_test", action="store_true", help="Đánh giá Test 1 lần sau khi train xong")
    p.add_argument("--test_only", action="store_true", help="Bỏ qua train, chỉ chạy Test bằng best checkpoint")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def make_loader(path, tokenizer, args, shuffle):
    records = load_processed_json(path, args.max_samples)
    ds = MultiHeadDataset(records, tokenizer, args.max_len)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=MultiHeadCollator(tokenizer),
        num_workers=args.num_workers,
    )
    return records, loader


@torch.no_grad()
def evaluate(model, loader, device, use_amp, loss_fn=None):
    """Chạy dự đoán trên một tập, trả về (metrics, y_true, y_pred, thời gian/câu ms)."""
    model.eval()
    y_true, y_pred, total_loss, n_batches = [], [], 0.0, 0
    start = time.perf_counter()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(input_ids, attention_mask)
        if loss_fn is not None:
            total_loss += loss_fn(logits, labels).item()
            n_batches += 1
        y_pred.append(logits.argmax(dim=-1).cpu())
        y_true.append(labels.cpu())
    elapsed = time.perf_counter() - start

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    metrics = sentence_level_metrics(y_true, y_pred)
    if loss_fn is not None:
        metrics["loss"] = total_loss / max(n_batches, 1)
    ms_per_sentence = elapsed * 1000 / max(len(y_true), 1)
    return metrics, y_true, y_pred, ms_per_sentence


def train(args, tokenizer, device, use_amp):
    train_path = os.path.join(args.data_dir, "train_data_processed.json")
    dev_path = os.path.join(args.data_dir, "dev_data_processed.json")
    train_records, train_loader = make_loader(train_path, tokenizer, args, shuffle=True)
    _, dev_loader = make_loader(dev_path, tokenizer, args, shuffle=False)
    print(f"Train: {len(train_records)} mẫu | Dev: {len(dev_loader.dataset)} mẫu")

    # Class weights tính CHỈ trên tập Train
    class_weights = None
    if args.weight_smoothing != "none":
        class_weights = compute_class_weights(train_records, args.weight_smoothing)
    save_json(
        {
            "train_label_distribution": label_distribution(train_records),
            "class_weights": class_weights.tolist() if class_weights is not None else None,
        },
        os.path.join(args.results_dir, "train_stats.json"),
    )

    model = PhoBERTMultiHead(args.model_name, dropout=args.dropout).to(device)
    loss_fn = MultiHeadWeightedLoss(class_weights).to(device)

    optimizer = torch.optim.AdamW(model.param_groups(args.encoder_lr, args.head_lr, args.weight_decay))
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_ratio * total_steps), num_training_steps=total_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, "best_model.pt")
    best_f1, bad_epochs, history = -1.0, 0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, t0 = 0.0, time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / len(train_loader)
        dev_metrics, _, _, _ = evaluate(model, dev_loader, device, use_amp, loss_fn)
        dev_f1 = dev_metrics["mean_aspect_macro_f1"]

        print(f"\n[Epoch {epoch}] train_loss={train_loss:.4f} | dev_loss={dev_metrics['loss']:.4f} "
              f"| time={time.time() - t0:.0f}s")
        print(format_metrics(dev_metrics))
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"dev_{k}": v for k, v in dev_metrics.items()}})

        # Checkpoint selection theo Mean Aspect Macro-F1 trên Dev
        if dev_f1 > best_f1:
            best_f1, bad_epochs = dev_f1, 0
            torch.save({"model_state": model.state_dict(), "args": vars(args), "epoch": epoch}, best_path)
            save_json({"epoch": epoch, **dev_metrics}, os.path.join(args.results_dir, "dev_best.json"))
            print(f">>> Lưu best model (Dev Mean F1 = {best_f1 * 100:.2f}%)")
        else:
            bad_epochs += 1
            print(f"Không cải thiện ({bad_epochs}/{args.patience})")
            if bad_epochs >= args.patience:
                print("Early stopping!")
                break

        save_json(history, os.path.join(args.results_dir, "train_history.json"))

    print(f"\nHoàn tất huấn luyện. Best Dev Mean Aspect Macro-F1 = {best_f1 * 100:.2f}%")
    return best_path


def test(args, tokenizer, device, use_amp, ckpt_path):
    """Kiểm định cuối cùng: chạy Test DUY NHẤT 1 lần với best checkpoint."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}. Hãy train trước.")
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt.get("args", {})

    model = PhoBERTMultiHead(
        saved_args.get("model_name", args.model_name), dropout=saved_args.get("dropout", args.dropout)
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    test_path = os.path.join(args.data_dir, "test_data_processed.json")
    test_records, test_loader = make_loader(test_path, tokenizer, args, shuffle=False)
    metrics, y_true, y_pred, ms = evaluate(model, test_loader, device, use_amp)
    metrics["inference_ms_per_sentence"] = ms
    metrics["best_epoch"] = ckpt.get("epoch")

    print("\n===== KẾT QUẢ TEST (CHÍNH THỨC) =====")
    print(format_metrics(metrics))
    print(f"Inference: {ms:.2f} ms/câu")

    save_json(metrics, os.path.join(args.results_dir, "test_results.json"))
    save_json(per_aspect_confusion(y_true, y_pred), os.path.join(args.results_dir, "test_confusion.json"))
    save_json(
        [
            {"id": r["id"], "comment": r["comment"], "gold": t.tolist(), "pred": p.tolist()}
            for r, t, p in zip(test_records, y_true, y_pred)
        ],
        os.path.join(args.results_dir, "test_predictions.json"),
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"  # AMP fp16 chỉ bật khi có GPU
    print(f"Device: {device} | AMP: {use_amp}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ckpt_path = os.path.join(args.output_dir, "best_model.pt")

    if not args.test_only:
        ckpt_path = train(args, tokenizer, device, use_amp)
    if args.test_only or args.do_test:
        test(args, tokenizer, device, use_amp, ckpt_path)


if __name__ == "__main__":
    main()
