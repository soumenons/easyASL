"""
train.py

Training loop for the ASL Transformer classifier.

Usage:
    python train.py --landmark_dir data/landmarks --num_classes 100
    python train.py --landmark_dir data/landmarks --num_classes 100 --resume checkpoints/best.pt
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import make_dataloaders
from model import build_model
import os
# Required on Windows to avoid OpenMP conflict between numpy/PyTorch and OpenCV/MediaPipe
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ---------------------------------------------------------------------------
# Train / eval passes
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for seqs, pad_masks, labels in tqdm(loader, leave=False,
                                             desc="train" if train else "val"):
            seqs      = seqs.to(device)        # (B, T, F)
            pad_masks = pad_masks.to(device)   # (B, T)
            labels    = labels.to(device)      # (B,)

            logits = model(seqs, src_key_padding_mask=pad_masks)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            preds = logits.argmax(dim=-1)
            total_loss    += loss.item() * len(labels)
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)

    return total_loss / total_samples, total_correct / total_samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Data
    train_dl, val_dl, test_dl = make_dataloaders(
        landmark_dir=args.landmark_dir,
        batch_size=args.batch_size,
        max_len=args.max_len,
        num_workers=args.num_workers,
    )

    # Model
    model = build_model(
        num_classes=args.num_classes,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        max_len=args.max_len,
        dropout=args.dropout,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss, optimiser, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Optionally resume
    start_epoch = 0
    best_val_acc = 0.0
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.4f}")

    # Training loop
    history = []
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = run_epoch(model, train_dl, criterion, optimizer, device, train=True)
        val_loss,   val_acc   = run_epoch(model, val_dl,   criterion, optimizer, device, train=False)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
            f"val loss: {val_loss:.4f} acc: {val_acc:.4f} | "
            f"lr: {lr_now:.2e}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss,     "val_acc": val_acc,
        })

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "args": vars(args),
            }, checkpoint_dir / "best.pt")
            print(f"  ✓ New best val acc: {best_val_acc:.4f} — checkpoint saved")

        # Always save latest
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "args": vars(args),
        }, checkpoint_dir / "latest.pt")

    # Save training history
    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final test evaluation
    print("\nEvaluating best model on test set...")
    best_ckpt = torch.load(checkpoint_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model"])
    test_loss, test_acc = run_epoch(model, test_dl, criterion, optimizer, device, train=False)
    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASL Transformer")

    # Data
    parser.add_argument("--landmark_dir",   type=Path, default=Path("data/landmarks"))
    parser.add_argument("--max_len",        type=int,  default=150)
    parser.add_argument("--num_workers",    type=int,  default=4)

    # Model
    parser.add_argument("--num_classes",    type=int,  required=True)
    parser.add_argument("--d_model",        type=int,  default=256)
    parser.add_argument("--num_heads",      type=int,  default=8)
    parser.add_argument("--num_layers",     type=int,  default=4)
    parser.add_argument("--ffn_dim",        type=int,  default=512)
    parser.add_argument("--dropout",        type=float,default=0.1)

    # Training
    parser.add_argument("--epochs",         type=int,  default=100)
    parser.add_argument("--batch_size",     type=int,  default=32)
    parser.add_argument("--lr",             type=float,default=1e-4)
    parser.add_argument("--checkpoint_dir", type=str,  default="checkpoints")
    parser.add_argument("--resume",         type=str,  default=None)

    args = parser.parse_args()
    main(args)