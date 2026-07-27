"""
CropGuard AI - Ensemble Training Script (Phase 1)
Fine-tunes EfficientNetV2 + Swin Transformer on PlantVillage + custom datasets.

Usage:
    python scripts/train_ensemble.py --model efficientnet --epochs 50 --data datasets/plantvillage
    python scripts/train_ensemble.py --model swin         --epochs 30 --data datasets/rice
    python scripts/train_ensemble.py --model all          --epochs 50 --data datasets/combined
"""
import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime
import json


# ─────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="CropGuard AI Ensemble Trainer")
    parser.add_argument("--model",   choices=["efficientnet", "swin", "all"], default="efficientnet")
    parser.add_argument("--data",    required=True, help="Path to dataset root (ImageFolder structure)")
    parser.add_argument("--epochs",  type=int, default=50)
    parser.add_argument("--batch",   type=int, default=32)
    parser.add_argument("--lr",      type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output",  default="static/models", help="Output directory for weights")
    parser.add_argument("--resume",  default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation split fraction")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Training Transforms
# ─────────────────────────────────────────────
TRAIN_TRANSFORMS = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.RandomRotation(30),
    T.RandomGrayscale(p=0.05),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# Core Trainer
# ─────────────────────────────────────────────
class EnsembleTrainer:
    def __init__(self, args):
        self.args   = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dir = args.output
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"🤖 Trainer initialized | Device: {self.device}")

    def load_dataset(self):
        """Load ImageFolder dataset and split into train/val."""
        full_dataset = ImageFolder(self.args.data, transform=TRAIN_TRANSFORMS)
        n_classes    = len(full_dataset.classes)
        print(f"📂 Dataset: {len(full_dataset)} images, {n_classes} classes")
        print(f"   Classes: {full_dataset.classes[:5]}... (and {max(0,n_classes-5)} more)")

        n_val   = int(len(full_dataset) * self.args.val_split)
        n_train = len(full_dataset) - n_val

        train_ds, val_ds = random_split(full_dataset, [n_train, n_val])
        # Apply proper transforms to val
        val_ds.dataset.transform = VAL_TRANSFORMS

        train_loader = DataLoader(train_ds, batch_size=self.args.batch, shuffle=True,
                                  num_workers=self.args.workers, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.args.batch, shuffle=False,
                                  num_workers=self.args.workers, pin_memory=True)

        self.n_classes = n_classes
        self.class_names = full_dataset.classes
        return train_loader, val_loader

    def build_efficientnet(self):
        from torchvision import models
        base = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, self.n_classes),
        )
        return base

    def build_swin(self):
        from torchvision import models
        base = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        in_features = base.head.in_features
        base.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.n_classes),
        )
        return base

    def train_model(self, model, model_name: str, train_loader, val_loader):
        model = model.to(self.device)
        optimizer = AdamW(model.parameters(), lr=self.args.lr, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_val_acc = 0.0
        history = []
        output_path = os.path.join(self.output_dir, f"cropguard_{model_name}.pt")

        print(f"\n🚀 Training {model_name} for {self.args.epochs} epochs...")
        for epoch in range(1, self.args.epochs + 1):
            # ── Train ────────────────────────────────────────────────────────────
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss    = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss    += loss.item()
                preds          = outputs.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                train_total   += labels.size(0)

                if batch_idx % 20 == 0:
                    print(f"  Epoch {epoch}/{self.args.epochs} | Batch {batch_idx}/{len(train_loader)} | "
                          f"Loss: {loss.item():.4f}", end="\r")

            scheduler.step()

            # ── Validate ──────────────────────────────────────────────────────────
            model.eval()
            val_correct, val_total, val_loss_sum = 0, 0, 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs  = model(images)
                    val_loss_sum += criterion(outputs, labels).item()
                    preds    = outputs.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total   += labels.size(0)

            train_acc = train_correct / train_total * 100
            val_acc   = val_correct   / val_total   * 100
            avg_loss  = train_loss / len(train_loader)

            record = {
                "epoch": epoch,
                "train_loss": round(avg_loss, 4),
                "train_acc":  round(train_acc, 2),
                "val_acc":    round(val_acc,   2),
            }
            history.append(record)
            print(f"  Epoch {epoch:3d}/{self.args.epochs} | "
                  f"Loss: {avg_loss:.4f} | Train acc: {train_acc:.1f}% | Val acc: {val_acc:.1f}%")

            # ── Save best model ───────────────────────────────────────────────────
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state":  optimizer.state_dict(),
                    "val_acc":          val_acc,
                    "n_classes":        self.n_classes,
                    "class_names":      self.class_names,
                    "model_name":       model_name,
                }, output_path)
                print(f"  ✅ Best model saved ({val_acc:.1f}%) → {output_path}")

        # ── Save training history ─────────────────────────────────────────────────
        history_path = output_path.replace(".pt", "_history.json")
        with open(history_path, "w") as f:
            json.dump({"model": model_name, "history": history, "best_val_acc": best_val_acc}, f, indent=2)

        print(f"\n🏆 {model_name} training complete! Best val acc: {best_val_acc:.1f}%")
        return output_path

    def run(self):
        train_loader, val_loader = self.load_dataset()

        if self.args.model in ("efficientnet", "all"):
            print("\n" + "="*60)
            print("PHASE 0 - Training EfficientNetV2")
            print("="*60)
            model = self.build_efficientnet()
            self.train_model(model, "efficientnet_v2", train_loader, val_loader)

        if self.args.model in ("swin", "all"):
            print("\n" + "="*60)
            print("PHASE 1 - Training Swin Transformer")
            print("="*60)
            model = self.build_swin()
            self.train_model(model, "swin_t", train_loader, val_loader)

        print("\n✅ All training complete!")
        print(f"   Weights saved to: {self.output_dir}/")
        print(f"   Update .env: EFFICIENTNET_WEIGHTS=static/models/cropguard_efficientnet_v2.pt")
        print(f"   Update .env: SWIN_WEIGHTS=static/models/cropguard_swin_t.pt")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    args    = parse_args()
    trainer = EnsembleTrainer(args)
    trainer.run()
