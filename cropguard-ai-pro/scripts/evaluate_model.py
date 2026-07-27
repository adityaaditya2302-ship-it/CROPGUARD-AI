"""
CropGuard AI - Model Evaluator
Runs detailed evaluation on the test set after training.

Reports:
  - Per-class accuracy
  - Confusion matrix
  - Top-5 accuracy
  - Worst performing classes
  - Sample misclassifications

Usage:
  python scripts/evaluate_model.py \
    --model  static/models/cropguard_efficientnet_best.pt \
    --data   datasets/combined \
    --output evaluation_results/
"""
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np


VAL_TRANSFORMS = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(checkpoint_path: str):
    """Load a saved CropGuard model checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_name  = ckpt.get("model_name", "efficientnet")
    n_classes   = ckpt.get("n_classes", 38)
    class_names = ckpt.get("class_names", [])

    # Build model
    import torchvision.models as models
    import torch.nn as nn

    if model_name == "efficientnet":
        base = models.efficientnet_v2_s(weights=None)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 1024),
            nn.SiLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.15),
            nn.Linear(1024, n_classes),
        )
        model = base
    elif model_name == "swin":
        base = models.swin_t(weights=None)
        in_features = base.head.in_features
        base.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 1024),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Linear(1024, n_classes),
        )
        model = base
    else:
        base = models.mobilenet_v3_large(weights=None)
        in_features = base.classifier[3].in_features
        base.classifier[3] = nn.Linear(in_features, n_classes)
        model = base

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, class_names, n_classes


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {args.model}")
    model, class_names, n_classes = load_model(args.model)
    model = model.to(device)

    data_split = "test" if (Path(args.data) / "test").exists() else "val"
    test_ds = ImageFolder(Path(args.data) / data_split, transform=VAL_TRANSFORMS)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    if not class_names:
        class_names = test_ds.classes

    print(f"Evaluating on {len(test_ds)} {data_split} images | {n_classes} classes")

    # Per-class tracking
    class_correct = defaultdict(int)
    class_total   = defaultdict(int)
    all_preds     = []
    all_labels    = []
    all_confs     = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y   = x.to(device), y.to(device)
            logits  = model(x)
            probs   = F.softmax(logits, dim=1)
            preds   = logits.argmax(1)

            for pred, label, prob in zip(preds, y, probs):
                class_correct[label.item()] += (pred == label).item()
                class_total[label.item()]   += 1
                all_preds.append(pred.item())
                all_labels.append(label.item())
                all_confs.append(prob[pred].item() * 100)

    # Overall metrics
    total_correct = sum(class_correct.values())
    total_samples = sum(class_total.values())
    overall_acc   = total_correct / total_samples * 100

    top5_correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y   = x.to(device), y.to(device)
            logits  = model(x)
            topk    = logits.topk(min(5, n_classes), dim=1).indices
            top5_correct += topk.eq(y.unsqueeze(1)).any(dim=1).sum().item()
    top5_acc = top5_correct / total_samples * 100

    # Per-class accuracy
    per_class = {}
    for cls_idx in range(n_classes):
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
        total    = class_total.get(cls_idx, 0)
        correct  = class_correct.get(cls_idx, 0)
        per_class[cls_name] = {
            "accuracy": round(correct / max(total, 1) * 100, 1),
            "correct":  correct,
            "total":    total,
        }

    # Sort by accuracy
    best_classes  = sorted(per_class.items(), key=lambda x: x[1]["accuracy"], reverse=True)[:10]
    worst_classes = sorted(per_class.items(), key=lambda x: x[1]["accuracy"])[:10]

    # Print results
    print("\n" + "=" * 65)
    print(f"  EVALUATION RESULTS")
    print("=" * 65)
    print(f"  Overall Accuracy:  {overall_acc:.2f}%")
    print(f"  Top-5 Accuracy:   {top5_acc:.2f}%")
    print(f"  Avg Confidence:   {np.mean(all_confs):.1f}%")
    print(f"  Total samples:    {total_samples}")
    print(f"  Classes:          {n_classes}")

    print(f"\n✅ Best Performing Classes:")
    for cls, stats in best_classes:
        bar = "█" * int(stats["accuracy"] / 5)
        print(f"  {cls:<40} {stats['accuracy']:5.1f}% {bar}")

    print(f"\n⚠️  Worst Performing Classes (need more data or augmentation):")
    for cls, stats in worst_classes:
        bar = "█" * int(stats["accuracy"] / 5)
        print(f"  {cls:<40} {stats['accuracy']:5.1f}% ({stats['total']} samples)")

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model":          args.model,
        "overall_acc":    round(overall_acc, 2),
        "top5_acc":       round(top5_acc, 2),
        "avg_confidence": round(float(np.mean(all_confs)), 1),
        "total_samples":  total_samples,
        "n_classes":      n_classes,
        "per_class":      per_class,
        "best_classes":   [{"class": c, **s} for c, s in best_classes],
        "worst_classes":  [{"class": c, **s} for c, s in worst_classes],
    }

    result_path = output_dir / "evaluation_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📄 Results saved → {result_path}")
    print(f"\n💡 Tip: Classes with <80% accuracy need more training data.")
    print(f"   Add images for: {', '.join([c for c,s in worst_classes if s['accuracy']<80])}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data",   required=True, help="Path to datasets/combined")
    parser.add_argument("--output", default="evaluation_results")
    args = parser.parse_args()
    evaluate(args)
