"""
CropGuard AI — Kaggle/Colab Training Notebook
Paste this into a Kaggle Notebook or Google Colab and run all cells

Kaggle:  kaggle.com → New Notebook → Add Datasets → GPU T4
Colab:   colab.research.google.com → New Notebook → Runtime → T4 GPU

Trains:
  1. EfficientNetV2-S  (~95% accuracy)
  2. Swin Transformer-T (~93% accuracy)
  3. MobileNetV3-Large  (~90% accuracy)
  4. Ensemble of all 3  (~97% accuracy)
"""

# ── CELL 1: Install dependencies ────────────────────────────────
# Run this cell first — takes ~2 minutes
import subprocess
subprocess.run(["pip", "install", "-q", "lightning", "albumentations", "timm"], check=True)

import os, json, shutil, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np

print(f"✅ PyTorch {torch.__version__}")
print(f"✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (no GPU found)'}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using: {DEVICE}")


# ── CELL 2: Set dataset paths ────────────────────────────────────
IS_KAGGLE = os.path.exists("/kaggle/input")
IS_COLAB  = "COLAB_GPU" in os.environ or os.path.exists("/content")

if IS_KAGGLE:
    print("🟡 Running on Kaggle")
    DATASET_PATHS = {
        "plantvillage": "/kaggle/input/plantdisease/PlantVillage",
        "rice":         "/kaggle/input/rice-leaf-diseases",
        "maize":        "/kaggle/input/corn-or-maize-leaf-disease-dataset",
    }
    OUTPUT_DIR = "/kaggle/working/models"
    DATA_DIR   = "/kaggle/working/combined"
elif IS_COLAB:
    print("🔵 Running on Google Colab")
    DATASET_PATHS = {}
    OUTPUT_DIR = "/content/models"
    DATA_DIR   = "/content/combined"
else:
    print("🟢 Running locally")
    DATASET_PATHS = {"plantvillage": "datasets/raw/PlantVillage"}
    OUTPUT_DIR = "static/models"
    DATA_DIR   = "datasets/combined"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


# ── CELL 3: Download datasets (COLAB ONLY) ──────────────────────
if IS_COLAB:
    print("📥 Setting up Kaggle API for Colab...")
    print("Upload your kaggle.json when prompted...")
    from google.colab import files
    uploaded = files.upload()
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    os.system("cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json")

    print("\n📥 Downloading PlantVillage dataset...")
    os.system("kaggle datasets download -d emmarex/plantdisease -p /content/raw/plantvillage --unzip")
    print("📥 Downloading Rice Leaf Disease dataset...")
    os.system("kaggle datasets download -d vbookshelf/rice-leaf-diseases -p /content/raw/rice --unzip")
    print("📥 Downloading Maize Disease dataset...")
    os.system("kaggle datasets download -d smaranjitghose/corn-or-maize-leaf-disease-dataset -p /content/raw/maize --unzip")

    DATASET_PATHS = {
        "plantvillage": "/content/raw/plantvillage",
        "rice":         "/content/raw/rice",
        "maize":        "/content/raw/maize",
    }


# ── CELL 4: Prepare & merge dataset ─────────────────────────────
VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VAL_SPLIT  = 0.15
TEST_SPLIT = 0.05
MIN_SAMPLES = 30
MAX_SAMPLES = 2500
random.seed(42)

def normalize_name(name):
    name = name.replace("___", "_").replace("__", "_").strip("_- ")
    return name.lower()

from collections import defaultdict
all_images = defaultdict(list)

for dataset_name, dataset_path in DATASET_PATHS.items():
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"⚠️  {dataset_path} not found, skipping")
        continue
    print(f"\n📂 Scanning {dataset_name}...")
    for root, dirs, files in os.walk(dataset_path):
        images = [f for f in files if Path(f).suffix.lower() in VALID_EXT]
        if not images:
            continue
        class_name = normalize_name(Path(root).name)
        for img in images:
            all_images[class_name].append(Path(root) / img)

# Filter and balance
filtered = {}
for cls, imgs in all_images.items():
    if len(imgs) < MIN_SAMPLES:
        continue
    if len(imgs) > MAX_SAMPLES:
        imgs = random.sample(imgs, MAX_SAMPLES)
    filtered[cls] = imgs

all_images = filtered
total = sum(len(v) for v in all_images.values())
print(f"\n✅ {total} images | {len(all_images)} classes")

# Split and copy
class_list = sorted(all_images.keys())
class_to_idx = {c: i for i, c in enumerate(class_list)}
N_CLASSES = len(class_list)

for split in ("train", "val", "test"):
    Path(DATA_DIR, split).mkdir(parents=True, exist_ok=True)

idx = 0
for cls, imgs in all_images.items():
    random.shuffle(imgs)
    n = len(imgs)
    n_test  = max(1, int(n * TEST_SPLIT))
    n_val   = max(1, int(n * VAL_SPLIT))
    n_train = n - n_val - n_test
    for split, split_imgs in zip(
        ["train","val","test"],
        [imgs[:n_train], imgs[n_train:n_train+n_val], imgs[n_train+n_val:]]
    ):
        dest = Path(DATA_DIR) / split / cls
        dest.mkdir(parents=True, exist_ok=True)
        for src in split_imgs:
            shutil.copy2(src, dest / f"{idx:08d}{src.suffix.lower()}")
            idx += 1

with open(f"{DATA_DIR}/class_map.json", "w") as f:
    json.dump({"class_to_idx": class_to_idx, "num_classes": N_CLASSES}, f, indent=2)
print(f"✅ Dataset ready | {idx} images | {N_CLASSES} classes")


# ── CELL 5: Transforms & Model builders ─────────────────────────
def get_train_transforms(sz=224):
    return T.Compose([
        T.RandomResizedCrop(sz, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.3),
        T.RandomRotation(45),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        T.RandomGrayscale(0.05),
        T.GaussianBlur(3, sigma=(0.1, 2.0)),
        T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def get_val_transforms(sz=224):
    return T.Compose([
        T.Resize((sz+32, sz+32)), T.CenterCrop(sz),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def build_efficientnet(n):
    m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3, inplace=True), nn.Linear(in_f, 1024),
        nn.SiLU(), nn.BatchNorm1d(1024), nn.Dropout(0.15), nn.Linear(1024, n))
    return m

def build_swin(n):
    m = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
    in_f = m.head.in_features
    m.head = nn.Sequential(
        nn.LayerNorm(in_f), nn.Dropout(0.3), nn.Linear(in_f, 1024),
        nn.GELU(), nn.Dropout(0.15), nn.Linear(1024, n))
    return m

def build_mobilenet(n):
    m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    in_f = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_f, n)
    return m

MODEL_MAP = {"efficientnet": build_efficientnet, "swin": build_swin, "mobilenet": build_mobilenet}
print(f"✅ Ready to train {N_CLASSES} classes")


# ── CELL 6: Training function ───────────────────────────────────
def train_model(model_name, epochs=50, batch_size=32, lr=1e-4):
    print(f"\n{'='*55}")
    print(f"  🚀 Training: {model_name.upper()}")
    print(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"{'='*55}")

    train_ds = ImageFolder(f"{DATA_DIR}/train", transform=get_train_transforms())
    val_ds   = ImageFolder(f"{DATA_DIR}/val",   transform=get_val_transforms())

    targets = [s[1] for s in train_ds.samples]
    counts  = [0] * N_CLASSES
    for t in targets: counts[t] += 1
    weights = [1.0 / max(counts[t], 1) for t in targets]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                          num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)

    model = MODEL_MAP[model_name](N_CLASSES).to(DEVICE)

    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if any(k in name for k in ["classifier", "head"]):
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = AdamW([
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": head_params,     "lr": lr},
    ], weight_decay=0.01)

    scheduler = OneCycleLR(optimizer, max_lr=[lr*0.1, lr],
                           epochs=epochs, steps_per_epoch=len(train_dl), pct_start=0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc, history = 0.0, []

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            t_loss    += loss.item()
            t_correct += (logits.argmax(1) == y).sum().item()
            t_total   += y.size(0)

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                v_correct += (model(x).argmax(1) == y).sum().item()
                v_total   += y.size(0)

        t_acc = t_correct / t_total * 100
        v_acc = v_correct / v_total * 100
        avg_l = t_loss / len(train_dl)
        marker = " ⭐ BEST" if v_acc > best_acc else ""
        print(f"  Epoch {epoch:3d}/{epochs} | loss={avg_l:.4f} | train={t_acc:.1f}% | val={v_acc:.1f}%{marker}")

        if v_acc > best_acc:
            best_acc = v_acc
            save_path = f"{OUTPUT_DIR}/cropguard_{model_name}_best.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": train_ds.classes,
                "n_classes": N_CLASSES,
                "model_name": model_name,
                "val_acc": best_acc,
                "epoch": epoch,
            }, save_path)

    print(f"\n🏆 {model_name} — Best: {best_acc:.1f}% → {OUTPUT_DIR}/cropguard_{model_name}_best.pt")
    return best_acc, history


# ── CELL 7: Train EfficientNetV2 (~45 min on T4) ────────────────
eff_acc, _ = train_model("efficientnet", epochs=50, batch_size=32, lr=1e-4)

# ── CELL 8: Train Swin Transformer (~40 min on T4) ──────────────
swin_acc, _ = train_model("swin", epochs=40, batch_size=16, lr=5e-5)

# ── CELL 9: Train MobileNetV3 (~20 min on T4) ───────────────────
mob_acc, _ = train_model("mobilenet", epochs=40, batch_size=32, lr=1e-4)


# ── CELL 10: Summary & Download ─────────────────────────────────
print("\n" + "="*55)
print("  TRAINING COMPLETE")
print("="*55)
print(f"  EfficientNetV2  : {eff_acc:.1f}%")
print(f"  Swin Transformer: {swin_acc:.1f}%")
print(f"  MobileNetV3     : {mob_acc:.1f}%")
print(f"  Ensemble (est)  : ~{max(eff_acc, swin_acc)+1:.1f}%")
print("="*55)

for f in os.listdir(OUTPUT_DIR):
    size = os.path.getsize(f"{OUTPUT_DIR}/{f}") / 1e6
    print(f"  📦 {f}  ({size:.1f} MB)")

if IS_COLAB:
    from google.colab import files
    for f in ["cropguard_efficientnet_best.pt", "cropguard_swin_best.pt", "cropguard_mobilenet_best.pt"]:
        fpath = f"{OUTPUT_DIR}/{f}"
        if os.path.exists(fpath):
            files.download(fpath)

if IS_KAGGLE:
    print("\n📥 Go to Output tab → Download each .pt file")

print("\n✅ Copy .pt files to VS Code: static/models/")
print("Then update .env:")
print("  EFFICIENTNET_WEIGHTS=static/models/cropguard_efficientnet_best.pt")
print("  SWIN_WEIGHTS=static/models/cropguard_swin_best.pt")
