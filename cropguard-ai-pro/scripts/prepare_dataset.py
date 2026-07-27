"""
CropGuard AI - Dataset Preparation & Merger
Merges all raw datasets into a unified ImageFolder structure.

Input structure (after download):
  datasets/raw/
    PlantVillage/
      Tomato__Bacterial_spot/  ← original mixed naming
      Tomato__healthy/
      ...
    rice/
      Blast/
      Brown_Spot/
    cassava/
      cmd/
    ...

Output structure (unified):
  datasets/combined/
    train/
      tomato_bacterial_spot/   ← normalized class names
      tomato_healthy/
      rice_blast/
      rice_brown_spot/
      cassava_cmd/
      ...
    val/
      ...
    test/
      ...

  datasets/combined/class_map.json   ← class index → name map
  datasets/combined/dataset_stats.json  ← counts per class

Usage:
  python scripts/prepare_dataset.py \
    --input  datasets/raw \
    --output datasets/combined \
    --val_split 0.15 \
    --test_split 0.05 \
    --min_samples 50     # skip classes with fewer than 50 images
"""
import os
import sys
import json
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict


# ─────────────────────────────────────────────────────────
# CLASS NAME NORMALIZER
# Maps raw folder names → clean unified class names
# ─────────────────────────────────────────────────────────
CLASS_MAP = {
    # ── PlantVillage ─────────────────────────────────────
    "Tomato___Bacterial_spot":              "tomato_bacterial_spot",
    "Tomato___Early_blight":                "tomato_early_blight",
    "Tomato___Late_blight":                 "tomato_late_blight",
    "Tomato___Leaf_Miner":                  "tomato_leaf_miner",
    "Tomato___Leaf_Mold":                   "tomato_leaf_mold",
    "Tomato___Septoria_leaf_spot":          "tomato_septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "tomato_spider_mites",
    "Tomato___Target_Spot":                 "tomato_target_spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato_yellow_leaf_curl",
    "Tomato___Tomato_mosaic_virus":         "tomato_mosaic_virus",
    "Tomato___healthy":                     "tomato_healthy",
    "Potato___Early_Blight":                "potato_early_blight",
    "Potato___Late_Blight":                 "potato_late_blight",
    "Potato___healthy":                     "potato_healthy",
    "Pepper,_bell___Bacterial_spot":        "pepper_bacterial_spot",
    "Pepper,_bell___healthy":              "pepper_healthy",
    "Apple___Apple_scab":                   "apple_scab",
    "Apple___Black_rot":                    "apple_black_rot",
    "Apple___Cedar_apple_rust":             "apple_cedar_rust",
    "Apple___healthy":                      "apple_healthy",
    "Grape___Black_rot":                    "grape_black_rot",
    "Grape___Esca_(Black_Measles)":         "grape_esca",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "grape_leaf_blight",
    "Grape___healthy":                      "grape_healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "maize_gray_leaf_spot",
    "Corn_(maize)___Common_rust_":          "maize_common_rust",
    "Corn_(maize)___Northern_Leaf_Blight":  "maize_northern_blight",
    "Corn_(maize)___healthy":               "maize_healthy",
    "Rice___Leaf_Blast":                    "rice_leaf_blast",
    "Rice___Brown_Spot":                    "rice_brown_spot",
    "Rice___Hispa":                         "rice_hispa",
    "Wheat___Leaf_rust":                    "wheat_leaf_rust",
    "Wheat___Crown_and_Root_Rot":           "wheat_root_rot",
    "Wheat___Healthy":                      "wheat_healthy",
    "Strawberry___Leaf_scorch":             "strawberry_leaf_scorch",
    "Strawberry___healthy":                 "strawberry_healthy",
    "Peach___Bacterial_spot":               "peach_bacterial_spot",
    "Peach___healthy":                      "peach_healthy",
    "Cherry_(including_sour)___Powdery_mildew": "cherry_powdery_mildew",
    "Cherry_(including_sour)___healthy":    "cherry_healthy",

    # ── Rice ─────────────────────────────────────────────
    "Bacterial leaf blight":                "rice_bacterial_blight",
    "Brown Spot":                           "rice_brown_spot",
    "Leaf smut":                            "rice_leaf_smut",
    "blast":                                "rice_leaf_blast",
    "blight":                               "rice_bacterial_blight",
    "tungro":                               "rice_tungro",

    # ── Cassava ──────────────────────────────────────────
    "Cassava Bacterial Blight (CBB)":       "cassava_bacterial_blight",
    "Cassava Brown Streak Disease (CBSD)":  "cassava_brown_streak",
    "Cassava Green Mottle (CGM)":           "cassava_green_mottle",
    "Cassava Mosaic Disease (CMD)":         "cassava_mosaic",
    "Healthy":                              "cassava_healthy",
    "cmd":                                  "cassava_mosaic",
    "cbb":                                  "cassava_bacterial_blight",
    "cbsd":                                 "cassava_brown_streak",
    "cgm":                                  "cassava_green_mottle",
    "healthy":                              "cassava_healthy",

    # ── Maize ────────────────────────────────────────────
    "Blight":                               "maize_northern_blight",
    "Common_Rust":                          "maize_common_rust",
    "Gray_Leaf_Spot":                       "maize_gray_leaf_spot",

    # ── Wheat ────────────────────────────────────────────
    "Loose Smut":                           "wheat_loose_smut",
    "Crown and Root Rot":                   "wheat_root_rot",
    "Powdery Mildew":                       "wheat_powdery_mildew",
    "Septoria":                             "wheat_septoria",
    "Strip Rust":                           "wheat_stripe_rust",
    "Yellow Rust":                          "wheat_stripe_rust",
    "Brown Rust":                           "wheat_leaf_rust",
    "Black Rust":                           "wheat_stem_rust",

    # ── Mango ────────────────────────────────────────────
    "Anthracnose":                          "mango_anthracnose",
    "Bacterial Canker":                     "mango_bacterial_canker",
    "Cutting Weevil":                       "mango_cutting_weevil",
    "Die Back":                             "mango_die_back",
    "Gall Midge":                           "mango_gall_midge",
    "Powdery Mildew":                       "mango_powdery_mildew",
    "Sooty Mould":                          "mango_sooty_mould",

    # ── Sugarcane ────────────────────────────────────────
    "Red Rot":                              "sugarcane_red_rot",
    "Smut":                                 "sugarcane_smut",
    "Mosaic":                               "sugarcane_mosaic",
    "Yellow Leaf":                          "sugarcane_yellow_leaf",
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


class DatasetPreparer:
    def __init__(self, input_dir: str, output_dir: str,
                 val_split: float = 0.15, test_split: float = 0.05,
                 min_samples: int = 50, max_samples_per_class: int = 3000,
                 seed: int = 42):
        self.input_dir  = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.val_split  = val_split
        self.test_split = test_split
        self.min_samples = min_samples
        self.max_samples = max_samples_per_class
        random.seed(seed)

        self.all_images  = defaultdict(list)  # class_name -> [image_paths]
        self.skipped     = []
        self.stats       = {}

    def normalize_class_name(self, raw_name: str) -> str:
        """Convert raw folder name to unified class name."""
        # Direct lookup
        if raw_name in CLASS_MAP:
            return CLASS_MAP[raw_name]

        # Fuzzy: try case-insensitive
        for k, v in CLASS_MAP.items():
            if k.lower() == raw_name.lower():
                return v

        # Auto-normalize: replace spaces/special chars
        normalized = raw_name.lower()
        normalized = normalized.replace(" ", "_").replace("-", "_")
        normalized = normalized.replace("(", "").replace(")", "").replace(",", "")
        normalized = "_".join([w for w in normalized.split("_") if w])
        return normalized

    def scan_all_datasets(self):
        """Walk all raw dataset directories and collect images by class."""
        print(f"\n📂 Scanning datasets in: {self.input_dir}")

        if not self.input_dir.exists():
            print(f"❌ Input directory not found: {self.input_dir}")
            print("   Run download_datasets.py first, or create datasets/raw/ manually.")
            return

        for dataset_dir in sorted(self.input_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            print(f"\n  📁 {dataset_dir.name}:")
            self._scan_dataset(dataset_dir)

        total_images = sum(len(v) for v in self.all_images.values())
        print(f"\n✅ Found {total_images} images across {len(self.all_images)} classes")

    def _scan_dataset(self, dataset_dir: Path):
        """Scan a single dataset directory (handles nested structures)."""
        # Walk up to 4 levels deep looking for image directories
        for root, dirs, files in os.walk(dataset_dir):
            root_path = Path(root)
            images = [f for f in files if Path(f).suffix.lower() in VALID_EXTENSIONS]
            if not images:
                continue

            # The class name is the parent folder name
            class_raw = root_path.name
            class_name = self.normalize_class_name(class_raw)

            count_before = len(self.all_images[class_name])
            for img in images:
                self.all_images[class_name].append(root_path / img)

            added = len(self.all_images[class_name]) - count_before
            if added > 0:
                print(f"    → {class_name}: +{added} images")

    def filter_and_balance(self):
        """Remove classes with too few samples, cap classes with too many."""
        print(f"\n⚖️  Filtering classes (min={self.min_samples}, max={self.max_samples})...")
        filtered = {}
        for cls, imgs in self.all_images.items():
            if len(imgs) < self.min_samples:
                self.skipped.append((cls, len(imgs)))
                print(f"  ⏭️  Skipping '{cls}': only {len(imgs)} images")
                continue

            # Cap at max_samples (random sample)
            if len(imgs) > self.max_samples:
                imgs = random.sample(imgs, self.max_samples)
                print(f"  ✂️  Capped '{cls}' to {self.max_samples} images")

            filtered[cls] = imgs

        self.all_images = filtered
        print(f"\n✅ Final: {len(self.all_images)} classes, "
              f"{sum(len(v) for v in self.all_images.values())} images")

    def split_and_copy(self):
        """Split into train/val/test and copy images."""
        print(f"\n📋 Splitting and copying images...")
        print(f"   Train: {int((1-self.val_split-self.test_split)*100)}% | "
              f"Val: {int(self.val_split*100)}% | "
              f"Test: {int(self.test_split*100)}%")

        splits = ("train", "val", "test")
        for split in splits:
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)

        class_list = sorted(self.all_images.keys())
        class_to_idx = {c: i for i, c in enumerate(class_list)}

        total_copied = 0
        for cls, imgs in self.all_images.items():
            random.shuffle(imgs)
            n = len(imgs)
            n_test  = max(1, int(n * self.test_split))
            n_val   = max(1, int(n * self.val_split))
            n_train = n - n_val - n_test

            split_imgs = {
                "train": imgs[:n_train],
                "val":   imgs[n_train:n_train+n_val],
                "test":  imgs[n_train+n_val:],
            }

            self.stats[cls] = {
                "idx":   class_to_idx[cls],
                "train": n_train,
                "val":   n_val,
                "test":  n_test,
                "total": n,
            }

            for split, split_list in split_imgs.items():
                dest_dir = self.output_dir / split / cls
                dest_dir.mkdir(parents=True, exist_ok=True)
                for src in split_list:
                    ext  = src.suffix.lower()
                    name = f"{total_copied:07d}{ext}"
                    shutil.copy2(src, dest_dir / name)
                    total_copied += 1

        print(f"✅ Copied {total_copied} images to {self.output_dir}")
        return class_to_idx

    def save_metadata(self, class_to_idx: dict):
        """Save class map and dataset statistics."""
        # class_map.json
        class_map_path = self.output_dir / "class_map.json"
        with open(class_map_path, "w") as f:
            json.dump({"idx_to_class": {v: k for k, v in class_to_idx.items()},
                       "class_to_idx": class_to_idx,
                       "num_classes":  len(class_to_idx)}, f, indent=2)

        # dataset_stats.json
        stats_path = self.output_dir / "dataset_stats.json"
        total = sum(v["total"] for v in self.stats.values())
        with open(stats_path, "w") as f:
            json.dump({
                "total_images":   total,
                "num_classes":    len(self.stats),
                "skipped_classes": self.skipped,
                "val_split":      self.val_split,
                "test_split":     self.test_split,
                "per_class":      dict(sorted(self.stats.items())),
            }, f, indent=2)

        print(f"\n📄 class_map.json → {class_map_path}")
        print(f"📄 dataset_stats.json → {stats_path}")
        print(f"\n🏷️  Classes ({len(class_to_idx)}):")
        for cls, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
            s = self.stats[cls]
            print(f"  [{idx:3d}] {cls:<45} train={s['train']:4d} val={s['val']:3d} test={s['test']:3d}")

    def run(self):
        self.scan_all_datasets()
        if not self.all_images:
            print("\n❌ No images found! Check your datasets/raw/ directory.")
            return
        self.filter_and_balance()
        class_to_idx = self.split_and_copy()
        self.save_metadata(class_to_idx)
        print(f"\n✅ Dataset ready at: {self.output_dir.absolute()}")
        print("\nNext step:")
        print("  python scripts/train_lightning.py --data datasets/combined --epochs 50")


def main():
    parser = argparse.ArgumentParser(description="Prepare merged CropGuard AI dataset")
    parser.add_argument("--input",       default="datasets/raw",      help="Raw datasets dir")
    parser.add_argument("--output",      default="datasets/combined",  help="Output dir")
    parser.add_argument("--val_split",   type=float, default=0.15)
    parser.add_argument("--test_split",  type=float, default=0.05)
    parser.add_argument("--min_samples", type=int,   default=50,   help="Minimum images per class")
    parser.add_argument("--max_samples", type=int,   default=3000, help="Max images per class (balance)")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    preparer = DatasetPreparer(
        input_dir=args.input,
        output_dir=args.output,
        val_split=args.val_split,
        test_split=args.test_split,
        min_samples=args.min_samples,
        max_samples_per_class=args.max_samples,
        seed=args.seed,
    )
    preparer.run()


if __name__ == "__main__":
    main()
