"""
CropGuard AI - Dataset Downloader
Downloads all training datasets from Kaggle and Roboflow.

Datasets:
  1. PlantVillage       - 38 classes, ~54k images (Kaggle)
  2. Rice Leaf Disease  - 3 classes,  ~3k  images (Kaggle)
  3. Cassava Leaf       - 5 classes,  ~21k images (Kaggle)
  4. Maize Disease      - 4 classes,  ~3k  images (Kaggle)
  5. Wheat Disease      - 5 classes,  ~5k  images (Kaggle)
  6. Mango Disease      - 3 classes,  ~3k  images (Kaggle)
  7. Sugarcane Disease  - 4 classes,  ~2k  images (Kaggle)
  8. Indian Agri        - local crops, ~20k images (Roboflow)

Usage:
  # Install Kaggle API first:
  pip install kaggle roboflow

  # Place kaggle.json in ~/.kaggle/
  # Get it from: kaggle.com → Profile → Settings → API

  python scripts/download_datasets.py --output datasets/raw
"""
import os
import sys
import json
import shutil
import argparse
import zipfile
from pathlib import Path


# ─────────────────────────────────────────────────────────
# DATASET REGISTRY
# ─────────────────────────────────────────────────────────
KAGGLE_DATASETS = [
    {
        "name":    "PlantVillage",
        "slug":    "emmarex/plantdisease",
        "folder":  "PlantVillage",
        "classes": 38,
        "size":    "54k",
        "notes":   "Main backbone dataset — 38 crop-disease classes",
    },
    {
        "name":    "Rice Leaf Disease",
        "slug":    "vbookshelf/rice-leaf-diseases",
        "folder":  "rice",
        "classes": 3,
        "size":    "3k",
        "notes":   "Blast, Brown Spot, Sheath Blight",
    },
    {
        "name":    "Cassava Leaf Disease",
        "slug":    "ipythonx/cassava-leaf-disease",
        "folder":  "cassava",
        "classes": 5,
        "size":    "21k",
        "notes":   "CMD, CBB, CGM, CBSD, Healthy",
    },
    {
        "name":    "Maize (Corn) Disease",
        "slug":    "smaranjitghose/corn-or-maize-leaf-disease-dataset",
        "folder":  "maize",
        "classes": 4,
        "size":    "3k",
        "notes":   "Northern Blight, Gray Leaf Spot, Common Rust, Healthy",
    },
    {
        "name":    "Wheat Disease",
        "slug":    "kushagra0301/wheat-disease-dataset",
        "folder":  "wheat",
        "classes": 5,
        "size":    "5k",
        "notes":   "Rust, Smut, Powdery Mildew, Leaf Blight, Healthy",
    },
    {
        "name":    "Mango Leaf Disease",
        "slug":    "warcoder/mango-leaf-disease-dataset",
        "folder":  "mango",
        "classes": 8,
        "size":    "4k",
        "notes":   "Anthracnose, Powdery Mildew, Bacterial Canker, etc.",
    },
    {
        "name":    "Sugarcane Disease",
        "slug":    "nirmalsankalana/sugarcane-disease-detection-dataset",
        "folder":  "sugarcane",
        "classes": 4,
        "size":    "2k",
        "notes":   "Red Rot, Smut, Mosaic, Healthy",
    },
    {
        "name":    "Grape Disease",
        "slug":    "rm1000/grape-disease-dataset",
        "folder":  "grape",
        "classes": 4,
        "size":    "3k",
        "notes":   "Black Rot, Leaf Blight, Esca, Healthy",
    },
    {
        "name":    "Tomato Disease (Extra)",
        "slug":    "kaustubhb11/plantdisease",
        "folder":  "tomato_extra",
        "classes": 10,
        "size":    "16k",
        "notes":   "Additional tomato disease data with Indian field images",
    },
]


def download_kaggle_dataset(slug: str, output_dir: Path, name: str) -> bool:
    """Download a Kaggle dataset."""
    try:
        import kaggle
        print(f"\n📥 Downloading: {name} ({slug})")
        kaggle.api.dataset_download_files(slug, path=str(output_dir), unzip=True)
        print(f"   ✅ Done → {output_dir}")
        return True
    except ImportError:
        print("❌ kaggle package not installed. Run: pip install kaggle")
        return False
    except Exception as e:
        print(f"   ⚠️  Failed to download {name}: {e}")
        print(f"   ℹ️  Manual download: https://www.kaggle.com/datasets/{slug}")
        return False


def setup_kaggle_credentials():
    """Check and setup Kaggle API credentials."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("⚠️  Kaggle credentials not found!")
        print("\nTo set up Kaggle API:")
        print("  1. Go to https://www.kaggle.com → Profile → Settings → API")
        print("  2. Click 'Create New API Token'")
        print("  3. It downloads kaggle.json")
        print(f"  4. Move it to: {kaggle_json}")
        print("\nOr set environment variables:")
        print("  KAGGLE_USERNAME=your_username")
        print("  KAGGLE_KEY=your_api_key")

        # Try env vars
        if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
            print("\n✅ Found KAGGLE_USERNAME and KAGGLE_KEY env vars")
            return True
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Download CropGuard AI training datasets")
    parser.add_argument("--output", default="datasets/raw", help="Output directory")
    parser.add_argument("--dataset", default="all", help="Dataset name or 'all'")
    parser.add_argument("--list", action="store_true", help="List all available datasets")
    args = parser.parse_args()

    if args.list:
        print("\n📚 Available Datasets:")
        print(f"{'Name':<30} {'Slug':<45} {'Classes':<10} {'Size'}")
        print("-" * 100)
        for d in KAGGLE_DATASETS:
            print(f"{d['name']:<30} {d['slug']:<45} {d['classes']:<10} {d['size']}")
        return

    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  CropGuard AI Dataset Downloader")
    print("=" * 60)
    print(f"  Output directory: {output_base.absolute()}")
    print(f"  Total datasets: {len(KAGGLE_DATASETS)}")

    if not setup_kaggle_credentials():
        print("\n⚠️  Cannot download without Kaggle credentials.")
        print("    Creating manual download guide instead...\n")
        create_manual_guide(output_base)
        return

    datasets = KAGGLE_DATASETS
    if args.dataset != "all":
        datasets = [d for d in KAGGLE_DATASETS if d["name"].lower() == args.dataset.lower()]
        if not datasets:
            print(f"❌ Dataset '{args.dataset}' not found. Use --list to see all.")
            return

    success, failed = [], []
    for ds in datasets:
        dst = output_base / ds["folder"]
        dst.mkdir(parents=True, exist_ok=True)
        ok = download_kaggle_dataset(ds["slug"], dst, ds["name"])
        (success if ok else failed).append(ds["name"])

    print("\n" + "=" * 60)
    print(f"✅ Downloaded: {len(success)}/{len(datasets)} datasets")
    if failed:
        print(f"⚠️  Failed: {', '.join(failed)}")
        print("\nFailed datasets — download manually from Kaggle:")
        for ds in datasets:
            if ds["name"] in failed:
                print(f"  https://www.kaggle.com/datasets/{ds['slug']}")
                print(f"  → Extract to: datasets/raw/{ds['folder']}/")

    print(f"\n📁 Datasets saved to: {output_base.absolute()}")
    print("\n✅ Next step: python scripts/prepare_dataset.py --input datasets/raw --output datasets/combined")


def create_manual_guide(output_dir: Path):
    """Create a manual download guide if Kaggle auth fails."""
    guide = output_dir / "DOWNLOAD_GUIDE.md"
    content = "# Manual Dataset Download Guide\n\n"
    for ds in KAGGLE_DATASETS:
        content += f"## {ds['name']}\n"
        content += f"- **Classes:** {ds['classes']} | **Size:** {ds['size']}\n"
        content += f"- **Download:** https://www.kaggle.com/datasets/{ds['slug']}\n"
        content += f"- **Save to:** `datasets/raw/{ds['folder']}/`\n"
        content += f"- **Notes:** {ds['notes']}\n\n"
    guide.write_text(content)
    print(f"📄 Manual guide created: {guide}")
    print("\nDownload each dataset and place in:")
    for ds in KAGGLE_DATASETS:
        print(f"  datasets/raw/{ds['folder']}/")


if __name__ == "__main__":
    main()
