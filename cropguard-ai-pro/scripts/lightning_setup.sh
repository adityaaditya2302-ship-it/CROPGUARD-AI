#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  CropGuard AI — Lightning AI Auto-Setup Script
#  Run this ONCE after cloning the repo in Lightning AI Studio
#
#  Usage (in Lightning AI Studio terminal):
#    chmod +x scripts/lightning_setup.sh
#    ./scripts/lightning_setup.sh
# ═══════════════════════════════════════════════════════════════════

set -e  # Exit on any error

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   CropGuard AI — Lightning AI Setup      ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── 1. Install dependencies ─────────────────────────────────────────
echo "📦 Installing training dependencies..."
pip install --quiet lightning torch torchvision torchaudio
pip install --quiet kaggle albumentations tensorboard
pip install --quiet numpy pillow scikit-learn
pip install --quiet timm  # extra model variants
echo "✅ Dependencies installed"

# ── 2. Verify GPU ───────────────────────────────────────────────────
echo ""
echo "🖥️  Checking GPU..."
python -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'✅ GPU: {gpu} ({mem:.1f} GB VRAM)')
else:
    print('⚠️  No GPU found — training will be slow on CPU')
    print('   In Lightning AI: Studio Settings → Compute → Select GPU')
"

# ── 3. Kaggle setup reminder ────────────────────────────────────────
echo ""
echo "🔑 Kaggle API Setup:"
if [ -f ~/.kaggle/kaggle.json ]; then
    echo "   ✅ kaggle.json already exists"
else
    echo "   ⚠️  kaggle.json NOT found"
    echo "   To set up:"
    echo "   1. Go to kaggle.com → Profile → Settings → API"
    echo "   2. Click 'Create New API Token' → downloads kaggle.json"
    echo "   3. Run:"
    echo "      mkdir -p ~/.kaggle"
    echo "      cp /path/to/kaggle.json ~/.kaggle/"
    echo "      chmod 600 ~/.kaggle/kaggle.json"
fi

# ── 4. Create output directories ────────────────────────────────────
echo ""
mkdir -p datasets/raw datasets/combined static/models lightning_logs evaluation_results
echo "✅ Directories created"

echo ""
echo "══════════════════════════════════════════════"
echo "  Setup complete! Next steps:"
echo ""
echo "  STEP 1 — Download datasets:"
echo "    python scripts/download_datasets.py --output datasets/raw"
echo ""
echo "  STEP 2 — Prepare datasets:"
echo "    python scripts/prepare_dataset.py --input datasets/raw --output datasets/combined"
echo ""
echo "  STEP 3 — Train all models:"
echo "    ./scripts/train_all_models.sh"
echo "══════════════════════════════════════════════"
