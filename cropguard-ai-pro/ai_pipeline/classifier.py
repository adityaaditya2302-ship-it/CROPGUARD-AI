"""
CropGuard AI - EfficientNetV2 Disease Classifier
Stage 2 of the ensemble pipeline: high-accuracy CNN classification
Uses EfficientNetV2-S pretrained on ImageNet, fine-tuned for plant disease
"""
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────
# PlantVillage class index → (crop, disease) mapping
# Add more as you train on additional datasets
# ─────────────────────────────────────────────
EFFICIENTNET_CLASSES = {
    0:  ("apple",      "Apple Scab"),
    1:  ("apple",      "Black Rot"),
    2:  ("apple",      "Cedar Apple Rust"),
    3:  ("apple",      "Healthy"),
    4:  ("blueberry",  "Healthy"),
    5:  ("cherry",     "Powdery Mildew"),
    6:  ("cherry",     "Healthy"),
    7:  ("corn",       "Cercospora Leaf Spot"),
    8:  ("corn",       "Common Rust"),
    9:  ("corn",       "Northern Leaf Blight"),
    10: ("corn",       "Healthy"),
    11: ("grape",      "Black Rot"),
    12: ("grape",      "Esca / Black Measles"),
    13: ("grape",      "Leaf Blight"),
    14: ("grape",      "Healthy"),
    15: ("orange",     "Citrus Greening"),
    16: ("peach",      "Bacterial Spot"),
    17: ("peach",      "Healthy"),
    18: ("pepper",     "Bacterial Spot"),
    19: ("pepper",     "Healthy"),
    20: ("potato",     "Early Blight"),
    21: ("potato",     "Late Blight"),
    22: ("potato",     "Healthy"),
    23: ("raspberry",  "Healthy"),
    24: ("soybean",    "Healthy"),
    25: ("squash",     "Powdery Mildew"),
    26: ("strawberry", "Leaf Scorch"),
    27: ("strawberry", "Healthy"),
    28: ("tomato",     "Bacterial Spot"),
    29: ("tomato",     "Early Blight"),
    30: ("tomato",     "Late Blight"),
    31: ("tomato",     "Leaf Mold"),
    32: ("tomato",     "Septoria Leaf Spot"),
    33: ("tomato",     "Spider Mites"),
    34: ("tomato",     "Target Spot"),
    35: ("tomato",     "Mosaic Virus"),
    36: ("tomato",     "Yellow Leaf Curl Virus"),
    37: ("tomato",     "Healthy"),
    # Extended: Rice
    38: ("rice",       "Brown Spot"),
    39: ("rice",       "Leaf Blast"),
    40: ("rice",       "Neck Blast"),
    41: ("rice",       "Sheath Blight"),
    42: ("rice",       "Healthy"),
    # Extended: Wheat
    43: ("wheat",      "Brown Rust"),
    44: ("wheat",      "Yellow Rust"),
    45: ("wheat",      "Powdery Mildew"),
    46: ("wheat",      "Septoria"),
    47: ("wheat",      "Healthy"),
    # Extended: Mango
    48: ("mango",      "Anthracnose"),
    49: ("mango",      "Powdery Mildew"),
    50: ("mango",      "Bacterial Canker"),
    51: ("mango",      "Healthy"),
    # Extended: Sugarcane
    52: ("sugarcane",  "Red Rot"),
    53: ("sugarcane",  "Smut"),
    54: ("sugarcane",  "Mosaic Virus"),
    55: ("sugarcane",  "Healthy"),
    # Extended: Cotton
    56: ("cotton",     "Bacterial Blight"),
    57: ("cotton",     "Curl Virus"),
    58: ("cotton",     "Fusarium Wilt"),
    59: ("cotton",     "Healthy"),
}

NUM_CLASSES = len(EFFICIENTNET_CLASSES)

# ImageNet normalization (used by all torchvision pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

INFERENCE_TRANSFORMS = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class EfficientNetClassifier:
    """
    EfficientNetV2-S fine-tuned plant disease classifier.

    If a saved checkpoint exists at `weights_path`, it is loaded.
    Otherwise the model runs with frozen ImageNet weights + random head
    (useful for testing the pipeline without training).
    """

    def __init__(self, weights_path: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()
        self.model.eval()

        if weights_path and os.path.exists(weights_path):
            self._load_weights(weights_path)
            print(f"✅ EfficientNetV2 classifier loaded from {weights_path}")
        else:
            print("⚠️  EfficientNetV2: no fine-tuned weights found. "
                  "Running with ImageNet backbone + random head (inference only).")

        self.model.to(self.device)

    # ── private helpers ────────────────────────────────────────────────────────

    def _build_model(self) -> nn.Module:
        """Build EfficientNetV2-S with a custom classification head."""
        base = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, NUM_CLASSES),
        )
        return base

    def _load_weights(self, path: str):
        state = torch.load(path, map_location=self.device)
        # Support both raw state-dict and checkpoint dicts
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state, strict=False)

    def _preprocess(self, image_input) -> torch.Tensor:
        """Accept file path, PIL Image, or numpy array."""
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input).convert("RGB")
        else:
            img = image_input.convert("RGB")
        return INFERENCE_TRANSFORMS(img).unsqueeze(0).to(self.device)

    # ── public API ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, image_input, top_k: int = 3) -> list[dict]:
        """
        Run inference and return top-k predictions.

        Returns a list of dicts, each with:
          crop        – e.g. "tomato"
          disease     – e.g. "Late Blight"
          confidence  – float 0-100
          class_id    – int
        """
        tensor = self._preprocess(image_input)
        logits = self.model(tensor)                       # (1, NUM_CLASSES)
        probs  = torch.softmax(logits, dim=1)[0]          # (NUM_CLASSES,)

        top_k = min(top_k, NUM_CLASSES)
        values, indices = torch.topk(probs, top_k)

        results = []
        for conf, idx in zip(values.cpu().numpy(), indices.cpu().numpy()):
            crop, disease = EFFICIENTNET_CLASSES.get(int(idx), ("unknown", "Unknown"))
            results.append({
                "crop":       crop,
                "disease":    disease,
                "confidence": round(float(conf) * 100, 2),
                "class_id":   int(idx),
                "model":      "EfficientNetV2",
            })
        return results

    def get_feature_maps(self, image_input):
        """Return the last conv feature maps (used by Grad-CAM)."""
        tensor = self._preprocess(image_input)
        # Hook into features.8 (last conv block of EfficientNetV2-S)
        feature_maps = []

        def hook_fn(module, input, output):
            feature_maps.append(output)

        handle = self.model.features[-1].register_forward_hook(hook_fn)
        _ = self.model(tensor)
        handle.remove()
        return feature_maps[0], tensor  # (1, C, H, W)


# ─────────────────────────────────────────────
# Singleton accessor
# ─────────────────────────────────────────────
_instance: EfficientNetClassifier | None = None


def get_efficientnet_classifier(weights_path: str = None) -> EfficientNetClassifier:
    global _instance
    if _instance is None:
        _instance = EfficientNetClassifier(weights_path=weights_path)
    return _instance
