"""
CropGuard AI - Swin Transformer Classifier
Stage 3 of the ensemble pipeline: texture-aware transformer-based classification.
Uses Microsoft Swin-T pretrained on ImageNet — excellent at leaf texture patterns.
"""
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import numpy as np

# Re-use the same class mapping as EfficientNet
from .classifier import EFFICIENTNET_CLASSES, NUM_CLASSES, IMAGENET_MEAN, IMAGENET_STD

SWIN_TRANSFORMS = T.Compose([
    T.Resize((232, 232)),          # Swin-T preferred input slightly larger
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class SwinTransformerClassifier:
    """
    Swin-T based plant disease classifier.
    Complements EfficientNetV2 by capturing long-range texture dependencies.
    """

    def __init__(self, weights_path: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = self._build_model()
        self.model.eval()

        if weights_path and os.path.exists(weights_path):
            self._load_weights(weights_path)
            print(f"✅ Swin Transformer loaded from {weights_path}")
        else:
            print("⚠️  Swin Transformer: no fine-tuned weights found. "
                  "Running with ImageNet backbone + random head.")

        self.model.to(self.device)

    def _build_model(self) -> nn.Module:
        base = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        in_features = base.head.in_features
        base.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, NUM_CLASSES),
        )
        return base

    def _load_weights(self, path: str):
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state, strict=False)

    def _preprocess(self, image_input) -> torch.Tensor:
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input).convert("RGB")
        else:
            img = image_input.convert("RGB")
        return SWIN_TRANSFORMS(img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image_input, top_k: int = 3) -> list[dict]:
        """Same interface as EfficientNetClassifier.predict()"""
        tensor = self._preprocess(image_input)
        logits = self.model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

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
                "model":      "SwinTransformer",
            })
        return results


# Singleton
_swin_instance: SwinTransformerClassifier | None = None

def get_swin_classifier(weights_path: str = None) -> SwinTransformerClassifier:
    global _swin_instance
    if _swin_instance is None:
        _swin_instance = SwinTransformerClassifier(weights_path=weights_path)
    return _swin_instance
