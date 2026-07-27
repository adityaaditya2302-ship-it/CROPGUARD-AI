"""
CropGuard AI - Grad-CAM Explainable AI Module
Generates visual heatmaps showing WHICH part of the leaf the AI focused on.
Supports EfficientNetV2 + Swin Transformer target layers.

Output: Base64-encoded JPEG heatmap overlay on original image.
"""
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import base64
from io import BytesIO


class GradCAMGenerator:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Works with any CNN or hybrid model by:
      1. Registering forward + backward hooks on the target conv layer
      2. Computing gradient of the target class score w.r.t. feature maps
      3. Weighting feature maps by global-average-pooled gradients
      4. Upsampling result to input resolution and overlaying on image
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module, device: str = "cpu"):
        self.model = model
        self.target_layer = target_layer
        self.device = device

        self._feature_maps = None
        self._gradients = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_features)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    # ── hooks ──────────────────────────────────────────────────────────────────

    def _save_features(self, module, input, output):
        self._feature_maps = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def remove_hooks(self):
        """Call this when done to avoid memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    # ── core generation ────────────────────────────────────────────────────────

    def generate(self, image_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap.

        Args:
            image_tensor: (1, 3, H, W) preprocessed tensor
            class_idx:    target class for which to compute CAM.
                          If None, uses the predicted class.
        Returns:
            heatmap as (H, W) float32 array in [0, 1]
        """
        self.model.eval()
        image_tensor = image_tensor.to(self.device).requires_grad_(True)

        # Forward pass
        logits = self.model(image_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Global average pool of gradients: shape (C,)
        weights = self._gradients.mean(dim=[2, 3], keepdim=True)

        # Weighted combination of feature maps: shape (H', W')
        cam = (weights * self._feature_maps).sum(dim=1).squeeze()
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam.cpu().numpy()

    # ── overlay utility ────────────────────────────────────────────────────────

    @staticmethod
    def overlay_on_image(original_image_path: str, heatmap: np.ndarray,
                         alpha: float = 0.45, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Overlay heatmap on original image.

        Args:
            original_image_path: path to original (or PIL Image / numpy BGR)
            heatmap:             (H, W) float32 [0,1]
            alpha:               blend factor for heatmap
            colormap:            cv2 colormap constant

        Returns:
            BGR numpy array of the blended overlay
        """
        if isinstance(original_image_path, str):
            orig = cv2.imread(original_image_path)
        elif isinstance(original_image_path, np.ndarray):
            orig = original_image_path
        else:
            orig = cv2.cvtColor(np.array(original_image_path), cv2.COLOR_RGB2BGR)

        h, w = orig.shape[:2]

        # Resize heatmap to match image dimensions
        heat_resized = cv2.resize(heatmap, (w, h))
        heat_uint8   = np.uint8(255 * heat_resized)
        heat_colored = cv2.applyColorMap(heat_uint8, colormap)

        # Blend
        overlay = cv2.addWeighted(orig, 1 - alpha, heat_colored, alpha, 0)
        return overlay

    @staticmethod
    def to_base64_jpeg(image: np.ndarray, quality: int = 85) -> str:
        """Convert BGR numpy array to base64-encoded JPEG string."""
        _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf).decode("utf-8")

    # ── convenience end-to-end ─────────────────────────────────────────────────

    def generate_overlay_b64(self, image_path: str, image_tensor: torch.Tensor,
                              class_idx: int = None) -> dict:
        """
        Full pipeline: generate CAM → overlay → return base64 JPEG + metadata.

        Returns dict with keys:
          heatmap_b64   – base64 JPEG of overlay
          class_idx     – predicted or provided class index
          cam_max       – maximum CAM activation (proxy for focus strength)
          highlighted_regions – approx. percentage of image highlighted
        """
        heatmap = self.generate(image_tensor, class_idx)

        overlay_bgr = self.overlay_on_image(image_path, heatmap)
        b64 = self.to_base64_jpeg(overlay_bgr)

        # Stats
        threshold = 0.6
        highlighted_pct = float(np.mean(heatmap > threshold) * 100)

        return {
            "heatmap_b64": b64,
            "class_idx": class_idx if class_idx is not None else int(np.argmax(heatmap)),
            "cam_max": float(heatmap.max()),
            "highlighted_regions_pct": round(highlighted_pct, 1),
        }


# ─────────────────────────────────────────────
# Standalone Grad-CAM for EfficientNetV2
# (without needing a full GradCAMGenerator instance)
# ─────────────────────────────────────────────

def generate_gradcam_for_efficientnet(classifier_instance, image_path: str,
                                       class_idx: int = None) -> dict:
    """
    Convenience function: run Grad-CAM on an EfficientNetClassifier instance.

    Returns the same dict as GradCAMGenerator.generate_overlay_b64()
    """
    from .classifier import INFERENCE_TRANSFORMS

    # Preprocess
    img = Image.open(image_path).convert("RGB")
    tensor = INFERENCE_TRANSFORMS(img).unsqueeze(0).to(classifier_instance.device)

    # Target the last conv block
    target_layer = classifier_instance.model.features[-1]
    cam_gen = GradCAMGenerator(
        model=classifier_instance.model,
        target_layer=target_layer,
        device=classifier_instance.device,
    )
    result = cam_gen.generate_overlay_b64(image_path, tensor, class_idx)
    cam_gen.remove_hooks()
    return result
