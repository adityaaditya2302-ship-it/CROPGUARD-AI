"""
CropGuard AI - Ensemble Pipeline
Combines YOLOv8 + EfficientNetV2 + Swin Transformer into one prediction.

Pipeline stages:
  1. YOLOv8  → detect & localize diseased regions (bounding boxes)
  2. EfficientNetV2 → classify each cropped region (CNN accuracy)
  3. Swin Transformer → re-classify for texture confidence boost
  4. Weighted ensemble → merge all scores
  5. Grad-CAM → generate visual explanation heatmap
  6. Disease database → enrich with severity, treatment, yield loss
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional


# ── DISEASE DATABASE ENRICHMENT ────────────────────────────────────────────────
# Severity → numeric score mapping
SEVERITY_SCORES = {
    "Healthy":    0,
    "Low":        25,
    "Medium":     55,
    "High":       80,
    "Critical":   95,
    "Unknown":    50,
}

# Estimated yield loss % by severity
YIELD_LOSS_BY_SEVERITY = {
    "Healthy":    0,
    "Low":        5,
    "Medium":     15,
    "High":       30,
    "Critical":   60,
    "Unknown":    20,
}

# Days to recovery estimates
RECOVERY_DAYS_BY_SEVERITY = {
    "Healthy":    0,
    "Low":        7,
    "Medium":     14,
    "High":       21,
    "Critical":   35,
    "Unknown":    14,
}

# Treatment urgency by severity
URGENCY_BY_SEVERITY = {
    "Healthy":    "None",
    "Low":        "Within 1 week",
    "Medium":     "Within 3 days",
    "High":       "Immediately",
    "Critical":   "URGENT – Within 24 hours",
    "Unknown":    "As soon as possible",
}


class EnsemblePipeline:
    """
    Master orchestrator for the CropGuard AI ensemble.

    Usage:
        pipeline = EnsemblePipeline()
        result   = pipeline.analyze(image_path)
    """

    def __init__(
        self,
        yolo_model_path:      Optional[str] = None,
        efficientnet_weights: Optional[str] = None,
        swin_weights:         Optional[str] = None,
        enable_gradcam:       bool = True,
        efficientnet_weight:  float = 0.50,   # ensemble weight for EfficientNet
        swin_weight:          float = 0.35,   # ensemble weight for Swin
        yolo_weight:          float = 0.15,   # ensemble weight for YOLO cls
    ):
        self.enable_gradcam       = enable_gradcam
        self.eff_w                = efficientnet_weight
        self.swin_w               = swin_weight
        self.yolo_w               = yolo_weight

        # ── load sub-models ────────────────────────────────────────────────────
        print("🤖 Loading CropGuard AI Ensemble Pipeline...")

        # YOLO (existing detector)
        self.yolo = None
        try:
            from yolo_detector import get_detector
            self.yolo = get_detector(model_path=yolo_model_path)
            print("  ✅ YOLOv8 localization model ready")
        except Exception as e:
            print(f"  ⚠️  YOLOv8 not available: {e}")

        # EfficientNetV2
        self.efficientnet = None
        try:
            from ai_pipeline.classifier import get_efficientnet_classifier
            self.efficientnet = get_efficientnet_classifier(weights_path=efficientnet_weights)
            print("  ✅ EfficientNetV2 classifier ready")
        except Exception as e:
            print(f"  ⚠️  EfficientNetV2 not available: {e}")

        # Swin Transformer
        self.swin = None
        try:
            from ai_pipeline.swin_transformer import get_swin_classifier
            self.swin = get_swin_classifier(weights_path=swin_weights)
            print("  ✅ Swin Transformer classifier ready")
        except Exception as e:
            print(f"  ⚠️  Swin Transformer not available: {e}")

        # Grad-CAM
        self.gradcam = None
        if enable_gradcam and self.efficientnet is not None:
            try:
                from ai_pipeline.gradcam import GradCAMGenerator
                target_layer = self.efficientnet.model.features[-1]
                self.gradcam = GradCAMGenerator(
                    model=self.efficientnet.model,
                    target_layer=target_layer,
                    device=self.efficientnet.device,
                )
                print("  ✅ Grad-CAM XAI module ready")
            except Exception as e:
                print(f"  ⚠️  Grad-CAM not available: {e}")

        print("🚀 Ensemble pipeline ready!\n")

    # ── internal helpers ───────────────────────────────────────────────────────

    def _get_disease_info(self, crop_key: str, disease_name: str) -> dict:
        """Lookup enriched disease info from disease_database."""
        try:
            from disease_database import CROP_DISEASE_DB
            crop_data = CROP_DISEASE_DB.get(crop_key.lower(), {})
            diseases  = crop_data.get("diseases", {})

            # Exact match
            if disease_name in diseases:
                return diseases[disease_name]
            # Fuzzy match
            for name, info in diseases.items():
                if disease_name.lower() in name.lower() or name.lower() in disease_name.lower():
                    return info
        except ImportError:
            pass

        return {
            "severity":    "Unknown",
            "description": f"Detected {disease_name}. Consult local agricultural extension.",
            "symptoms":    ["Visual symptoms detected by AI"],
            "treatments":  {
                "chemical":   ["Consult local agricultural extension"],
                "organic":    ["Neem oil spray", "Compost tea"],
                "prevention": ["Crop rotation", "Field sanitation"],
            },
        }

    def _merge_predictions(self, eff_preds: list, swin_preds: list, yolo_preds: list) -> list:
        """
        Weighted merge of all model predictions into a ranked list.
        Uses a score dictionary keyed by (crop, disease).
        """
        scores = {}

        def add(preds, weight):
            total_conf = sum(p["confidence"] for p in preds) or 1.0
            for p in preds:
                key = (p["crop"], p["disease"])
                norm_conf = (p["confidence"] / 100.0) * weight
                scores[key] = scores.get(key, 0.0) + norm_conf

        add(eff_preds  or [], self.eff_w)
        add(swin_preds or [], self.swin_w)
        add(yolo_preds or [], self.yolo_w)

        # Sort by total weighted score
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for (crop, disease), score in sorted_items[:5]:
            results.append({
                "crop":       crop,
                "disease":    disease,
                "confidence": round(min(score * 100, 99.9), 2),
            })
        return results

    # ── main analysis method ───────────────────────────────────────────────────

    def analyze(self, image_path: str, crop_hint: str = "auto") -> dict:
        """
        Full ensemble analysis of a crop image.

        Args:
            image_path: path to image file
            crop_hint:  optional crop type hint ('tomato', 'rice', etc.)

        Returns:
            Comprehensive Farm Doctor report dict
        """
        if not os.path.exists(image_path):
            return {"success": False, "error": f"Image not found: {image_path}"}

        eff_preds   = []
        swin_preds  = []
        yolo_preds  = []
        gradcam_b64 = None

        # ── Stage 1: YOLOv8 Localization ──────────────────────────────────────
        yolo_result = {}
        if self.yolo:
            yolo_result = self.yolo.detect(image_path, crop_group=crop_hint)
            raw_detections = yolo_result.get("detections", [])
            for d in raw_detections[:3]:
                yolo_preds.append({
                    "crop":       d.get("crop", "unknown"),
                    "disease":    d.get("disease", "Unknown"),
                    "confidence": d.get("confidence", 0),
                })

        # ── Stage 2: EfficientNetV2 Classification ────────────────────────────
        if self.efficientnet:
            eff_preds = self.efficientnet.predict(image_path, top_k=3)

        # ── Stage 3: Swin Transformer Classification ──────────────────────────
        if self.swin:
            swin_preds = self.swin.predict(image_path, top_k=3)

        # ── Stage 4: Weighted Ensemble ────────────────────────────────────────
        merged = self._merge_predictions(eff_preds, swin_preds, yolo_preds)

        # Fallback to YOLO raw detections if ensemble produced nothing
        if not merged and yolo_result.get("detections"):
            top_d = yolo_result["detections"][0]
            merged = [{
                "crop":       top_d.get("crop", "unknown"),
                "disease":    top_d.get("disease", "Unknown"),
                "confidence": top_d.get("confidence", 0),
            }]

        # ── Stage 5: Grad-CAM Explanation ─────────────────────────────────────
        if self.gradcam and self.efficientnet and eff_preds:
            try:
                from ai_pipeline.classifier import INFERENCE_TRANSFORMS
                from PIL import Image
                import torch
                img     = Image.open(image_path).convert("RGB")
                tensor  = INFERENCE_TRANSFORMS(img).unsqueeze(0).to(self.efficientnet.device)
                cam_res = self.gradcam.generate_overlay_b64(
                    image_path, tensor,
                    class_idx=eff_preds[0]["class_id"] if eff_preds else None,
                )
                gradcam_b64 = cam_res.get("heatmap_b64")
            except Exception as e:
                print(f"⚠️  Grad-CAM failed: {e}")

        # ── Stage 6: Enrich with Disease Database ─────────────────────────────
        if merged:
            top = merged[0]
            crop_key     = top["crop"]
            disease_name = top["disease"]
            confidence   = top["confidence"]
        else:
            crop_key     = "unknown"
            disease_name = "Unable to determine"
            confidence   = 0.0

        disease_info = self._get_disease_info(crop_key, disease_name)
        severity     = disease_info.get("severity", "Unknown")

        # ── Compute spread probability from severity + confidence ──────────────
        sev_score  = SEVERITY_SCORES.get(severity, 50)
        spread_pct = min(int(sev_score * (confidence / 100)), 95)
        if disease_name.lower() in ("healthy",):
            spread_pct = 0

        # ── Final structured Farm Doctor Report ───────────────────────────────
        report = {
            "success": True,

            # ── Core prediction ──────────────────────────────────────────────
            "crop": {
                "key":  crop_key,
                "name": crop_key.title(),
            },
            "disease": {
                "name":         disease_name,
                "confidence":   confidence,
                "severity":     severity,
                "severity_score": sev_score,
                "description":  disease_info.get("description", ""),
                "symptoms":     disease_info.get("symptoms", []),
            },

            # ── Impact assessment ────────────────────────────────────────────
            "impact": {
                "yield_loss_estimate_pct": YIELD_LOSS_BY_SEVERITY.get(severity, 20),
                "spread_probability_pct":  spread_pct,
                "days_to_recovery":        RECOVERY_DAYS_BY_SEVERITY.get(severity, 14),
                "treatment_urgency":       URGENCY_BY_SEVERITY.get(severity, "As soon as possible"),
            },

            # ── Treatments ───────────────────────────────────────────────────
            "treatments": disease_info.get("treatments", {}),

            # ── Explainability ───────────────────────────────────────────────
            "xai": {
                "gradcam_heatmap_b64": gradcam_b64,
                "explanation": (
                    f"The AI detected {disease_name} patterns in the highlighted region. "
                    f"Confidence: {confidence:.1f}%. "
                    f"The heatmap shows which leaf areas most influenced this decision."
                ) if gradcam_b64 else "Grad-CAM not available for this prediction.",
            },

            # ── Model metadata ───────────────────────────────────────────────
            "models_used": {
                "yolo":          self.yolo is not None and self.yolo.model_loaded,
                "efficientnet":  self.efficientnet is not None,
                "swin":          self.swin is not None,
                "gradcam":       self.gradcam is not None,
            },
            "all_predictions":    merged,
            "yolo_raw_detections": yolo_result.get("detections", []),
            "annotated_image":    yolo_result.get("annotated_image"),
        }

        return report

    def cleanup(self):
        """Release Grad-CAM hooks to prevent memory leaks."""
        if self.gradcam:
            self.gradcam.remove_hooks()


# ─────────────────────────────────────────────
# Singleton factory
# ─────────────────────────────────────────────
_pipeline_instance: EnsemblePipeline | None = None


def get_pipeline(
    yolo_model_path:      str = None,
    efficientnet_weights: str = None,
    swin_weights:         str = None,
) -> EnsemblePipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = EnsemblePipeline(
            yolo_model_path=yolo_model_path,
            efficientnet_weights=efficientnet_weights,
            swin_weights=swin_weights,
        )
    return _pipeline_instance
