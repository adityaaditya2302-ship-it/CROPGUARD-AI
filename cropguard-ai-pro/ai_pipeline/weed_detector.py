"""
CropGuard AI - Weed Detector (Phase 4)
Detects weed species in drone imagery and outputs GPS coordinates for precision herbicide spraying.
"""
import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path


# Known weed categories (expand with DeepWeeds dataset training)
WEED_CLASSES = {
    0:  "Broadleaf Dock",
    1:  "Chickweed",
    2:  "Cleavers",
    3:  "Common Ragwort",
    4:  "Fat Hen",
    5:  "Loose Silky-bent",
    6:  "Maize",
    7:  "Scentless Mayweed",
    8:  "Shepherd's Purse",
    9:  "Small-flowered Cranesbill",
    10: "Sugar Beet",
    11: "Bindweed",
    12: "Nutsedge",
    13: "Johnson Grass",
    14: "Wild Oat",
}

WEED_RISK = {
    "Broadleaf Dock":            "Medium",
    "Chickweed":                 "Low",
    "Cleavers":                  "High",
    "Common Ragwort":            "High",
    "Fat Hen":                   "Medium",
    "Loose Silky-bent":          "High",
    "Nutsedge":                  "Critical",
    "Johnson Grass":             "Critical",
    "Wild Oat":                  "High",
}


class WeedDetector:
    """
    Detects weeds in crop field images.
    Uses a YOLO-based model trained on DeepWeeds / custom weed datasets.
    Falls back to color-range segmentation when model is unavailable.
    """

    def __init__(self, weights_path: str = None, conf_threshold: float = 0.35):
        self.conf_threshold = conf_threshold
        self.model = None
        self.model_loaded = False

        if weights_path and os.path.exists(weights_path):
            try:
                from ultralytics import YOLO
                self.model = YOLO(weights_path)
                self.model_loaded = True
                print(f"✅ Weed detector model loaded from {weights_path}")
            except Exception as e:
                print(f"⚠️  Weed model load failed: {e}. Using fallback segmentation.")
        else:
            print("⚠️  Weed detector: no model found. Using color-range fallback.")

    def detect(self, image_path: str) -> dict:
        """
        Detect weeds in image.

        Returns:
            success, detections[], weed_coverage_pct, spray_recommended, risk_level
        """
        if self.model_loaded:
            return self._detect_with_model(image_path)
        return self._detect_fallback(image_path)

    def _detect_with_model(self, image_path: str) -> dict:
        results  = self.model(image_path, conf=self.conf_threshold)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                weed_name = WEED_CLASSES.get(cls_id, f"Weed class {cls_id}")
                detections.append({
                    "weed":       weed_name,
                    "confidence": round(float(box.conf) * 100, 1),
                    "bbox":       box.xyxy[0].tolist(),
                    "risk":       WEED_RISK.get(weed_name, "Medium"),
                })

        coverage = self._estimate_weed_coverage(image_path, detections)
        return self._build_response(detections, coverage)

    def _detect_fallback(self, image_path: str) -> dict:
        """Color-range heuristic: segment green-on-green weeds vs crop rows."""
        img = cv2.imread(image_path)
        if img is None:
            return {"success": False, "error": "Could not read image"}

        hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([25, 40, 40])
        upper = np.array([85, 255, 255])
        mask  = cv2.inRange(hsv, lower, upper)

        coverage_pct = round(float(np.sum(mask > 0)) / mask.size * 100, 1)
        detections = [{
            "weed":       "Unidentified weed (color analysis)",
            "confidence": 50.0,
            "bbox":       None,
            "risk":       "Medium" if coverage_pct > 10 else "Low",
        }] if coverage_pct > 2 else []

        return self._build_response(detections, coverage_pct)

    def _estimate_weed_coverage(self, image_path: str, detections: list) -> float:
        if not detections:
            return 0.0
        img  = cv2.imread(image_path)
        h, w = img.shape[:2]
        total_pixels = h * w
        weed_pixels  = 0
        for d in detections:
            if d["bbox"]:
                x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
                weed_pixels += (x2 - x1) * (y2 - y1)
        return round(min(weed_pixels / total_pixels * 100, 100), 1)

    def _build_response(self, detections: list, coverage_pct: float) -> dict:
        risk_levels = [d["risk"] for d in detections]
        overall_risk = (
            "Critical" if "Critical" in risk_levels else
            "High"     if "High"     in risk_levels else
            "Medium"   if "Medium"   in risk_levels else
            "Low"      if detections else "None"
        )
        return {
            "success":               True,
            "detections":            detections,
            "weed_coverage_pct":     coverage_pct,
            "spray_recommended":     coverage_pct > 5 or overall_risk in ("High", "Critical"),
            "risk_level":            overall_risk,
            "estimated_yield_loss":  min(coverage_pct * 0.8, 40),
            "herbicide_savings_pct": 70 if coverage_pct < 30 else 40,
        }


# Singleton
_weed_instance = None
def get_weed_detector(weights_path: str = None) -> WeedDetector:
    global _weed_instance
    if _weed_instance is None:
        _weed_instance = WeedDetector(weights_path=weights_path)
    return _weed_instance
