"""
CropGuard AI - Pest Detector (Phase 4)
Detects insects, larvae, eggs, and pest infestations in crop images.
Generates pest density heatmaps and alerts for drone spray missions.
"""
import os
import cv2
import numpy as np


# Pest taxonomy (training target classes)
PEST_CLASSES = {
    0:  {"name": "Aphid",              "type": "insect",  "risk": "High"},
    1:  {"name": "Armyworm",           "type": "larvae",  "risk": "Critical"},
    2:  {"name": "Bollworm",           "type": "larvae",  "risk": "Critical"},
    3:  {"name": "Brown Planthopper",  "type": "insect",  "risk": "High"},
    4:  {"name": "Diamondback Moth",   "type": "larvae",  "risk": "High"},
    5:  {"name": "Fall Armyworm",      "type": "larvae",  "risk": "Critical"},
    6:  {"name": "Fruit Fly",          "type": "insect",  "risk": "Medium"},
    7:  {"name": "Green Leaf Hopper",  "type": "insect",  "risk": "High"},
    8:  {"name": "Locust",             "type": "insect",  "risk": "Critical"},
    9:  {"name": "Mealybug",           "type": "insect",  "risk": "Medium"},
    10: {"name": "Mite (Spider/Red)",  "type": "mite",    "risk": "Medium"},
    11: {"name": "Stem Borer",         "type": "larvae",  "risk": "Critical"},
    12: {"name": "Thrips",             "type": "insect",  "risk": "High"},
    13: {"name": "Whitefly",           "type": "insect",  "risk": "High"},
    14: {"name": "Aphid Eggs",         "type": "eggs",    "risk": "Medium"},
    15: {"name": "Moth Eggs",          "type": "eggs",    "risk": "High"},
}

TREATMENT_BY_PEST = {
    "Aphid":            {"chemical": "Imidacloprid 17.8% SL @ 0.5ml/L", "organic": "Neem oil + soap spray"},
    "Armyworm":         {"chemical": "Chlorpyrifos 20% EC @ 2ml/L",      "organic": "Bacillus thuringiensis (Bt) spray"},
    "Bollworm":         {"chemical": "Emamectin Benzoate 5% SG @ 0.4g/L","organic": "NPV (Nuclear Polyhedrosis Virus)"},
    "Brown Planthopper":{"chemical": "Thiamethoxam 25% WG @ 0.3g/L",     "organic": "Push-pull trap cropping"},
    "Fall Armyworm":    {"chemical": "Spinetoram 11.7% SC @ 0.5ml/L",    "organic": "Trichogramma biological control"},
    "Locust":           {"chemical": "Chlorpyrifos 50% EC aerial spray",  "organic": "Green Muscle (Metarhizium fungus)"},
    "Stem Borer":       {"chemical": "Carbofuran 3G granules in whorl",   "organic": "Trichogramma egg parasitoid"},
    "Whitefly":         {"chemical": "Spirotetramat 15% OD @ 1ml/L",     "organic": "Yellow sticky traps + reflective mulch"},
}


class PestDetector:
    """
    Detects pests in crop field images using YOLO (when available)
    or a color/texture heuristic fallback.
    """

    def __init__(self, weights_path: str = None, conf_threshold: float = 0.30):
        self.conf_threshold = conf_threshold
        self.model = None
        self.model_loaded = False

        if weights_path and os.path.exists(weights_path):
            try:
                from ultralytics import YOLO
                self.model = YOLO(weights_path)
                self.model_loaded = True
                print(f"✅ Pest detector loaded from {weights_path}")
            except Exception as e:
                print(f"⚠️  Pest model load failed: {e}")
        else:
            print("⚠️  Pest detector: no trained model found. Using heuristic fallback.")

    def detect(self, image_path: str) -> dict:
        if self.model_loaded:
            return self._detect_with_model(image_path)
        return self._detect_fallback(image_path)

    def _detect_with_model(self, image_path: str) -> dict:
        results = self.model(image_path, conf=self.conf_threshold)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id   = int(box.cls)
                pest_cfg = PEST_CLASSES.get(cls_id, {"name": f"Unknown pest {cls_id}", "type": "insect", "risk": "Medium"})
                name     = pest_cfg["name"]
                treatment = TREATMENT_BY_PEST.get(name, {
                    "chemical": "Consult local extension office",
                    "organic":  "Neem oil spray",
                })
                detections.append({
                    "pest":       name,
                    "type":       pest_cfg["type"],
                    "risk":       pest_cfg["risk"],
                    "confidence": round(float(box.conf) * 100, 1),
                    "bbox":       box.xyxy[0].tolist(),
                    "treatment":  treatment,
                })
        return self._build_response(detections)

    def _detect_fallback(self, image_path: str) -> dict:
        """Heuristic: detect tiny dot patterns typical of aphid/mite infestations."""
        img = cv2.imread(image_path)
        if img is None:
            return {"success": False, "error": "Could not read image"}

        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred= cv2.GaussianBlur(gray, (5, 5), 0)
        _, th  = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        small_dots = [c for c in contours if 1 < cv2.contourArea(c) < 50]
        pest_density = len(small_dots)

        detections = []
        if pest_density > 30:
            detections.append({
                "pest":       "Possible mite/aphid (heuristic)",
                "type":       "insect",
                "risk":       "Medium",
                "confidence": 45.0,
                "bbox":       None,
                "treatment":  TREATMENT_BY_PEST.get("Aphid", {}),
            })
        return self._build_response(detections, pest_density)

    def _build_response(self, detections: list, extra_count: int = 0) -> dict:
        risk_levels = [d["risk"] for d in detections]
        overall_risk = (
            "Critical" if "Critical" in risk_levels else
            "High"     if "High"     in risk_levels else
            "Medium"   if "Medium"   in risk_levels else
            "Low"      if detections else "None"
        )
        spray_needed = overall_risk in ("High", "Critical")
        return {
            "success":          True,
            "detections":       detections,
            "pest_count":       len(detections) + extra_count,
            "overall_risk":     overall_risk,
            "spray_needed":     spray_needed,
            "spray_urgency":    "Immediately" if overall_risk == "Critical" else
                                "Within 48 hours" if spray_needed else "Monitor only",
            "economic_threshold_exceeded": overall_risk in ("High", "Critical"),
        }


# Singleton
_pest_instance = None
def get_pest_detector(weights_path: str = None) -> PestDetector:
    global _pest_instance
    if _pest_instance is None:
        _pest_instance = PestDetector(weights_path=weights_path)
    return _pest_instance
