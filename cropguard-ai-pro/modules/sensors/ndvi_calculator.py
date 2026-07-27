"""
CropGuard AI - NDVI Calculator (Phase 3/4)
Computes Normalized Difference Vegetation Index from multispectral images.

NDVI = (NIR - Red) / (NIR + Red)

NDVI range: -1 to +1
  > 0.6  : Dense, healthy vegetation
  0.4-0.6: Moderate vegetation
  0.2-0.4: Sparse vegetation / stressed
  < 0.2  : Bare soil / severe stress
"""
import os
import cv2
import numpy as np
from pathlib import Path


# NDVI health thresholds
NDVI_THRESHOLDS = {
    "excellent": (0.60, 1.00),
    "good":      (0.40, 0.60),
    "moderate":  (0.20, 0.40),
    "stressed":  (0.05, 0.20),
    "critical":  (-1.0, 0.05),
}

NDVI_COLORS_BGR = {
    "excellent": (0, 200, 0),    # Green
    "good":      (100, 200, 50), # Light green
    "moderate":  (0, 200, 200),  # Yellow
    "stressed":  (0, 100, 255),  # Orange
    "critical":  (0, 0, 255),    # Red
}


class NDVICalculator:
    """
    Computes NDVI from either:
      1. True multispectral images (NIR + Red bands)
      2. RGB-approximated NDVI using red-edge simulation (less accurate)
    """

    def from_multispectral(self, nir_path: str, red_path: str) -> dict:
        """
        Compute NDVI from separate NIR and Red band images.
        True multispectral (e.g. MicaSense, Parrot Sequoia).
        """
        nir = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE).astype(float)
        red = cv2.imread(red_path, cv2.IMREAD_GRAYSCALE).astype(float)

        if nir is None or red is None:
            return {"success": False, "error": "Could not read multispectral images"}

        # Ensure same size
        if nir.shape != red.shape:
            red = cv2.resize(red, (nir.shape[1], nir.shape[0]))

        ndvi = self._compute_ndvi(nir, red)
        return self._build_result(ndvi, source="multispectral")

    def from_rgb(self, rgb_path: str) -> dict:
        """
        Approximate NDVI from standard RGB image.
        Uses Red channel as proxy. Less accurate than true multispectral.
        """
        img = cv2.imread(rgb_path)
        if img is None:
            return {"success": False, "error": "Could not read RGB image"}

        # Extract channels
        b, g, r = cv2.split(img.astype(float))

        # RGB-NDVI approximation (uses green as proxy for NIR)
        # More accurate than nothing, less accurate than true NIR sensor
        numerator   = g - r
        denominator = g + r + 1e-6  # avoid /0
        ndvi = np.clip(numerator / denominator, -1, 1)

        return self._build_result(ndvi, source="rgb_approximation")

    def _compute_ndvi(self, nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        nir_f = nir.astype(float)
        red_f = red.astype(float)
        denominator = nir_f + red_f
        denominator[denominator == 0] = 1e-6
        ndvi = (nir_f - red_f) / denominator
        return np.clip(ndvi, -1, 1)

    def _build_result(self, ndvi: np.ndarray, source: str) -> dict:
        mean_ndvi = float(np.mean(ndvi))
        std_ndvi  = float(np.std(ndvi))

        # Zone coverage percentages
        coverage = {}
        for zone, (low, high) in NDVI_THRESHOLDS.items():
            mask = (ndvi >= low) & (ndvi < high)
            coverage[zone] = round(float(np.sum(mask)) / ndvi.size * 100, 1)

        # Dominant health status
        status = max(coverage, key=coverage.get)

        # Generate colorized heatmap
        heatmap_bgr = self._colorize_ndvi(ndvi)
        heatmap_b64 = self._to_base64(heatmap_bgr)

        return {
            "success":       True,
            "source":        source,
            "mean_ndvi":     round(mean_ndvi, 3),
            "std_ndvi":      round(std_ndvi, 3),
            "min_ndvi":      round(float(np.min(ndvi)), 3),
            "max_ndvi":      round(float(np.max(ndvi)), 3),
            "health_status": status,
            "coverage_pct":  coverage,
            "heatmap_b64":   heatmap_b64,
            "interpretation": self._interpret(mean_ndvi, coverage),
            "action_needed":  status in ("stressed", "critical"),
        }

    def _colorize_ndvi(self, ndvi: np.ndarray) -> np.ndarray:
        """Convert NDVI array to false-color BGR image."""
        normalized = ((ndvi + 1) / 2 * 255).astype(np.uint8)
        colored    = cv2.applyColorMap(normalized, cv2.COLORMAP_RdYlGn)
        return colored

    def _to_base64(self, image: np.ndarray) -> str:
        import base64
        _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf).decode("utf-8")

    def _interpret(self, mean_ndvi: float, coverage: dict) -> str:
        if mean_ndvi >= 0.6:
            return f"Excellent crop health. Dense, vigorous vegetation. NDVI: {mean_ndvi:.2f}"
        if mean_ndvi >= 0.4:
            return f"Good crop health. Normal growth pattern. NDVI: {mean_ndvi:.2f}"
        if mean_ndvi >= 0.2:
            return (f"Moderate stress detected. Consider irrigation/fertilizer. "
                    f"NDVI: {mean_ndvi:.2f}. Stressed area: {coverage.get('stressed', 0):.0f}%")
        if mean_ndvi >= 0.05:
            return (f"⚠️ Significant crop stress. Immediate action needed. "
                    f"NDVI: {mean_ndvi:.2f}. Critical area: {coverage.get('critical', 0):.0f}%")
        return (f"🚨 Severe stress or crop failure in major areas. "
                f"NDVI: {mean_ndvi:.2f}. Investigate urgently.")


# Convenience function
_calculator: NDVICalculator | None = None

def calculate_ndvi(image_path: str, mode: str = "rgb") -> dict:
    """Quick NDVI calculation from a single RGB image path."""
    global _calculator
    if _calculator is None:
        _calculator = NDVICalculator()
    return _calculator.from_rgb(image_path)
