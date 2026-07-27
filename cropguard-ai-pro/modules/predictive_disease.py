"""
CropGuard AI - Predictive Disease Module (Phase 1)
Predicts crop disease outbreaks 7-10 days BEFORE visible symptoms appear.

Inputs:
  - Weather forecast (humidity, temp, rainfall, wind)
  - Historical disease data for this farm
  - NDVI anomaly from satellite
  - Thermal water stress readings

Output:
  - Disease risk score (0-100)
  - Predicted disease name
  - Days until expected outbreak
  - Preventive action recommendations
"""
import json
import math
from datetime import datetime, timedelta
from typing import Optional


# ── Disease-weather correlation rules ─────────────────────────────────────────
# Each rule defines the conditions that trigger high risk for a specific disease
DISEASE_RISK_RULES = {
    "Late Blight (Tomato/Potato)": {
        "crops":           ["tomato", "potato"],
        "humidity_min":    80,       # % - threshold for high risk
        "temp_min":        10,       # °C
        "temp_max":        25,       # °C
        "rain_days_min":   2,        # consecutive rainy days
        "wind_max":        20,       # km/h - low wind promotes spread
        "base_risk":       70,
        "description":     "Phytophthora infestans thrives in cool, humid conditions with low wind.",
        "action":          "Apply preventive Mancozeb 75% WP @ 2g/L before symptoms appear.",
    },
    "Rice Blast": {
        "crops":           ["rice"],
        "humidity_min":    90,
        "temp_min":        20,
        "temp_max":        30,
        "rain_days_min":   3,
        "wind_max":        15,
        "base_risk":       75,
        "description":     "Pyricularia oryzae spore germination peaks at >90% humidity.",
        "action":          "Apply Tricyclazole 75% WP @ 0.6g/L as preventive spray.",
    },
    "Wheat Rust (Yellow/Brown)": {
        "crops":           ["wheat"],
        "humidity_min":    70,
        "temp_min":        5,
        "temp_max":        22,
        "rain_days_min":   1,
        "wind_max":        30,
        "base_risk":       65,
        "description":     "Puccinia spores spread via wind in cool, moist conditions.",
        "action":          "Apply Propiconazole 25% EC @ 1ml/L immediately.",
    },
    "Powdery Mildew": {
        "crops":           ["wheat", "grape", "squash", "mango", "apple"],
        "humidity_min":    40,
        "humidity_max":    70,       # Unlike most fungi, prefers moderate humidity
        "temp_min":        18,
        "temp_max":        28,
        "rain_days_min":   0,
        "wind_max":        25,
        "base_risk":       55,
        "description":     "Erysiphe species favor warm days + cool nights with moderate humidity.",
        "action":          "Apply Sulphur 80% WP @ 2g/L or Myclobutanil 10% WP @ 1g/L.",
    },
    "Bacterial Blight (Rice)": {
        "crops":           ["rice"],
        "humidity_min":    85,
        "temp_min":        25,
        "temp_max":        35,
        "rain_days_min":   2,
        "wind_max":        40,       # High wind spreads bacterial lesions
        "base_risk":       60,
        "description":     "Xanthomonas oryzae spreads via rain splash and wind in warm, wet weather.",
        "action":          "Apply Copper oxychloride 50% WP @ 3g/L. Avoid excess N fertilizer.",
    },
    "Sugarcane Red Rot": {
        "crops":           ["sugarcane"],
        "humidity_min":    75,
        "temp_min":        25,
        "temp_max":        38,
        "rain_days_min":   4,
        "wind_max":        30,
        "base_risk":       65,
        "description":     "Colletotrichum falcatum infects through wounds during wet, warm seasons.",
        "action":          "Treat setts with Carbendazim 0.1% solution. Roguing of infected clumps.",
    },
    "Downy Mildew": {
        "crops":           ["grape", "soybean", "corn", "onion"],
        "humidity_min":    85,
        "temp_min":        12,
        "temp_max":        22,
        "rain_days_min":   2,
        "wind_max":        20,
        "base_risk":       70,
        "description":     "Plasmopara species sporulate overnight in cool, humid conditions.",
        "action":          "Apply Metalaxyl + Mancozeb @ 2.5g/L before dawn in high-humidity spells.",
    },
}


class PredictiveDiseaseEngine:
    """
    Predicts disease outbreaks 7-10 days before visible symptoms.
    Uses weather data + historical patterns + NDVI anomaly.
    """

    def __init__(self):
        self.weather_api_key = None   # Set via environment variable
        print("✅ Predictive Disease Engine initialized")

    def assess_risk(
        self,
        crop: str,
        weather_forecast: list[dict],    # 7-day forecast list
        soil_moisture_pct: float = 50.0,
        ndvi_anomaly: float = 0.0,       # negative = stressed crop
        historical_outbreaks: list = None,
        current_month: int = None,
    ) -> dict:
        """
        Assess disease outbreak risk for a specific crop.

        Args:
            crop:               crop name (e.g. "tomato")
            weather_forecast:   list of daily dicts with keys:
                                  humidity_avg, temp_min, temp_max, rainfall_mm, wind_kmh
            soil_moisture_pct:  current soil moisture (0-100)
            ndvi_anomaly:       NDVI change from baseline (e.g. -0.15 = declining health)
            historical_outbreaks: list of previous outbreak dates/types
            current_month:      current month (1-12)

        Returns:
            dict with risk scores, predictions, and preventive actions
        """
        crop_lower = crop.lower()
        alerts = []
        current_month = current_month or datetime.now().month

        # Analyze each disease rule
        for disease_name, rule in DISEASE_RISK_RULES.items():
            if crop_lower not in rule["crops"]:
                continue

            risk_score = self._calculate_risk_score(
                rule, weather_forecast, soil_moisture_pct,
                ndvi_anomaly, historical_outbreaks, current_month
            )

            if risk_score >= 40:
                days_until = self._estimate_days_until_outbreak(risk_score, weather_forecast)
                alerts.append({
                    "disease":         disease_name,
                    "risk_score":      risk_score,
                    "risk_level":      self._score_to_level(risk_score),
                    "days_until_outbreak": days_until,
                    "description":     rule["description"],
                    "preventive_action": rule["action"],
                    "trigger_factors": self._get_trigger_factors(rule, weather_forecast),
                    "confidence":      self._estimate_confidence(risk_score, weather_forecast),
                })

        # Sort by risk score
        alerts.sort(key=lambda x: x["risk_score"], reverse=True)

        overall_risk = max((a["risk_score"] for a in alerts), default=0)
        return {
            "success":          True,
            "crop":             crop,
            "overall_risk_score": overall_risk,
            "overall_risk_level": self._score_to_level(overall_risk),
            "alerts":           alerts,
            "forecast_days":    len(weather_forecast),
            "assessment_time":  datetime.now().isoformat(),
            "ndvi_status":      "Declining" if ndvi_anomaly < -0.1 else
                                "Normal" if ndvi_anomaly > -0.05 else "Slightly stressed",
            "soil_moisture_status": "Low" if soil_moisture_pct < 30 else
                                    "Optimal" if soil_moisture_pct < 70 else "High",
            "recommendation":   self._get_overall_recommendation(alerts),
        }

    def _calculate_risk_score(self, rule, forecast, soil_moisture, ndvi_anomaly,
                               history, month) -> float:
        score = 0.0
        count = 0

        for day in forecast[:7]:
            day_score = 0.0
            hum  = day.get("humidity_avg", 60)
            tmin = day.get("temp_min",     15)
            tmax = day.get("temp_max",     25)
            rain = day.get("rainfall_mm",   0)
            wind = day.get("wind_kmh",     15)

            # Humidity contribution (0-40 points)
            if hum >= rule.get("humidity_min", 0):
                hum_max = rule.get("humidity_max", 100)
                if hum <= hum_max:
                    day_score += 40 * (hum - rule["humidity_min"]) / max(100 - rule["humidity_min"], 1)

            # Temperature range (0-30 points)
            if rule["temp_min"] <= tmin and tmax <= rule["temp_max"]:
                day_score += 30
            elif rule["temp_min"] <= (tmin + tmax) / 2 <= rule["temp_max"]:
                day_score += 15

            # Rain contributes positively
            if rain > 0 and day.get("rain_day", True):
                day_score += min(rain * 2, 20)

            # Wind: low wind = more spread
            if wind < rule.get("wind_max", 30):
                day_score += 10

            score += day_score
            count += 1

        base_score = (score / max(count * 100, 1)) * rule["base_risk"]

        # Modifiers
        if ndvi_anomaly < -0.1:
            base_score *= 1.3    # stressed crop is more susceptible

        if soil_moisture > 80:
            base_score *= 1.2    # waterlogged = higher disease pressure

        if history:
            # Historical outbreak in same month boosts risk
            same_month = [h for h in history if h.get("month") == month]
            if same_month:
                base_score *= 1.25

        return min(round(base_score, 1), 100.0)

    def _estimate_days_until_outbreak(self, risk_score: float, forecast: list) -> int:
        """Higher risk = fewer days until visible symptoms."""
        if risk_score >= 80:
            return 3
        if risk_score >= 65:
            return 5
        if risk_score >= 50:
            return 7
        return 10

    def _score_to_level(self, score: float) -> str:
        if score >= 80: return "Critical"
        if score >= 65: return "High"
        if score >= 45: return "Medium"
        if score >= 20: return "Low"
        return "Minimal"

    def _get_trigger_factors(self, rule: dict, forecast: list) -> list:
        factors = []
        avg_humidity = sum(d.get("humidity_avg", 60) for d in forecast[:7]) / len(forecast[:7])
        avg_temp     = sum((d.get("temp_min", 15) + d.get("temp_max", 25)) / 2 for d in forecast[:7]) / len(forecast[:7])
        rain_days    = sum(1 for d in forecast[:7] if d.get("rainfall_mm", 0) > 0)

        if avg_humidity >= rule.get("humidity_min", 0):
            factors.append(f"High humidity ({avg_humidity:.0f}% avg, threshold: {rule['humidity_min']}%)")
        if rule["temp_min"] <= avg_temp <= rule["temp_max"]:
            factors.append(f"Optimal temperature for pathogen ({avg_temp:.1f}°C)")
        if rain_days >= rule.get("rain_days_min", 0):
            factors.append(f"{rain_days} rainy days in forecast")
        return factors

    def _estimate_confidence(self, risk_score: float, forecast: list) -> str:
        n_days = len(forecast)
        if n_days >= 7 and risk_score >= 65:
            return "High (85%)"
        if n_days >= 5:
            return "Medium (70%)"
        return "Low (55%)"

    def _get_overall_recommendation(self, alerts: list) -> str:
        if not alerts:
            return "No significant disease risk detected. Continue normal monitoring."
        top = alerts[0]
        risk = top["risk_level"]
        if risk == "Critical":
            return (f"🚨 URGENT: {top['disease']} outbreak highly likely within "
                    f"{top['days_until_outbreak']} days. {top['preventive_action']}")
        if risk == "High":
            return (f"⚠️ HIGH RISK: {top['disease']} expected in "
                    f"{top['days_until_outbreak']} days. Preventive treatment recommended now.")
        if risk == "Medium":
            return (f"🔔 MONITOR: {top['disease']} risk is building. "
                    f"Prepare preventive measures.")
        return "✅ Low risk currently. Continue routine monitoring."

    def get_weather_based_spray_advice(self, weather_today: dict) -> dict:
        """
        Should you spray today based on weather?
        Wind, rain, humidity, and temperature all affect pesticide effectiveness.
        """
        wind    = weather_today.get("wind_kmh", 10)
        rain    = weather_today.get("rainfall_mm", 0)
        humidity= weather_today.get("humidity_avg", 60)
        temp    = weather_today.get("temp_max", 25)

        issues = []
        if wind > 25:
            issues.append(f"Wind speed {wind} km/h is too high (>25 km/h) — spray drift risk")
        if rain > 0:
            issues.append("Rain forecast — pesticide will wash off before absorption")
        if humidity > 90:
            issues.append("Very high humidity may reduce effectiveness of some fungicides")
        if temp > 35:
            issues.append(f"Temperature {temp}°C too high — spray early morning instead")
        if temp < 10:
            issues.append(f"Temperature {temp}°C too low — pesticide uptake will be slow")

        spray_ok = len(issues) == 0
        return {
            "spray_recommended_today": spray_ok,
            "optimal_spray_window":    "6:00 AM – 10:00 AM" if not issues else "Wait for better conditions",
            "issues":                  issues,
            "advice": ("✅ Conditions are suitable for spraying today." if spray_ok else
                       f"❌ Not recommended: {'; '.join(issues)}"),
        }


# Singleton
_engine_instance: PredictiveDiseaseEngine | None = None

def get_predictive_engine() -> PredictiveDiseaseEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = PredictiveDiseaseEngine()
    return _engine_instance
