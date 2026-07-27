"""
CropGuard AI - Yield Predictor (Phase 4)
Predicts crop yield using multi-source data fusion.

Inputs:
  - NDVI history (crop health trend)
  - Soil NPK levels
  - Weather data (rainfall, temperature)
  - Disease impact estimates
  - Crop growth stage
  - Historical yield data for farm

Output:
  - Expected yield (tonnes/acre)
  - Harvest date estimate
  - Market value estimate
  - Weekly yield update
"""
from datetime import datetime, timedelta


# Crop-specific baseline yields (tonnes/acre under optimal conditions)
CROP_BASELINE_YIELD = {
    "rice":       2.5,
    "wheat":      1.8,
    "tomato":     12.0,
    "potato":     8.0,
    "maize":      2.0,
    "sugarcane":  35.0,
    "cotton":     0.6,
    "soybean":    0.8,
    "onion":      5.0,
    "chilli":     1.5,
    "mango":      4.0,
    "banana":     10.0,
    "apple":      3.5,
    "grape":      5.0,
    "groundnut":  1.0,
}

# Crop growing seasons (weeks from sowing to harvest)
CROP_SEASON_WEEKS = {
    "rice":      14, "wheat": 18, "tomato": 10, "potato": 12,
    "maize":     12, "sugarcane": 52, "cotton": 20, "soybean": 14,
    "onion":     16, "chilli": 16, "mango": 0,  "banana": 40,
}

# Approximate market prices (INR per quintal, 2024 MSP/market rates)
CROP_MARKET_PRICE_PER_QUINTAL = {
    "rice":       2183,   # MSP
    "wheat":      2275,   # MSP
    "tomato":     1500,
    "potato":     1200,
    "maize":      2090,
    "sugarcane":  350,    # per quintal cane
    "cotton":     7020,   # MSP
    "soybean":    4600,
    "onion":      1800,
    "chilli":     8000,
    "mango":      4000,
    "banana":     1200,
    "apple":      6000,
    "grape":      3500,
    "groundnut":  6377,
}


class YieldPredictor:
    """
    Predicts crop yield using a rule-based model enriched with real data.
    Can be upgraded to ML regression when training data is available.
    """

    def __init__(self):
        print("✅ Yield Predictor initialized")

    def predict(
        self,
        crop: str,
        field_area_acres: float,
        sowing_date: str = None,              # ISO date string
        ndvi_current: float = 0.5,            # current NDVI (0-1)
        ndvi_trend: float = 0.0,              # weekly NDVI change (+/-)
        disease_yield_loss_pct: float = 0.0,  # from disease detection
        soil_nitrogen: float = 100.0,          # mg/kg
        soil_moisture_pct: float = 50.0,
        rainfall_mm_7day: float = 15.0,
        temp_avg_c: float = 25.0,
        fertilizer_applied: bool = True,
        irrigation_adequate: bool = True,
    ) -> dict:
        """
        Predict yield for a field.

        Returns comprehensive yield report.
        """
        crop_lower = crop.lower()
        baseline   = CROP_BASELINE_YIELD.get(crop_lower, 1.5)
        price_q    = CROP_MARKET_PRICE_PER_QUINTAL.get(crop_lower, 2000)

        # ── Calculate yield modifiers ──────────────────────────────────────────
        modifiers = {}

        # NDVI health modifier (±30%)
        ndvi_factor = max(0.5, min(1.3, 0.5 + ndvi_current * 1.0))
        modifiers["ndvi_health"] = ndvi_factor

        # NDVI trend modifier (±10%)
        trend_factor = max(0.9, min(1.1, 1.0 + ndvi_trend * 5))
        modifiers["ndvi_trend"] = trend_factor

        # Disease loss
        disease_factor = 1.0 - (disease_yield_loss_pct / 100)
        modifiers["disease_loss"] = disease_factor

        # Soil nitrogen
        n_factor = max(0.7, min(1.1, 0.7 + soil_nitrogen / 300))
        modifiers["soil_nitrogen"] = n_factor

        # Rainfall
        optimal_rain = 20  # mm/week
        rain_factor  = max(0.8, min(1.1, 1 - abs(rainfall_mm_7day - optimal_rain) / 100))
        modifiers["rainfall"] = rain_factor

        # Temperature
        temp_factor = max(0.8, min(1.05, 1 - abs(temp_avg_c - 25) / 50))
        modifiers["temperature"] = temp_factor

        # Management factors
        modifiers["fertilizer"]  = 1.0 if fertilizer_applied  else 0.85
        modifiers["irrigation"]  = 1.0 if irrigation_adequate else 0.80

        # Combined yield factor
        yield_factor = 1.0
        for f in modifiers.values():
            yield_factor *= f

        # ── Predicted yield ────────────────────────────────────────────────────
        predicted_yield_per_acre = round(baseline * yield_factor, 2)
        total_yield_tonnes       = round(predicted_yield_per_acre * field_area_acres, 2)
        total_yield_quintals     = round(total_yield_tonnes * 10, 1)  # 1 tonne = 10 quintal
        market_value_inr         = round(total_yield_quintals * price_q)

        # ── Harvest date estimate ──────────────────────────────────────────────
        season_weeks = CROP_SEASON_WEEKS.get(crop_lower, 14)
        if sowing_date:
            try:
                sow = datetime.fromisoformat(sowing_date)
                harvest_date = sow + timedelta(weeks=season_weeks)
            except ValueError:
                harvest_date = datetime.now() + timedelta(weeks=4)
        else:
            harvest_date = datetime.now() + timedelta(weeks=4)

        days_to_harvest = max(0, (harvest_date - datetime.now()).days)

        # ── Improvement recommendations ────────────────────────────────────────
        recommendations = self._get_recommendations(
            ndvi_current, disease_yield_loss_pct, soil_nitrogen,
            rainfall_mm_7day, fertilizer_applied, irrigation_adequate, crop_lower
        )

        # ── Confidence score ───────────────────────────────────────────────────
        confidence_factors = sum([
            ndvi_current > 0.3,
            disease_yield_loss_pct < 20,
            soil_nitrogen > 50,
            fertilizer_applied,
            irrigation_adequate,
        ])
        confidence_pct = 60 + confidence_factors * 6

        return {
            "success":                 True,
            "crop":                    crop,
            "field_area_acres":        field_area_acres,

            # Yield prediction
            "predicted_yield_per_acre": predicted_yield_per_acre,
            "total_yield_tonnes":      total_yield_tonnes,
            "total_yield_quintals":    total_yield_quintals,
            "baseline_yield_per_acre": baseline,
            "yield_efficiency_pct":    round(yield_factor * 100, 1),

            # Harvest
            "estimated_harvest_date":  harvest_date.strftime("%B %d, %Y"),
            "days_to_harvest":         days_to_harvest,

            # Market
            "market_price_per_quintal": price_q,
            "estimated_market_value_inr": market_value_inr,
            "estimated_profit_inr":    round(market_value_inr * 0.65),  # ~35% input cost

            # Health snapshot
            "crop_health": {
                "ndvi":              ndvi_current,
                "ndvi_trend":        "Improving" if ndvi_trend > 0.01 else
                                     "Declining" if ndvi_trend < -0.01 else "Stable",
                "disease_impact_pct": disease_yield_loss_pct,
                "soil_n_status":     "Low" if soil_nitrogen < 50 else "Adequate",
            },

            # Modifiers breakdown
            "yield_modifiers":         {k: round(v, 3) for k, v in modifiers.items()},

            # Advice
            "recommendations":        recommendations,
            "confidence_pct":         confidence_pct,
            "prediction_date":        datetime.now().isoformat(),
        }

    def _get_recommendations(self, ndvi, disease_loss, soil_n,
                             rainfall, fertilizer, irrigation, crop) -> list:
        recs = []

        if ndvi < 0.4:
            recs.append("🌿 Low NDVI detected – check for nutrient deficiency or water stress")
        if disease_loss > 10:
            recs.append(f"💊 Disease is reducing yield by {disease_loss:.0f}% – treat immediately")
        if soil_n < 50:
            recs.append("🧪 Low nitrogen – apply Urea @ 50kg/acre as top dressing")
        if not fertilizer:
            recs.append("📦 Apply balanced NPK fertilizer to improve yield potential")
        if not irrigation:
            recs.append("💧 Ensure adequate irrigation especially during flowering stage")
        if rainfall > 80:
            recs.append("🌧️ Excess rainfall – ensure good drainage to prevent root rot")

        if not recs:
            recs.append("✅ Crop conditions are good – maintain current practices")

        return recs

    def weekly_update(self, previous_prediction: dict, new_ndvi: float,
                      new_disease_loss: float = None) -> dict:
        """Update yield prediction with new NDVI reading."""
        ndvi_change = new_ndvi - previous_prediction["crop_health"]["ndvi"]
        crop  = previous_prediction["crop"]
        acres = previous_prediction["field_area_acres"]
        loss  = new_disease_loss or previous_prediction["crop_health"]["disease_impact_pct"]

        return self.predict(
            crop=crop,
            field_area_acres=acres,
            ndvi_current=new_ndvi,
            ndvi_trend=ndvi_change,
            disease_yield_loss_pct=loss,
        )


# Singleton
_predictor_instance: YieldPredictor | None = None

def get_yield_predictor() -> YieldPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = YieldPredictor()
    return _predictor_instance
