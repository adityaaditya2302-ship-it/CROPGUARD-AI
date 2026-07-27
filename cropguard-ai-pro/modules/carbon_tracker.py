"""
CropGuard AI - Carbon Tracker (Phase 5)
Tracks carbon footprint and generates sustainability scores.

Features:
  - Carbon storage estimation per crop type
  - Pesticide reduction tracking (precision vs blanket)
  - Emission estimation (diesel drones, irrigation pumps)
  - Sustainability score (0-100)
  - Carbon credit potential report
"""
from datetime import datetime


# Carbon sequestration rates (tonnes CO₂/hectare/year)
CROP_CARBON_SEQUESTRATION = {
    "rice":       1.2,    # Paddy + straw
    "wheat":      0.9,
    "sugarcane":  8.5,    # High biomass
    "maize":      1.5,
    "soybean":    1.1,    # N-fixation bonus
    "cotton":     0.7,
    "mango":      5.2,    # Perennial tree
    "banana":     3.1,
    "grape":      2.8,
    "apple":      4.0,
    "tomato":     0.5,
    "potato":     0.4,
}

# Carbon credit market (VERRA/Gold Standard, USD/tonne CO₂ equivalent)
CARBON_CREDIT_USD_PER_TONNE = 15.0   # conservative estimate
INR_USD_RATE = 83.0


class CarbonTracker:
    """
    Estimates carbon footprint and sustainability metrics for the farm.
    """

    def __init__(self):
        self.records = []
        print("✅ Carbon Tracker initialized")

    def calculate_farm_carbon_score(
        self,
        crop: str,
        field_area_acres: float,
        precision_spray_pct: float = 70.0,       # % reduction in pesticide from precision spraying
        organic_fertilizer_pct: float = 0.0,     # % of fertilization done organically
        solar_powered_pumps: bool = False,
        drip_irrigation: bool = False,
        drone_electric: bool = True,              # electric drone vs petrol
        cover_crop: bool = False,
        crop_rotation: bool = False,
    ) -> dict:
        """
        Calculate carbon score for farming practices.

        Returns:
            Sustainability score, carbon credits potential, recommendations
        """
        crop_lower = crop.lower()
        area_ha = field_area_acres * 0.4047  # acres to hectares

        # ── Carbon sequestration ────────────────────────────────────────────────
        seq_rate = CROP_CARBON_SEQUESTRATION.get(crop_lower, 1.0)
        total_sequestration = seq_rate * area_ha

        # ── Emission calculations ───────────────────────────────────────────────
        emissions = {}

        # Chemical fertilizer emissions: ~5 kg CO₂e per kg N applied
        # Assume 100kg/ha urea (46% N) = 46kg N → 46*5 = 230 kg CO₂e/ha
        chem_fert_pct = 1 - (organic_fertilizer_pct / 100)
        emissions["fertilizer_kg_co2e"] = 230 * chem_fert_pct * area_ha

        # Pesticide emissions: ~5 kg CO₂e per kg active ingredient
        # Precision spraying reduces by precision_spray_pct%
        baseline_pesticide_co2 = 15 * area_ha
        emissions["pesticide_kg_co2e"]  = baseline_pesticide_co2 * (1 - precision_spray_pct / 100)
        emissions["pesticide_saved_kg"] = baseline_pesticide_co2 * (precision_spray_pct / 100)

        # Irrigation pump emissions
        if solar_powered_pumps:
            emissions["irrigation_kg_co2e"] = 0
        else:
            # Diesel pump: ~2.7 kg CO₂/liter diesel × ~10L/hour × 4 hours/week × 14 weeks
            emissions["irrigation_kg_co2e"] = 2.7 * 10 * 4 * 14 * (area_ha / 5)

        # Drone emissions
        emissions["drone_kg_co2e"] = 0 if drone_electric else (0.5 * area_ha)  # electric = 0

        total_emissions_kg = sum(emissions.values())
        net_carbon_kg      = total_sequestration * 1000 - total_emissions_kg
        net_carbon_tonnes  = net_carbon_kg / 1000

        # ── Sustainability Score (0-100) ────────────────────────────────────────
        score = 50  # base score

        if precision_spray_pct >= 60:  score += 15
        if organic_fertilizer_pct >= 30: score += 10
        if solar_powered_pumps:         score += 10
        if drip_irrigation:             score += 8
        if drone_electric:              score += 5
        if cover_crop:                  score += 7
        if crop_rotation:               score += 8
        if net_carbon_kg > 0:           score += 5  # net carbon sink

        score = min(100, max(0, score))
        score_label = (
            "Excellent 🟢" if score >= 80 else
            "Good 🟡"     if score >= 60 else
            "Fair 🟠"     if score >= 40 else
            "Poor 🔴"
        )

        # ── Carbon credit potential ─────────────────────────────────────────────
        eligible_credits = max(0, net_carbon_tonnes)
        credit_value_usd = eligible_credits * CARBON_CREDIT_USD_PER_TONNE
        credit_value_inr = credit_value_usd * INR_USD_RATE

        # ── Recommendations ─────────────────────────────────────────────────────
        recommendations = []
        if precision_spray_pct < 60:
            recommendations.append("🚁 Use drone precision spraying to reduce pesticide emissions by 60-80%")
        if not drip_irrigation:
            recommendations.append("💧 Switch to drip irrigation (saves 40% water + reduces pump emissions)")
        if not solar_powered_pumps:
            recommendations.append("☀️ Install solar pumps to eliminate irrigation carbon footprint")
        if organic_fertilizer_pct < 20:
            recommendations.append("🌱 Add FYM/compost (20%+ organic fertilization reduces N₂O emissions)")
        if not cover_crop:
            recommendations.append("🌾 Plant cover crops in off-season to increase carbon sequestration")
        if not recommendations:
            recommendations.append("✅ Your farm practices are highly sustainable!")

        record = {
            "timestamp":            datetime.now().isoformat(),
            "crop":                 crop,
            "area_ha":              round(area_ha, 2),
            "carbon_sequestered_t": round(total_sequestration, 2),
            "total_emissions_kg":   round(total_emissions_kg, 1),
            "net_carbon_t":         round(net_carbon_tonnes, 2),
            "sustainability_score": score,
            "score_label":          score_label,
            "emissions_breakdown":  {k: round(v, 1) for k, v in emissions.items()},
            "carbon_credits": {
                "eligible_tonnes":  round(eligible_credits, 2),
                "value_usd":        round(credit_value_usd, 2),
                "value_inr":        round(credit_value_inr, 2),
                "registry":         "VERRA / Gold Standard (consult carbon broker)",
            },
            "recommendations":      recommendations,
        }
        self.records.append(record)
        return {"success": True, **record}

    def get_carbon_history(self) -> list:
        return self.records


# Singleton
_carbon_instance: CarbonTracker | None = None

def get_carbon_tracker() -> CarbonTracker:
    global _carbon_instance
    if _carbon_instance is None:
        _carbon_instance = CarbonTracker()
    return _carbon_instance
