"""
CropGuard AI - Farm Knowledge Memory (Phase 5)
The AI's long-term memory for each farm.

Stores and recalls:
  - Historical disease outbreaks
  - Pesticide usage history
  - Fertilizer applications
  - Irrigation events
  - Yield records
  - Weather events
  - Crop rotation history

Provides:
  - Pattern recognition ("Blast appeared in August last year too")
  - Seasonal risk predictions
  - Agronomic recommendations based on farm-specific history
"""
import os
import json
from datetime import datetime, timedelta
from pathlib import Path


MEMORY_DIR = os.environ.get("FARM_MEMORY_DIR", "farm_memory")


class FarmMemory:
    """
    Persistent farm knowledge base.
    Stores data as JSON files per farm (upgradeable to PostgreSQL).
    """

    def __init__(self, storage_dir: str = MEMORY_DIR):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        print(f"✅ Farm Memory initialized → {self.storage_dir}")

    def _farm_file(self, farm_id: str) -> Path:
        return self.storage_dir / f"farm_{farm_id}.json"

    def _load(self, farm_id: str) -> dict:
        path = self._farm_file(farm_id)
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return {
            "farm_id":       farm_id,
            "disease_history":   [],
            "spray_history":     [],
            "fertilizer_history":[],
            "irrigation_log":    [],
            "yield_records":     [],
            "weather_events":    [],
            "crop_rotation":     [],
            "soil_readings":     [],
        }

    def _save(self, farm_id: str, data: dict):
        path = self._farm_file(farm_id)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ── Record events ──────────────────────────────────────────────────────────

    def record_disease(self, farm_id: str, crop: str, disease: str,
                       severity: str, affected_area_pct: float = 0,
                       lat: float = None, lon: float = None):
        """Record a disease detection event."""
        mem = self._load(farm_id)
        mem["disease_history"].append({
            "timestamp":       datetime.now().isoformat(),
            "month":           datetime.now().month,
            "crop":            crop,
            "disease":         disease,
            "severity":        severity,
            "affected_area_pct": affected_area_pct,
            "lat":             lat,
            "lon":             lon,
        })
        self._save(farm_id, mem)

    def record_spray(self, farm_id: str, chemical: str, dose: str,
                     area_acres: float, reason: str = "", cost_inr: float = 0):
        """Record a pesticide/herbicide spray event."""
        mem = self._load(farm_id)
        mem["spray_history"].append({
            "timestamp":  datetime.now().isoformat(),
            "month":      datetime.now().month,
            "chemical":   chemical,
            "dose":       dose,
            "area_acres": area_acres,
            "reason":     reason,
            "cost_inr":   cost_inr,
        })
        self._save(farm_id, mem)

    def record_yield(self, farm_id: str, crop: str, season: str,
                     yield_tonnes: float, area_acres: float,
                     market_value_inr: float = 0):
        """Record harvest yield for a season."""
        mem = self._load(farm_id)
        mem["yield_records"].append({
            "timestamp":       datetime.now().isoformat(),
            "year":            datetime.now().year,
            "season":          season,    # Kharif/Rabi/Zaid
            "crop":            crop,
            "yield_tonnes":    yield_tonnes,
            "yield_per_acre":  round(yield_tonnes / max(area_acres, 0.1), 2),
            "area_acres":      area_acres,
            "market_value_inr": market_value_inr,
        })
        self._save(farm_id, mem)

    def record_fertilizer(self, farm_id: str, fertilizer: str, dose_kg_acre: float,
                           crop: str = "", stage: str = ""):
        mem = self._load(farm_id)
        mem["fertilizer_history"].append({
            "timestamp":    datetime.now().isoformat(),
            "fertilizer":   fertilizer,
            "dose_kg_acre": dose_kg_acre,
            "crop":         crop,
            "stage":        stage,
        })
        self._save(farm_id, mem)

    def record_irrigation(self, farm_id: str, method: str, duration_hrs: float,
                           water_mm: float = 0, reason: str = ""):
        mem = self._load(farm_id)
        mem["irrigation_log"].append({
            "timestamp":    datetime.now().isoformat(),
            "method":       method,       # drip/flood/sprinkler
            "duration_hrs": duration_hrs,
            "water_mm":     water_mm,
            "reason":       reason,
        })
        self._save(farm_id, mem)

    def record_soil_reading(self, farm_id: str, sensor_id: str, readings: dict):
        mem = self._load(farm_id)
        mem["soil_readings"].append({
            "timestamp":  datetime.now().isoformat(),
            "sensor_id":  sensor_id,
            **readings,
        })
        self._save(farm_id, mem)

    # ── Intelligence ───────────────────────────────────────────────────────────

    def get_insights(self, farm_id: str, crop: str = None) -> dict:
        """
        Generate AI insights from farm history.

        Returns patterns, predictions, and recommendations.
        """
        mem = self._load(farm_id)
        insights = []
        warnings = []

        # ── Disease pattern analysis ───────────────────────────────────────────
        disease_hist = mem.get("disease_history", [])
        current_month = datetime.now().month

        if disease_hist:
            # Same month last year
            same_month = [
                d for d in disease_hist
                if d.get("month") == current_month
                and d.get("crop", "").lower() == (crop or "").lower()
            ]
            if same_month:
                diseases_this_month = list(set(d["disease"] for d in same_month))
                warnings.append({
                    "type":    "HISTORICAL_RISK",
                    "message": (f"⚠️ In the same month last year, {', '.join(diseases_this_month)} "
                                f"was detected on {crop or 'this field'}. High recurrence risk."),
                    "action":  "Apply preventive fungicide/bactericide now.",
                })

            # Most frequent disease
            from collections import Counter
            freq = Counter(d["disease"] for d in disease_hist)
            top_disease, top_count = freq.most_common(1)[0]
            insights.append(f"📊 Most frequent disease: {top_disease} ({top_count} times recorded)")

        # ── Yield trend analysis ────────────────────────────────────────────────
        yield_records = mem.get("yield_records", [])
        if len(yield_records) >= 2:
            recent = sorted(yield_records, key=lambda x: x["timestamp"])
            yields = [r["yield_per_acre"] for r in recent[-4:]]  # last 4 seasons
            trend  = yields[-1] - yields[0]
            if trend < -0.2:
                insights.append(
                    f"📉 Yield declining: {yields[0]:.1f} → {yields[-1]:.1f} tonnes/acre. "
                    f"Consider soil testing and crop rotation."
                )
            elif trend > 0.2:
                insights.append(
                    f"📈 Yield improving: {yields[0]:.1f} → {yields[-1]:.1f} tonnes/acre. "
                    f"Current practices are working well."
                )

        # ── Spray chemical usage ────────────────────────────────────────────────
        spray_hist = mem.get("spray_history", [])
        if spray_hist:
            total_cost = sum(s.get("cost_inr", 0) for s in spray_hist)
            insights.append(f"💊 Total pesticide cost recorded: ₹{total_cost:,.0f}")

            # Check for same chemical overuse (resistance risk)
            chem_counts = {}
            for s in spray_hist[-10:]:   # last 10 sprays
                c = s.get("chemical", "")
                chem_counts[c] = chem_counts.get(c, 0) + 1
            overused = [c for c, n in chem_counts.items() if n > 3]
            if overused:
                warnings.append({
                    "type":    "RESISTANCE_RISK",
                    "message": (f"🔄 {', '.join(overused)} used repeatedly – "
                                f"rotate chemicals to prevent pesticide resistance."),
                    "action":  "Switch to a different chemical class this season.",
                })

        # ── Crop rotation recommendations ──────────────────────────────────────
        rotation = mem.get("crop_rotation", [])
        if rotation and crop:
            last_crop = rotation[-1].get("crop", "") if rotation else ""
            if last_crop.lower() == crop.lower():
                warnings.append({
                    "type":    "ROTATION_ALERT",
                    "message": (f"🔁 {crop} was grown in the previous season too. "
                                f"Continuous cropping increases soil-borne disease risk."),
                    "action":  "Consider rotating with a legume (e.g., soybean, groundnut).",
                })

        return {
            "success":       True,
            "farm_id":       farm_id,
            "data_points":   {
                "disease_events":     len(disease_hist),
                "spray_events":       len(spray_hist),
                "yield_records":      len(yield_records),
                "soil_readings":      len(mem.get("soil_readings", [])),
                "irrigation_events":  len(mem.get("irrigation_log", [])),
            },
            "insights":      insights,
            "warnings":      warnings,
            "memory_quality": ("Rich" if len(disease_hist) > 10 else
                               "Growing" if len(disease_hist) > 3 else
                               "Early – keep adding scan data"),
        }

    def get_history_summary(self, farm_id: str, months: int = 6) -> dict:
        """Get summary of last N months of farm activity."""
        mem = self._load(farm_id)
        cutoff = (datetime.now() - timedelta(days=months * 30)).isoformat()

        def filter_recent(records):
            return [r for r in records if r.get("timestamp", "") >= cutoff]

        return {
            "farm_id":          farm_id,
            "period_months":    months,
            "disease_events":   filter_recent(mem.get("disease_history", [])),
            "spray_events":     filter_recent(mem.get("spray_history", [])),
            "yield_records":    filter_recent(mem.get("yield_records", [])),
            "irrigation_events": filter_recent(mem.get("irrigation_log", [])),
        }


# Singleton (per process)
_memory_instance: FarmMemory | None = None

def get_farm_memory() -> FarmMemory:
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = FarmMemory()
    return _memory_instance
