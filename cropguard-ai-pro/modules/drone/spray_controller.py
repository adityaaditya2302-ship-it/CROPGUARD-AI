"""
CropGuard AI - Spray Controller (Phase 3)
Converts AI disease detection results into precision spray commands.
Controls the sprayer pump via drone actuator channels.
"""
import math
import asyncio
from datetime import datetime


class SprayController:
    """
    Converts disease GPS coordinates into spray missions.
    Works with MAVLinkDroneBridge to execute precision spraying.
    """

    def __init__(self, pump_channel: int = 9, flow_rate_lph: float = 2.0):
        """
        Args:
            pump_channel:   MAVLink servo channel for pump (default 9)
            flow_rate_lph:  pump flow rate in liters per hour
        """
        self.pump_channel   = pump_channel
        self.flow_rate_lph  = flow_rate_lph
        self.total_sprayed_l = 0.0
        self.spray_log = []
        print(f"✅ Spray Controller initialized (channel {pump_channel}, {flow_rate_lph}L/hr)")

    def calculate_dose(self, severity: str, disease_name: str) -> dict:
        """
        Calculate required spray dose based on severity.

        Returns:
            spray_duration_s, chemical_volume_ml, concentration_note
        """
        DOSE_TABLE = {
            "Critical": {"duration_s": 8.0,  "conc_note": "Full label dose"},
            "High":     {"duration_s": 5.0,  "conc_note": "Full label dose"},
            "Medium":   {"duration_s": 3.0,  "conc_note": "Standard dose"},
            "Low":      {"duration_s": 2.0,  "conc_note": "Half dose preventive"},
        }
        dose = DOSE_TABLE.get(severity, {"duration_s": 3.0, "conc_note": "Standard dose"})
        volume_ml = (self.flow_rate_lph / 3600) * dose["duration_s"] * 1000

        return {
            "spray_duration_s":   dose["duration_s"],
            "chemical_volume_ml": round(volume_ml, 1),
            "concentration_note": dose["conc_note"],
        }

    async def execute_spray_at_point(self, bridge, lat: float, lon: float,
                                      severity: str, disease: str = "") -> dict:
        """
        Navigate to GPS point and spray. Requires active drone bridge.
        """
        dose = self.calculate_dose(severity, disease)
        print(f"💊 Spraying at ({lat:.5f}, {lon:.5f}) | {disease} | {severity} | {dose['spray_duration_s']}s")

        await bridge.activate_sprayer(duration_seconds=dose["spray_duration_s"])

        log_entry = {
            "timestamp":    datetime.utcnow().isoformat(),
            "lat":          lat,
            "lon":          lon,
            "disease":      disease,
            "severity":     severity,
            "volume_ml":    dose["chemical_volume_ml"],
            "duration_s":   dose["spray_duration_s"],
        }
        self.spray_log.append(log_entry)
        self.total_sprayed_l += dose["chemical_volume_ml"] / 1000

        return {"success": True, **log_entry}

    def get_spray_summary(self) -> dict:
        return {
            "total_points_sprayed": len(self.spray_log),
            "total_chemical_liters": round(self.total_sprayed_l, 3),
            "spray_log":            self.spray_log,
        }

    def generate_spray_report(self, field_area_ha: float) -> dict:
        """Compare precision vs blanket spraying."""
        precision_liters = self.total_sprayed_l
        blanket_liters   = field_area_ha * 200   # typical 200L/ha blanket

        savings_liters = max(0, blanket_liters - precision_liters)
        savings_pct    = savings_liters / max(blanket_liters, 1) * 100

        return {
            "precision_liters_used":   round(precision_liters, 2),
            "blanket_liters_would_use": round(blanket_liters, 1),
            "chemical_savings_liters": round(savings_liters, 1),
            "chemical_savings_pct":    round(savings_pct, 1),
            "cost_savings_inr":        round(savings_liters * 150, 0),   # ₹150/L typical
            "environmental_benefit":   "Reduced chemical runoff and soil contamination",
        }


_spray_instance: SprayController | None = None

def get_spray_controller() -> SprayController:
    global _spray_instance
    if _spray_instance is None:
        _spray_instance = SprayController()
    return _spray_instance
