"""
CropGuard AI - Weather Intelligence Module (Phase 2)
Fetches 7-day forecast and interprets it for farming decisions.

Features:
  - Disease risk from weather patterns
  - Spray timing advisor
  - Irrigation recommendations
  - Storm/flood alerts
  - Frost warnings
"""
import os
import json
from datetime import datetime, timedelta


# Open-Meteo API (free, no API key needed)
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"

# OpenWeatherMap (requires API key for extras)
OWM_BASE = "https://api.openweathermap.org/data/2.5"


class WeatherIntelligence:
    """
    Fetches and interprets weather data for agricultural decisions.
    Primary: Open-Meteo (free, no key needed)
    Fallback: Mock data for offline development
    """

    def __init__(self):
        self.owm_key = os.environ.get("OPENWEATHERMAP_API_KEY", "")
        print("✅ Weather Intelligence module initialized (Open-Meteo free API)")

    def get_forecast(self, lat: float, lon: float, days: int = 7) -> dict:
        """
        Fetch weather forecast for farm location.

        Args:
            lat:  latitude
            lon:  longitude
            days: forecast days (1-16)

        Returns:
            dict with daily forecasts + farming interpretations
        """
        try:
            import urllib.request
            url = (
                f"{OPEN_METEO_BASE}?latitude={lat}&longitude={lon}"
                f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
                f"windspeed_10m_max,relative_humidity_2m_max,relative_humidity_2m_min,"
                f"weathercode,sunrise,sunset"
                f"&timezone=Asia%2FKolkata&forecast_days={days}"
            )
            with urllib.request.urlopen(url, timeout=10) as resp:
                raw = json.loads(resp.read().decode())
            return self._parse_open_meteo(raw)
        except Exception as e:
            print(f"⚠️  Weather API failed: {e}. Using mock data.")
            return self._mock_forecast(lat, lon, days)

    def get_current_weather(self, lat: float, lon: float) -> dict:
        """Get current weather conditions."""
        try:
            import urllib.request
            url = (
                f"{OPEN_METEO_BASE}?latitude={lat}&longitude={lon}"
                f"&current_weather=true"
                f"&hourly=relativehumidity_2m,precipitation"
                f"&timezone=Asia%2FKolkata&forecast_days=1"
            )
            with urllib.request.urlopen(url, timeout=10) as resp:
                raw = json.loads(resp.read().decode())
            current = raw.get("current_weather", {})
            return {
                "temp_c":     current.get("temperature", 25),
                "wind_kmh":   current.get("windspeed", 10),
                "is_day":     current.get("is_day", 1),
                "weathercode": current.get("weathercode", 0),
                "description": self._decode_wmo(current.get("weathercode", 0)),
                "humidity_avg": self._get_current_humidity(raw),
            }
        except Exception as e:
            print(f"⚠️  Current weather failed: {e}")
            return {"temp_c": 25, "wind_kmh": 10, "humidity_avg": 60, "description": "Data unavailable"}

    # ── parsing ────────────────────────────────────────────────────────────────

    def _parse_open_meteo(self, raw: dict) -> dict:
        daily = raw.get("daily", {})
        dates = daily.get("time", [])
        n     = len(dates)

        forecast_days = []
        for i in range(n):
            hum_max = daily.get("relative_humidity_2m_max", [70]*n)[i] or 70
            hum_min = daily.get("relative_humidity_2m_min", [50]*n)[i] or 50
            hum_avg = (hum_max + hum_min) / 2

            day = {
                "date":          dates[i],
                "temp_max":      daily.get("temperature_2m_max", [25]*n)[i],
                "temp_min":      daily.get("temperature_2m_min", [15]*n)[i],
                "rainfall_mm":   daily.get("precipitation_sum", [0]*n)[i] or 0,
                "wind_kmh":      daily.get("windspeed_10m_max", [10]*n)[i] or 10,
                "humidity_avg":  round(hum_avg, 1),
                "humidity_max":  hum_max,
                "weathercode":   daily.get("weathercode", [0]*n)[i],
                "description":   self._decode_wmo(daily.get("weathercode", [0]*n)[i]),
                "sunrise":       daily.get("sunrise", ["06:00"]*n)[i],
                "sunset":        daily.get("sunset", ["18:30"]*n)[i],
                "rain_day":      (daily.get("precipitation_sum", [0]*n)[i] or 0) > 0.5,
            }
            day["farming_advice"] = self._interpret_day(day)
            forecast_days.append(day)

        spray_window = self._find_best_spray_window(forecast_days)
        summary      = self._weather_summary(forecast_days)

        return {
            "success":       True,
            "source":        "open_meteo",
            "forecast_days": forecast_days,
            "spray_window":  spray_window,
            "summary":       summary,
            "disease_pressure": self._assess_disease_pressure(forecast_days),
        }

    def _mock_forecast(self, lat: float, lon: float, days: int) -> dict:
        """Generate realistic mock data for testing without internet."""
        import random
        from datetime import date

        forecast_days = []
        for i in range(days):
            d = date.today() + timedelta(days=i)
            hum_avg = random.randint(55, 85)
            temp_max = random.randint(28, 36)
            temp_min = random.randint(18, 25)
            rainfall = random.choice([0, 0, 0, 2, 8, 15])

            day = {
                "date":         str(d),
                "temp_max":     temp_max,
                "temp_min":     temp_min,
                "rainfall_mm":  rainfall,
                "wind_kmh":     random.randint(5, 30),
                "humidity_avg": hum_avg,
                "humidity_max": hum_avg + 10,
                "weathercode":  61 if rainfall > 0 else 1,
                "description":  "Light rain" if rainfall > 0 else "Partly cloudy",
                "sunrise":      "06:15",
                "sunset":       "18:45",
                "rain_day":     rainfall > 0,
            }
            day["farming_advice"] = self._interpret_day(day)
            forecast_days.append(day)

        return {
            "success":          True,
            "source":           "mock_offline",
            "note":             "Using mock data (no internet or API key)",
            "forecast_days":    forecast_days,
            "spray_window":     self._find_best_spray_window(forecast_days),
            "summary":          self._weather_summary(forecast_days),
            "disease_pressure": self._assess_disease_pressure(forecast_days),
        }

    # ── interpretation ─────────────────────────────────────────────────────────

    def _interpret_day(self, day: dict) -> dict:
        issues = []
        positives = []
        spray_ok = True

        if day["rainfall_mm"] > 0.5:
            issues.append("Rain expected – avoid spraying")
            spray_ok = False
        if day["wind_kmh"] > 25:
            issues.append(f"High wind {day['wind_kmh']} km/h – spray drift risk")
            spray_ok = False
        if day["temp_max"] > 35:
            issues.append(f"High temperature {day['temp_max']}°C – spray early morning")
        if day["humidity_avg"] > 85:
            issues.append(f"Very high humidity {day['humidity_avg']}% – disease risk elevated")
        elif day["humidity_avg"] < 30:
            issues.append("Low humidity – extra irrigation may be needed")

        if spray_ok and day["wind_kmh"] < 15:
            positives.append("Good spray conditions")
        if not day["rain_day"] and day["humidity_avg"] < 75:
            positives.append("Low disease pressure day")

        return {
            "spray_suitable":     spray_ok,
            "best_spray_time":    "6:00–10:00 AM" if spray_ok else "Not recommended",
            "issues":             issues,
            "positives":          positives,
            "irrigation_needed":  day["rainfall_mm"] < 2 and day["temp_max"] > 30,
        }

    def _find_best_spray_window(self, forecast: list) -> dict:
        """Find the next best 2-day window for pesticide spraying."""
        best_days = []
        for day in forecast:
            if day["farming_advice"]["spray_suitable"]:
                best_days.append(day["date"])
            if len(best_days) >= 2:
                break
        return {
            "best_dates":       best_days,
            "recommendation":   (f"Best spray window: {', '.join(best_days)}"
                                 if best_days else "No suitable spray window in next 7 days – wait for calmer weather"),
        }

    def _assess_disease_pressure(self, forecast: list) -> dict:
        high_hum_days = sum(1 for d in forecast if d["humidity_avg"] > 80)
        rain_days     = sum(1 for d in forecast if d["rain_day"])
        pressure_score= min(high_hum_days * 15 + rain_days * 10, 100)

        return {
            "score":        pressure_score,
            "level":        ("High" if pressure_score > 60 else
                             "Medium" if pressure_score > 30 else "Low"),
            "high_hum_days": high_hum_days,
            "rain_days":    rain_days,
            "advice": ("High fungal disease pressure expected – apply preventive fungicide" if pressure_score > 60 else
                       "Moderate pressure – monitor plants daily" if pressure_score > 30 else
                       "Low disease pressure – routine monitoring sufficient"),
        }

    def _weather_summary(self, forecast: list) -> str:
        if not forecast:
            return "No forecast available"
        avg_temp = sum(d["temp_max"] for d in forecast) / len(forecast)
        total_rain = sum(d["rainfall_mm"] for d in forecast)
        avg_hum  = sum(d["humidity_avg"] for d in forecast) / len(forecast)
        return (f"7-day outlook: Avg temp {avg_temp:.0f}°C, "
                f"Total rainfall {total_rain:.0f}mm, Avg humidity {avg_hum:.0f}%. "
                f"{'Wet spell ahead' if total_rain > 50 else 'Mostly dry conditions'}.")

    def _get_current_humidity(self, raw: dict) -> float:
        hourly = raw.get("hourly", {})
        hum_list = hourly.get("relativehumidity_2m", [60])
        return sum(hum_list[:6]) / min(len(hum_list[:6]), 1) if hum_list else 60.0

    @staticmethod
    def _decode_wmo(code: int) -> str:
        """Decode WMO weather code to description."""
        WMO = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Icy fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
            95: "Thunderstorm", 96: "Thunderstorm + slight hail", 99: "Thunderstorm + heavy hail",
        }
        return WMO.get(code, "Unknown")


# Singleton
_weather_instance: WeatherIntelligence | None = None

def get_weather_intelligence() -> WeatherIntelligence:
    global _weather_instance
    if _weather_instance is None:
        _weather_instance = WeatherIntelligence()
    return _weather_instance
