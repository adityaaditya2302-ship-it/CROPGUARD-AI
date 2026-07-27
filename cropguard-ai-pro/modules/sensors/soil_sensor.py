"""
CropGuard AI - IoT Soil Sensor Module (Phase 3)
Reads soil data from IoT sensors via MQTT protocol.

Sensor types supported:
  - Soil moisture (capacitive)
  - Soil pH
  - EC (Electrical Conductivity)
  - NPK (Nitrogen, Phosphorus, Potassium)
  - Soil temperature
"""
import os
import json
from datetime import datetime


MQTT_BROKER = os.environ.get("MQTT_BROKER", "broker.hivemq.com")
MQTT_PORT   = int(os.environ.get("MQTT_PORT", 1883))
MQTT_TOPIC_SOIL = "cropguard/sensors/soil/#"


class SoilSensorHub:
    """
    MQTT-based soil sensor data collector.
    Connects to HiveMQ or local MQTT broker.
    Falls back to mock data for development.
    """

    def __init__(self):
        self.client = None
        self.sensor_data = {}  # sensor_id → latest reading
        self.mqtt_available = self._check_paho()
        print(f"✅ Soil Sensor Hub initialized (MQTT: {'ready' if self.mqtt_available else 'offline/mock'})")

    def _check_paho(self) -> bool:
        try:
            import paho.mqtt.client as mqtt
            return True
        except ImportError:
            print("⚠️  paho-mqtt not installed. Install with: pip install paho-mqtt")
            return False

    def connect(self, broker: str = MQTT_BROKER, port: int = MQTT_PORT) -> bool:
        if not self.mqtt_available:
            print("ℹ️  Soil sensor running in MOCK mode.")
            return True
        try:
            import paho.mqtt.client as mqtt
            self.client = mqtt.Client(client_id="cropguard-soil-hub")
            self.client.on_message = self._on_message
            self.client.connect(broker, port, keepalive=60)
            self.client.subscribe(MQTT_TOPIC_SOIL)
            self.client.loop_start()
            print(f"✅ Connected to MQTT broker: {broker}:{port}")
            return True
        except Exception as e:
            print(f"⚠️  MQTT connection failed: {e}. Using mock data.")
            return False

    def _on_message(self, client, userdata, msg):
        """Parse incoming MQTT soil sensor message."""
        try:
            payload = json.loads(msg.payload.decode())
            sensor_id = payload.get("sensor_id", msg.topic.split("/")[-1])
            self.sensor_data[sensor_id] = {
                **payload,
                "timestamp": datetime.utcnow().isoformat(),
                "topic":     msg.topic,
            }
        except Exception as e:
            print(f"⚠️  MQTT message parse error: {e}")

    def get_readings(self, sensor_id: str = None) -> dict:
        """
        Get latest soil readings.

        Args:
            sensor_id: specific sensor ID, or None for all sensors

        Returns:
            dict with readings + interpretations
        """
        if sensor_id:
            raw = self.sensor_data.get(sensor_id) or self._mock_reading(sensor_id)
        else:
            if self.sensor_data:
                raw = list(self.sensor_data.values())
            else:
                raw = [self._mock_reading(f"SENSOR_{i}") for i in range(1, 4)]

        if isinstance(raw, list):
            return {
                "success":  True,
                "sensors":  [self._interpret_reading(r) for r in raw],
                "count":    len(raw),
                "timestamp": datetime.utcnow().isoformat(),
            }
        return {"success": True, "sensor": self._interpret_reading(raw)}

    def _interpret_reading(self, reading: dict) -> dict:
        """Add agronomic interpretation to raw sensor values."""
        result = dict(reading)

        # Soil moisture interpretation
        moisture = reading.get("moisture_pct", 50)
        result["moisture_status"] = (
            "Critical - Immediate irrigation needed" if moisture < 20 else
            "Low - Irrigate soon"                    if moisture < 35 else
            "Optimal"                                 if moisture < 70 else
            "High - Risk of root disease"
        )
        result["irrigation_needed"] = moisture < 35

        # pH interpretation
        ph = reading.get("ph", 6.5)
        result["ph_status"] = (
            "Very acidic - Add lime"       if ph < 5.0 else
            "Acidic - Slightly adjust"     if ph < 6.0 else
            "Optimal (6.0-7.0)"            if ph < 7.0 else
            "Neutral to alkaline - OK"     if ph < 7.5 else
            "Alkaline - Add sulfur/gypsum"
        )

        # NPK interpretation
        n = reading.get("nitrogen_mgkg", 100)
        p = reading.get("phosphorus_mgkg", 30)
        k = reading.get("potassium_mgkg", 120)
        result["npk_status"] = {
            "nitrogen":   "Low - Apply urea" if n < 50 else "Adequate" if n < 200 else "High",
            "phosphorus": "Low - Apply DAP"  if p < 15 else "Adequate" if p < 60  else "High",
            "potassium":  "Low - Apply MOP"  if k < 80 else "Adequate" if k < 250 else "High",
        }
        result["fertilizer_recommendation"] = self._fertilizer_rec(n, p, k)

        return result

    def _fertilizer_rec(self, n: float, p: float, k: float) -> list:
        recs = []
        if n < 50:   recs.append("Apply Urea (46% N) @ 50kg/acre")
        if p < 15:   recs.append("Apply DAP (18% N, 46% P₂O₅) @ 50kg/acre")
        if k < 80:   recs.append("Apply MOP (60% K₂O) @ 30kg/acre")
        if not recs: recs.append("Soil nutrition is adequate - maintain with FYM @ 2 tonnes/acre")
        return recs

    def _mock_reading(self, sensor_id: str) -> dict:
        import random
        return {
            "sensor_id":         sensor_id,
            "moisture_pct":      random.randint(25, 75),
            "ph":                round(random.uniform(5.5, 7.5), 1),
            "ec_dscm":           round(random.uniform(0.3, 2.0), 2),
            "temperature_c":     round(random.uniform(20, 35), 1),
            "nitrogen_mgkg":     random.randint(30, 200),
            "phosphorus_mgkg":   random.randint(10, 60),
            "potassium_mgkg":    random.randint(60, 250),
            "timestamp":         datetime.utcnow().isoformat(),
            "source":            "mock",
        }

    def disconnect(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()


# Singleton
_soil_hub_instance: SoilSensorHub | None = None

def get_soil_hub() -> SoilSensorHub:
    global _soil_hub_instance
    if _soil_hub_instance is None:
        _soil_hub_instance = SoilSensorHub()
        _soil_hub_instance.connect()
    return _soil_hub_instance
