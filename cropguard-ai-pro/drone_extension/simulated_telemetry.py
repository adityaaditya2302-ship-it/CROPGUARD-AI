"""
Phase 1-3: Simulated telemetry.

This produces realistic-looking, self-consistent telemetry for a
"connected" drone so the whole dashboard/UI/mission flow can be built
and demoed before any real hardware is involved. It is intentionally
deterministic-ish (small random walk) rather than pure noise, so
values look like a real flight rather than jitter.

Swap this for `mavlink_adapter.py` (Phase 4) or `dji_adapter.py`
(Phase 5) later - both expose the same `get_telemetry(device_id)`
shape, so routes_drone.py does not need to change.
"""
import math
import random
import time

_state = {}  # device_id -> dict of current simulated values


def _init_state(device_id):
    base_lat = 28.6139 + random.uniform(-0.01, 0.01)
    base_lon = 77.2090 + random.uniform(-0.01, 0.01)
    _state[device_id] = {
        'lat': base_lat,
        'lon': base_lon,
        'altitude_m': 0.0,
        'speed_mps': 0.0,
        'heading_deg': random.uniform(0, 360),
        'satellites': random.randint(11, 18),
        'battery_pct': 100.0,
        'signal_strength_pct': random.uniform(85, 100),
        'flight_mode': 'IDLE',
        'tank_pct': 100.0,
        'flow_rate_lpm': 0.0,
        'connected_at': time.time(),
        'mission_active': False,
    }


def connect(device_id):
    if device_id not in _state:
        _init_state(device_id)
    _state[device_id]['flight_mode'] = 'IDLE'
    return _state[device_id]


def disconnect(device_id):
    _state.pop(device_id, None)


def set_mission_active(device_id, active, speed_mps=4.0):
    if device_id not in _state:
        _init_state(device_id)
    _state[device_id]['mission_active'] = active
    _state[device_id]['flight_mode'] = 'AUTO' if active else 'HOVER'
    _state[device_id]['speed_mps'] = speed_mps if active else 0.0
    if active:
        _state[device_id]['altitude_m'] = max(_state[device_id]['altitude_m'], 3.0)


def get_telemetry(device_id):
    """Returns a plain dict (not a DB row) - matches the shape expected
    by DroneHubTelemetryLog.to_dict() minus id/timestamp."""
    if device_id not in _state:
        _init_state(device_id)
    s = _state[device_id]

    if s['mission_active']:
        s['lat'] += math.cos(math.radians(s['heading_deg'])) * 0.00003
        s['lon'] += math.sin(math.radians(s['heading_deg'])) * 0.00003
        s['heading_deg'] = (s['heading_deg'] + random.uniform(-3, 3)) % 360
        s['battery_pct'] = max(0.0, s['battery_pct'] - random.uniform(0.05, 0.15))
        s['tank_pct'] = max(0.0, s['tank_pct'] - random.uniform(0.1, 0.3))
        s['flow_rate_lpm'] = round(random.uniform(1.8, 2.4), 2) if s['tank_pct'] > 0 else 0.0
    else:
        s['battery_pct'] = max(0.0, s['battery_pct'] - random.uniform(0.0, 0.02))
        s['flow_rate_lpm'] = 0.0

    s['signal_strength_pct'] = max(40.0, min(100.0, s['signal_strength_pct'] + random.uniform(-1, 1)))

    return {
        'device_id': device_id,
        'latitude': round(s['lat'], 6),
        'longitude': round(s['lon'], 6),
        'altitude_m': round(s['altitude_m'], 1),
        'speed_mps': round(s['speed_mps'], 2),
        'heading_deg': round(s['heading_deg'], 1),
        'satellites': s['satellites'],
        'battery_pct': round(s['battery_pct'], 1),
        'signal_strength_pct': round(s['signal_strength_pct'], 1),
        'flight_mode': s['flight_mode'],
        'tank_pct': round(s['tank_pct'], 1),
        'flow_rate_lpm': s['flow_rate_lpm'],
    }


def health_check(device_id):
    if device_id not in _state:
        _init_state(device_id)
    s = _state[device_id]
    warnings = []
    if s['battery_pct'] < 20:
        warnings.append('Low battery')
    if s['signal_strength_pct'] < 50:
        warnings.append('Weak signal')
    if s['tank_pct'] < 10 and s['mission_active']:
        warnings.append('Spray tank low')
    return {
        'motor_health': 'OK',
        'esc_status': 'OK',
        'battery_health': 'OK' if s['battery_pct'] > 20 else 'WARNING',
        'compass': 'OK',
        'imu': 'OK',
        'gps_quality': 'GOOD' if s['satellites'] >= 10 else 'FAIR',
        'warnings': warnings,
        'safe_to_fly': len(warnings) == 0,
    }
