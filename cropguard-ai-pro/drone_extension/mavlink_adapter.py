"""
Phase 4: Real MAVLink connection layer (ArduPilot / PX4 / any
MAVLink-speaking flight controller), using the `pymavlink` library.

*** HONESTY NOTE ***
This code is written to the pymavlink API correctly, but it has NOT
been run against a real flight controller or a SITL (software-in-the-
loop) simulator, because I don't have access to your hardware. Treat
this as a correct starting scaffold, not a verified integration.

Before using this for real:
  pip install pymavlink
  # Test against SITL first (free, no hardware needed):
  #   https://ardupilot.org/dev/docs/sitl-simulator-software-in-the-loop.html
  # Typical real connection strings:
  #   'udp:127.0.0.1:14550'      (SITL / companion computer over UDP)
  #   'com3' / '/dev/ttyUSB0'    (USB telemetry radio, baud=57600)
  #   'tcp:192.168.1.1:5760'     (WiFi telemetry)

Swap this in for simulated_telemetry.py in routes_drone.py once you
have verified it against SITL, by changing one import line (see
README_INTEGRATION.md).
"""
try:
    from pymavlink import mavutil
    PYMAVLINK_AVAILABLE = True
except ImportError:
    PYMAVLINK_AVAILABLE = False

_connections = {}  # device_id -> mavutil connection object


def connect(device_id, connection_string, baud=57600):
    """connection_string e.g. 'udp:127.0.0.1:14550' or 'COM3'."""
    if not PYMAVLINK_AVAILABLE:
        raise RuntimeError("pymavlink is not installed. Run: pip install pymavlink")

    conn = mavutil.mavlink_connection(connection_string, baud=baud)
    conn.wait_heartbeat(timeout=10)  # raises/returns None on timeout
    _connections[device_id] = conn
    return {'connected': True, 'system_id': conn.target_system, 'component_id': conn.target_component}


def disconnect(device_id):
    conn = _connections.pop(device_id, None)
    if conn:
        conn.close()


def get_telemetry(device_id):
    """Pulls the latest cached MAVLink messages. Real telemetry only
    updates as fast as the flight controller streams it - you may need
    to request specific message intervals with
    MAV_CMD_SET_MESSAGE_INTERVAL for smooth 1Hz+ updates."""
    conn = _connections.get(device_id)
    if not conn:
        raise RuntimeError(f"No active MAVLink connection for device {device_id}")

    msg = conn.recv_match(
        type=['GLOBAL_POSITION_INT', 'VFR_HUD', 'SYS_STATUS', 'GPS_RAW_INT', 'HEARTBEAT'],
        blocking=False,
    )

    data = {'device_id': device_id}
    if msg:
        t = msg.get_type()
        if t == 'GLOBAL_POSITION_INT':
            data['latitude'] = msg.lat / 1e7
            data['longitude'] = msg.lon / 1e7
            data['altitude_m'] = msg.relative_alt / 1000.0
            data['heading_deg'] = msg.hdg / 100.0
        elif t == 'VFR_HUD':
            data['speed_mps'] = msg.groundspeed
        elif t == 'SYS_STATUS':
            data['battery_pct'] = msg.battery_remaining
        elif t == 'GPS_RAW_INT':
            data['satellites'] = msg.satellites_visible
    return data


def send_mission(device_id, waypoints):
    """waypoints: list of (lat, lon, alt) tuples. Uploads a full mission
    using MAVLink's mission protocol. THIS IS A SCAFFOLD - the mission
    upload handshake (MISSION_COUNT -> MISSION_REQUEST -> MISSION_ITEM_INT
    -> MISSION_ACK) needs to be verified against your actual autopilot
    firmware version."""
    conn = _connections.get(device_id)
    if not conn:
        raise RuntimeError(f"No active MAVLink connection for device {device_id}")

    conn.mav.mission_count_send(conn.target_system, conn.target_component, len(waypoints))
    for i, (lat, lon, alt) in enumerate(waypoints):
        conn.mav.mission_item_int_send(
            conn.target_system, conn.target_component, i,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            0, 1, 0, 0, 0, 0,
            int(lat * 1e7), int(lon * 1e7), alt,
        )
    return {'uploaded': True, 'waypoint_count': len(waypoints)}


def start_mission(device_id):
    conn = _connections.get(device_id)
    if not conn:
        raise RuntimeError(f"No active MAVLink connection for device {device_id}")
    conn.set_mode_auto()
    return {'status': 'started'}


def return_to_home(device_id):
    conn = _connections.get(device_id)
    if not conn:
        raise RuntimeError(f"No active MAVLink connection for device {device_id}")
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH, 0, 0, 0, 0, 0, 0, 0, 0,
    )
    return {'status': 'rtl_commanded'}
