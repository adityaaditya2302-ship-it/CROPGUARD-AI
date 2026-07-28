"""
Driver registry + auto-detection.

Real auto-detection is only implemented for MAVLink, because it's an
open protocol: any MAVLink device broadcasts a HEARTBEAT message that
we can listen for on common ports/serial devices without needing
manufacturer-specific pairing software. DJI/ROS/custom hardware each
have their own (often app-based) pairing flow that can't be
generically scanned for over a network - see each driver's docstring.

Usage:
    from drone_extension.drivers.driver_manager import get_driver, discover_mavlink

    driver = get_driver('mavlink')
    driver.connect('udp:127.0.0.1:14550')
"""
import glob

from .base import DroneDriverError
from .mavlink_driver import MAVLinkDriver, PYMAVLINK_AVAILABLE
from .dji_driver import DJIDriver
from .ros_driver import ROSDriver
from .custom_driver import CustomDriver

_DRIVERS = {
    'mavlink': MAVLinkDriver,
    'dji': DJIDriver,
    'ros': ROSDriver,
    'custom': CustomDriver,
}

# One live driver instance per logical device_id, so routes can look
# up "the driver connected as device 3" across requests.
_active = {}


def get_driver(driver_type: str):
    cls = _DRIVERS.get(driver_type)
    if not cls:
        raise DroneDriverError(f"Unknown driver type '{driver_type}'. Known: {list(_DRIVERS)}")
    return cls()


def connect_device(device_id, driver_type: str, connection_string: str, **kwargs) -> dict:
    driver = get_driver(driver_type)
    result = driver.connect(connection_string, **kwargs)
    _active[device_id] = driver
    return result


def get_active_driver(device_id):
    driver = _active.get(device_id)
    if not driver:
        raise DroneDriverError(f"No active real-drone connection for device {device_id}")
    return driver


def disconnect_device(device_id):
    driver = _active.pop(device_id, None)
    if driver:
        driver.disconnect()


def discover_mavlink(candidate_ports=None, udp_ports=(14550, 14540), serial_timeout=2):
    """Best-effort scan for MAVLink devices: tries common serial ports
    plus common UDP ports (14550 = typical GCS port, 14540 = typical
    SITL/companion-computer port). Returns connection strings that
    responded with a heartbeat - each is safe to hand to connect().

    This is a real scan (not simulated), but serial port names vary a
    lot by OS - extend candidate_ports for your setup if nothing is
    found (e.g. add '/dev/ttyACM0' on Linux, 'COM5' on Windows)."""
    if not PYMAVLINK_AVAILABLE:
        raise DroneDriverError("pymavlink is not installed. Run: pip install pymavlink")
    from pymavlink import mavutil

    if candidate_ports is None:
        candidate_ports = (
            glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
            + [f'COM{i}' for i in range(1, 21)]
        )

    found = []
    for port in candidate_ports:
        try:
            conn = mavutil.mavlink_connection(port, baud=57600)
            if conn.wait_heartbeat(timeout=serial_timeout):
                found.append({'connection_string': port, 'type': 'serial'})
            conn.close()
        except Exception:
            continue

    for port in udp_ports:
        conn_str = f'udp:127.0.0.1:{port}'
        try:
            conn = mavutil.mavlink_connection(conn_str, baud=57600)
            if conn.wait_heartbeat(timeout=serial_timeout):
                found.append({'connection_string': conn_str, 'type': 'udp'})
            conn.close()
        except Exception:
            continue

    return found
