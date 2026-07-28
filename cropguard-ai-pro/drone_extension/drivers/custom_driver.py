"""
Custom driver TEMPLATE - for bespoke hardware (e.g. an Arduino/ESP32
flight controller, a Betaflight/iNav FPV board over MSP, or anything
else that doesn't speak MAVLink).

This is deliberately a fill-in-the-blanks template, not a working
driver - "custom hardware" has no fixed protocol to write against.
The serial read/write plumbing below is real and works (pyserial),
but the actual command bytes/strings you send depend entirely on
firmware running on your specific board. Replace the TODOs with your
hardware's real command set (check its firmware's serial protocol
docs, or the source code if it's your own).
"""
import time

from .base import DroneDriver, DroneDriverError

try:
    import serial
    PYSERIAL_AVAILABLE = True
except ImportError:
    PYSERIAL_AVAILABLE = False


class CustomDriver(DroneDriver):
    def __init__(self):
        self._ser = None

    def connect(self, connection_string: str, baud: int = 115200, **kwargs) -> dict:
        if not PYSERIAL_AVAILABLE:
            raise DroneDriverError("pyserial is not installed. Run: pip install pyserial")
        try:
            self._ser = serial.Serial(connection_string, baudrate=baud, timeout=1)
        except Exception as e:
            raise DroneDriverError(f"Could not open serial port '{connection_string}': {e}")
        # TODO: send your board's actual "who are you" / handshake
        # command here and parse the real reply instead of assuming
        # success just because the port opened.
        return {'connected': True, 'port': connection_string}

    def disconnect(self) -> None:
        if self._ser:
            self._ser.close()
        self._ser = None

    def is_connected(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def _require_conn(self):
        if not self.is_connected():
            raise DroneDriverError("Not connected - call connect() first")
        return self._ser

    def _send(self, command: str) -> str:
        ser = self._require_conn()
        ser.write((command + '\n').encode())
        time.sleep(0.05)
        return ser.readline().decode(errors='ignore').strip()

    # ---- everything below is a guess at a plausible text-command
    # protocol - replace with your board's real command set. ----
    def arm(self) -> dict:
        # TODO: replace 'ARM' with your firmware's real arm command
        self._send('ARM')
        return {'armed': True}

    def disarm(self) -> dict:
        self._send('DISARM')
        return {'armed': False}

    def takeoff(self, altitude_m: float) -> dict:
        self._send(f'TAKEOFF {altitude_m}')
        return {'status': 'takeoff_commanded'}

    def land(self) -> dict:
        self._send('LAND')
        return {'status': 'land_commanded'}

    def return_to_home(self) -> dict:
        self._send('RTH')
        return {'status': 'rtl_commanded'}

    def upload_mission(self, waypoints: list) -> dict:
        raise DroneDriverError("upload_mission: define your board's waypoint upload format")

    def start_mission(self) -> dict:
        self._send('MISSION_START')
        return {'status': 'started'}

    def pause_mission(self) -> dict:
        self._send('MISSION_PAUSE')
        return {'status': 'paused'}

    def resume_mission(self) -> dict:
        self._send('MISSION_RESUME')
        return {'status': 'resumed'}

    def stop_mission(self) -> dict:
        self._send('MISSION_STOP')
        return {'status': 'stopped'}

    def camera_stream_url(self) -> str:
        raise DroneDriverError("camera_stream_url: set this to your board's RTSP/HTTP camera URL if it has one")

    def capture_photo(self) -> dict:
        raise DroneDriverError("capture_photo: not implemented for custom hardware")

    def start_video(self) -> dict:
        raise DroneDriverError("start_video: not implemented for custom hardware")

    def stop_video(self) -> dict:
        raise DroneDriverError("stop_video: not implemented for custom hardware")

    def spray_start(self, rate_lpm: float = None) -> dict:
        self._send('SPRAY_ON')
        return {'status': 'spraying'}

    def spray_stop(self) -> dict:
        self._send('SPRAY_OFF')
        return {'status': 'stopped'}

    def telemetry(self) -> dict:
        raise DroneDriverError("telemetry: parse your board's real telemetry output format here")

    def health(self) -> dict:
        return {'safe_to_fly': None, 'warnings': ['health check not implemented for custom hardware']}

    def emergency(self, action: str) -> dict:
        if action == 'return_home':
            return self.return_to_home()
        if action == 'emergency_land':
            return self.land()
        if action == 'stop_spraying':
            return self.spray_stop()
        if action == 'stop_mission':
            return self.stop_mission()
        raise DroneDriverError(f"Unsupported emergency action '{action}'")
