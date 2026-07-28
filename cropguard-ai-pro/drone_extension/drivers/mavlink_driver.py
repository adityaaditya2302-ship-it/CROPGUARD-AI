"""
Real MAVLink driver - covers ArduPilot, PX4, Pixhawk, Holybro,
CubePilot, and most other flight controllers that speak MAVLink
(the open standard, not tied to one manufacturer).

*** HONESTY NOTE ***
This is written correctly against the pymavlink API but has NOT been
run against a real flight controller or SITL, because there's no
hardware or simulator available in the environment this was written
in. Before trusting this near a real aircraft:

  1. pip install pymavlink
  2. Install ArduPilot SITL (free, no hardware):
     https://ardupilot.org/dev/docs/sitl-simulator-software-in-the-loop.html
  3. Point connect() at 'udp:127.0.0.1:14550' and test every method
     below by hand (arm/disarm first, in GUIDED/LOITER on the ground,
     before ever attempting takeoff).
  4. Only then point it at a real flight controller.

Command availability is firmware-dependent:
  - MAV_CMD_DO_PAUSE_CONTINUE (pause/resume) is PX4-native; on
    ArduPilot it falls back to switching to LOITER/AUTO modes.
  - Camera and gimbal commands (MAV_CMD_IMAGE_START_CAPTURE etc.)
    only work if your flight stack has a MAVLink camera/gimbal
    component (e.g. a companion computer running mavlink-camera-
    manager). A bare flight controller has no camera of its own.
  - Spray relay channel is airframe-specific - you MUST set
    spray_channel to whatever servo/relay output your sprayer pump
    is actually wired to (see your airframe's wiring diagram).

Safety: emergency('kill') is intentionally NOT implemented as an
in-air motor cutoff. Force-disarming mid-flight drops the aircraft
out of the sky - that's a physical RC kill-switch / GCS failsafe
decision, not something this software should do silently. Use
return_home / emergency_land instead; wire a real hardware kill
switch on the RC transmitter for the true emergency case.
"""
import time

from .base import DroneDriver, DroneDriverError

try:
    from pymavlink import mavutil
    PYMAVLINK_AVAILABLE = True
except ImportError:
    PYMAVLINK_AVAILABLE = False


class MAVLinkDriver(DroneDriver):

    def __init__(self):
        self._conn = None
        self._spray_channel = None  # set via connect(spray_channel=N)

    # ---- connection -----------------------------------------------
    def connect(self, connection_string: str, baud: int = 57600,
                spray_channel: int = None, **kwargs) -> dict:
        if not PYMAVLINK_AVAILABLE:
            raise DroneDriverError("pymavlink is not installed. Run: pip install pymavlink")
        try:
            self._conn = mavutil.mavlink_connection(connection_string, baud=baud)
            hb = self._conn.wait_heartbeat(timeout=10)
        except Exception as e:
            raise DroneDriverError(f"Could not connect via '{connection_string}': {e}")
        if hb is None:
            raise DroneDriverError(f"No heartbeat received on '{connection_string}' within 10s")

        self._spray_channel = spray_channel
        autopilot_names = {0: 'ArduPilot', 12: 'PX4'}
        return {
            'connected': True,
            'system_id': self._conn.target_system,
            'component_id': self._conn.target_component,
            'autopilot': autopilot_names.get(self._conn.mav_autopilot if hasattr(self._conn, 'mav_autopilot') else None, 'unknown'),
        }

    def disconnect(self) -> None:
        if self._conn:
            self._conn.close()
        self._conn = None

    def is_connected(self) -> bool:
        return self._conn is not None

    def _require_conn(self):
        if not self._conn:
            raise DroneDriverError("Not connected - call connect() first")
        return self._conn

    def _send_command(self, command, p1=0, p2=0, p3=0, p4=0, p5=0, p6=0, p7=0, timeout=5):
        """Send a MAV_CMD and wait for COMMAND_ACK. Raises on rejection
        or timeout instead of silently assuming success."""
        conn = self._require_conn()
        conn.mav.command_long_send(
            conn.target_system, conn.target_component,
            command, 0, p1, p2, p3, p4, p5, p6, p7,
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            ack = conn.recv_match(type='COMMAND_ACK', blocking=False)
            if ack and ack.command == command:
                if ack.result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    raise DroneDriverError(
                        f"Flight controller rejected command {command} (result={ack.result})")
                return True
            time.sleep(0.05)
        raise DroneDriverError(f"No COMMAND_ACK received for command {command} within {timeout}s")

    # ---- arm / flight -----------------------------------------------
    def arm(self) -> dict:
        self._send_command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, p1=1)
        return {'armed': True}

    def disarm(self) -> dict:
        self._send_command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, p1=0)
        return {'armed': False}

    def takeoff(self, altitude_m: float) -> dict:
        self._send_command(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, p7=altitude_m)
        return {'status': 'takeoff_commanded', 'target_altitude_m': altitude_m}

    def land(self) -> dict:
        self._send_command(mavutil.mavlink.MAV_CMD_NAV_LAND)
        return {'status': 'land_commanded'}

    def return_to_home(self) -> dict:
        self._send_command(mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH)
        return {'status': 'rtl_commanded'}

    # ---- mission -----------------------------------------------------
    def upload_mission(self, waypoints: list) -> dict:
        """Proper MISSION_COUNT -> (wait MISSION_REQUEST) -> MISSION_ITEM_INT
        -> MISSION_ACK handshake, per the MAVLink mission protocol."""
        conn = self._require_conn()
        conn.mav.mission_count_send(conn.target_system, conn.target_component, len(waypoints))

        sent = 0
        deadline = time.time() + 15
        while sent < len(waypoints) and time.time() < deadline:
            req = conn.recv_match(type=['MISSION_REQUEST', 'MISSION_REQUEST_INT'], blocking=False)
            if not req:
                time.sleep(0.05)
                continue
            i = req.seq
            lat, lon, alt = waypoints[i]
            conn.mav.mission_item_int_send(
                conn.target_system, conn.target_component, i,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                0, 1, 0, 0, 0, 0,
                int(lat * 1e7), int(lon * 1e7), alt,
            )
            sent = max(sent, i + 1)

        ack = conn.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
        if not ack or ack.type != mavutil.mavlink.MAV_MISSION_ACCEPTED:
            raise DroneDriverError(f"Mission upload not acknowledged as accepted (got: {ack})")
        return {'uploaded': True, 'waypoint_count': len(waypoints)}

    def start_mission(self) -> dict:
        conn = self._require_conn()
        conn.set_mode_auto()
        return {'status': 'started'}

    def pause_mission(self) -> dict:
        """PX4-native. On ArduPilot this falls back to LOITER, which
        is the closest equivalent (holds position, does not advance
        the mission)."""
        conn = self._require_conn()
        try:
            self._send_command(mavutil.mavlink.MAV_CMD_DO_PAUSE_CONTINUE, p1=0)
        except DroneDriverError:
            conn.set_mode('LOITER')
        return {'status': 'paused'}

    def resume_mission(self) -> dict:
        conn = self._require_conn()
        try:
            self._send_command(mavutil.mavlink.MAV_CMD_DO_PAUSE_CONTINUE, p1=1)
        except DroneDriverError:
            conn.set_mode_auto()
        return {'status': 'resumed'}

    def stop_mission(self) -> dict:
        conn = self._require_conn()
        try:
            conn.set_mode('LOITER')
        except Exception as e:
            raise DroneDriverError(f"Could not switch to a holding mode: {e}")
        return {'status': 'stopped'}

    # ---- camera --------------------------------------------------------
    def camera_stream_url(self) -> str:
        raise DroneDriverError(
            "Bare MAVLink telemetry links do not carry video. If your "
            "airframe has a companion computer (e.g. Raspberry Pi + "
            "mavlink-camera-manager, or a gimbal with an RTSP output), "
            "point the frontend directly at that RTSP/HTTP URL - it's "
            "independent of this telemetry connection."
        )

    def capture_photo(self) -> dict:
        try:
            self._send_command(mavutil.mavlink.MAV_CMD_IMAGE_START_CAPTURE, p4=1)
        except DroneDriverError as e:
            raise DroneDriverError(f"Camera command rejected/unsupported: {e}")
        return {'status': 'capture_commanded'}

    def start_video(self) -> dict:
        self._send_command(mavutil.mavlink.MAV_CMD_VIDEO_START_CAPTURE)
        return {'status': 'recording'}

    def stop_video(self) -> dict:
        self._send_command(mavutil.mavlink.MAV_CMD_VIDEO_STOP_CAPTURE)
        return {'status': 'stopped'}

    # ---- spray (servo/relay-controlled pump, airframe-specific) -----
    def spray_start(self, rate_lpm: float = None) -> dict:
        if self._spray_channel is None:
            raise DroneDriverError(
                "No spray_channel configured. Pass spray_channel=<servo "
                "output number> to connect() - find it on your airframe's "
                "wiring diagram (the PWM output your sprayer pump/relay "
                "is physically wired to)."
            )
        # 1900us ~= full-on for most relay/ESC-driven pumps; adjust to
        # your hardware's calibration.
        self._send_command(mavutil.mavlink.MAV_CMD_DO_SET_SERVO, p1=self._spray_channel, p2=1900)
        return {'status': 'spraying', 'channel': self._spray_channel, 'rate_lpm': rate_lpm}

    def spray_stop(self) -> dict:
        if self._spray_channel is None:
            raise DroneDriverError("No spray_channel configured.")
        self._send_command(mavutil.mavlink.MAV_CMD_DO_SET_SERVO, p1=self._spray_channel, p2=1100)
        return {'status': 'stopped'}

    # ---- telemetry / health / safety --------------------------------
    def telemetry(self) -> dict:
        conn = self._require_conn()
        data = {}
        # Drain whatever's arrived since last poll (non-blocking).
        for _ in range(20):
            msg = conn.recv_match(
                type=['GLOBAL_POSITION_INT', 'VFR_HUD', 'SYS_STATUS',
                      'GPS_RAW_INT', 'ATTITUDE', 'HEARTBEAT'],
                blocking=False,
            )
            if not msg:
                break
            t = msg.get_type()
            if t == 'GLOBAL_POSITION_INT':
                data['latitude'] = msg.lat / 1e7
                data['longitude'] = msg.lon / 1e7
                data['altitude_m'] = msg.relative_alt / 1000.0
                data['heading_deg'] = msg.hdg / 100.0
            elif t == 'VFR_HUD':
                data['speed_mps'] = msg.groundspeed
                data['climb_mps'] = msg.climb
            elif t == 'SYS_STATUS':
                data['battery_pct'] = msg.battery_remaining
            elif t == 'GPS_RAW_INT':
                data['satellites'] = msg.satellites_visible
            elif t == 'ATTITUDE':
                data['roll_deg'] = msg.roll * 57.2958
                data['pitch_deg'] = msg.pitch * 57.2958
                data['yaw_deg'] = msg.yaw * 57.2958
            elif t == 'HEARTBEAT':
                data['armed'] = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                data['flight_mode'] = mavutil.mode_string_v10(msg) if hasattr(mavutil, 'mode_string_v10') else None
        return data

    def health(self) -> dict:
        conn = self._require_conn()
        msg = conn.recv_match(type='SYS_STATUS', blocking=True, timeout=2)
        warnings = []
        if msg:
            if msg.battery_remaining is not None and msg.battery_remaining < 20:
                warnings.append('Low battery')
            present = msg.onboard_control_sensors_present
            health = msg.onboard_control_sensors_health
            if (present & health) != present:
                warnings.append('One or more onboard sensors report unhealthy')
        return {'safe_to_fly': len(warnings) == 0, 'warnings': warnings}

    def emergency(self, action: str) -> dict:
        if action == 'return_home':
            return self.return_to_home()
        if action == 'emergency_land':
            return self.land()
        if action == 'stop_spraying':
            return self.spray_stop()
        if action == 'stop_mission':
            return self.stop_mission()
        raise DroneDriverError(
            f"Unsupported or unsafe emergency action '{action}'. "
            "In-air motor-cutoff ('kill') is deliberately not exposed here - "
            "use a hardware RC kill switch / GCS failsafe for that."
        )
