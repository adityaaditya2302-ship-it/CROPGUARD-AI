"""
CropGuard AI - MAVLink Drone Bridge (Phase 3)
Connects to PX4/ArduPilot drones via MAVLink protocol.
Handles: connection, telemetry, mission upload, arm/takeoff/land.

Compatible with: DJI (via MSDK bridge), ArduPilot, PX4, any MAVLink drone.
Uses MAVSDK-Python for clean async drone communication.
"""
import os
import json
import math
import asyncio
from datetime import datetime
from typing import Optional


# MAVLink waypoint command codes
MAV_CMD_NAV_WAYPOINT    = 16
MAV_CMD_NAV_TAKEOFF     = 22
MAV_CMD_NAV_LAND        = 21
MAV_CMD_NAV_RETURN_TO_LAUNCH = 20
MAV_CMD_DO_SET_SERVO    = 183   # For spray pump control
MAV_CMD_NAV_LOITER_TIME = 19


class DroneConnectionError(Exception):
    pass


class MAVLinkDroneBridge:
    """
    Bridge between CropGuard AI and MAVLink-compatible drones.

    Usage:
        bridge = MAVLinkDroneBridge("udp://:14540")
        await bridge.connect()
        await bridge.fly_mission(waypoints)
        await bridge.disconnect()
    """

    def __init__(self, connection_string: str = "udp://:14540"):
        """
        Args:
            connection_string: MAVLink connection string, e.g.:
                "udp://:14540"          - UDP local (SITL/simulator)
                "serial:///dev/ttyS0:57600" - Serial to autopilot
                "tcp://192.168.1.100:5760" - TCP to companion computer
        """
        self.connection_string = connection_string
        self.drone = None
        self.is_connected = False
        self.telemetry = {}
        self._mavsdk_available = self._check_mavsdk()

    def _check_mavsdk(self) -> bool:
        try:
            import mavsdk
            return True
        except ImportError:
            print("⚠️  MAVSDK not installed. Install with: pip install mavsdk")
            print("   Running in simulation mode for development.")
            return False

    # ── Connection ─────────────────────────────────────────────────────────────

    async def connect(self, timeout: int = 30) -> bool:
        """Connect to drone. Returns True if successful."""
        if not self._mavsdk_available:
            print("🎮 DRONE SIMULATION MODE: No real drone connected.")
            self.is_connected = True
            self.telemetry = self._mock_telemetry()
            return True

        try:
            from mavsdk import System
            self.drone = System()
            await self.drone.connect(system_address=self.connection_string)

            # Wait for connection
            print(f"🔌 Connecting to drone at {self.connection_string}...")
            async for state in self.drone.core.connection_state():
                if state.is_connected:
                    print("✅ Drone connected!")
                    self.is_connected = True
                    break
                await asyncio.sleep(1)

            # Wait for GPS lock
            print("🛰️  Waiting for GPS lock...")
            async for health in self.drone.telemetry.health():
                if health.is_global_position_ok:
                    print("✅ GPS locked")
                    break
                await asyncio.sleep(0.5)

            return True

        except asyncio.TimeoutError:
            raise DroneConnectionError(f"Timeout connecting to drone at {self.connection_string}")
        except Exception as e:
            raise DroneConnectionError(f"Connection failed: {e}")

    async def disconnect(self):
        """Safely disconnect from drone."""
        self.is_connected = False
        self.drone = None
        print("🔌 Drone disconnected.")

    # ── Telemetry ──────────────────────────────────────────────────────────────

    async def get_telemetry(self) -> dict:
        """Return current drone telemetry as a dict."""
        if not self._mavsdk_available or not self.drone:
            return self._mock_telemetry()

        try:
            async for pos in self.drone.telemetry.position():
                lat = pos.latitude_deg
                lon = pos.longitude_deg
                alt = pos.relative_altitude_m
                break
        except Exception:
            lat, lon, alt = 0.0, 0.0, 0.0

        try:
            async for bat in self.drone.telemetry.battery():
                battery = bat.remaining_percent
                break
        except Exception:
            battery = 100.0

        try:
            async for flight in self.drone.telemetry.flight_mode():
                mode = str(flight)
                break
        except Exception:
            mode = "UNKNOWN"

        self.telemetry = {
            "timestamp":       datetime.utcnow().isoformat(),
            "latitude":        lat,
            "longitude":       lon,
            "altitude_m":      alt,
            "battery_pct":     battery,
            "flight_mode":     mode,
            "is_connected":    self.is_connected,
            "is_armed":        False,  # simplified
        }
        return self.telemetry

    def _mock_telemetry(self) -> dict:
        return {
            "timestamp":   datetime.utcnow().isoformat(),
            "latitude":    20.5937,
            "longitude":   78.9629,
            "altitude_m":  0.0,
            "battery_pct": 100.0,
            "flight_mode": "SIMULATION",
            "is_connected": True,
            "is_armed":    False,
            "note":        "Simulation mode - no real drone",
        }

    # ── Mission Control ────────────────────────────────────────────────────────

    async def arm_and_takeoff(self, altitude_m: float = 20.0):
        """Arm the drone and take off to specified altitude."""
        if not self._mavsdk_available or not self.drone:
            print(f"🎮 SIMULATE: Arm and takeoff to {altitude_m}m")
            return True

        print("🔒 Arming drone...")
        await self.drone.action.arm()

        print(f"🚁 Taking off to {altitude_m}m...")
        await self.drone.action.takeoff()
        await asyncio.sleep(5)
        return True

    async def fly_to_waypoint(self, lat: float, lon: float, alt_m: float, speed_ms: float = 5.0):
        """Fly to a single GPS waypoint."""
        if not self._mavsdk_available or not self.drone:
            print(f"🎮 SIMULATE: Flying to ({lat:.4f}, {lon:.4f}) @ {alt_m}m")
            return True

        from mavsdk.mission import MissionItem, MissionPlan
        mission_item = MissionItem(
            lat, lon, alt_m, speed_ms,
            is_fly_through=True,
            gimbal_pitch_deg=float("nan"),
            gimbal_yaw_deg=float("nan"),
            camera_action=MissionItem.CameraAction.NONE,
            loiter_time_s=float("nan"),
            camera_photo_interval_s=float("nan"),
            acceptance_radius_m=2.0,
            yaw_deg=float("nan"),
            camera_photo_distance_m=float("nan"),
        )
        plan = MissionPlan([mission_item])
        await self.drone.mission.set_return_to_launch_after_mission(False)
        await self.drone.mission.upload_mission(plan)
        await self.drone.mission.start_mission()
        return True

    async def fly_mission(self, waypoints: list[dict]) -> dict:
        """
        Upload and execute a multi-waypoint mission.

        Args:
            waypoints: list of dicts with keys: lat, lon, alt_m, [speed_ms], [spray]

        Returns:
            mission result dict
        """
        if not waypoints:
            return {"success": False, "error": "No waypoints provided"}

        print(f"📍 Starting mission: {len(waypoints)} waypoints")
        start_time = datetime.utcnow()

        if not self._mavsdk_available or not self.drone:
            print("🎮 SIMULATE: Full mission execution")
            return {
                "success":       True,
                "mode":          "simulation",
                "waypoints":     len(waypoints),
                "estimated_time_min": self._estimate_mission_time(waypoints),
                "images_expected": len(waypoints),
            }

        from mavsdk.mission import MissionItem, MissionPlan
        mission_items = []
        for wp in waypoints:
            item = MissionItem(
                wp["lat"], wp["lon"], wp.get("alt_m", 20),
                wp.get("speed_ms", 5),
                is_fly_through=not wp.get("hover", False),
                gimbal_pitch_deg=-90.0,    # Camera pointing down
                gimbal_yaw_deg=float("nan"),
                camera_action=MissionItem.CameraAction.TAKE_PHOTO if wp.get("photo", True) else MissionItem.CameraAction.NONE,
                loiter_time_s=wp.get("hover_time", float("nan")),
                camera_photo_interval_s=float("nan"),
                acceptance_radius_m=wp.get("radius_m", 2.0),
                yaw_deg=float("nan"),
                camera_photo_distance_m=float("nan"),
            )
            mission_items.append(item)

        plan = MissionPlan(mission_items)
        await self.drone.mission.set_return_to_launch_after_mission(True)
        await self.drone.mission.upload_mission(plan)
        await self.drone.mission.start_mission()

        return {
            "success":           True,
            "mode":              "live",
            "waypoints":         len(waypoints),
            "mission_start":     start_time.isoformat(),
            "estimated_time_min": self._estimate_mission_time(waypoints),
        }

    async def return_to_home(self):
        """Command drone to return to launch point."""
        if not self._mavsdk_available or not self.drone:
            print("🎮 SIMULATE: Return to home")
            return True
        await self.drone.action.return_to_launch()
        return True

    async def land(self):
        """Command drone to land at current position."""
        if not self._mavsdk_available or not self.drone:
            print("🎮 SIMULATE: Landing")
            return True
        await self.drone.action.land()
        return True

    # ── Spray Control ──────────────────────────────────────────────────────────

    async def activate_sprayer(self, duration_seconds: float = 3.0):
        """Activate the spray pump for specified duration."""
        if not self._mavsdk_available or not self.drone:
            print(f"🎮 SIMULATE: Spray ON for {duration_seconds}s")
            return True

        # Send servo command to activate pump (servo 9, PWM 2000 = ON)
        await self.drone.action.set_actuator(9, 1.0)
        await asyncio.sleep(duration_seconds)
        await self.drone.action.set_actuator(9, -1.0)   # OFF
        return True

    # ── Utility ────────────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_mission_time(waypoints: list[dict], speed_ms: float = 5.0) -> float:
        """Estimate mission time in minutes based on waypoints and speed."""
        if len(waypoints) < 2:
            return 5.0

        total_dist_m = 0.0
        for i in range(1, len(waypoints)):
            prev = waypoints[i-1]
            curr = waypoints[i]
            # Haversine approximate
            dlat = math.radians(curr["lat"] - prev["lat"])
            dlon = math.radians(curr["lon"] - prev["lon"])
            a = (math.sin(dlat/2)**2 +
                 math.cos(math.radians(prev["lat"])) *
                 math.cos(math.radians(curr["lat"])) *
                 math.sin(dlon/2)**2)
            dist = 2 * 6371000 * math.asin(math.sqrt(a))
            total_dist_m += dist

        flight_time_s = total_dist_m / speed_ms
        return round((flight_time_s + len(waypoints) * 5) / 60, 1)   # +5s per waypoint for stabilize


# Singleton
_drone_instance: MAVLinkDroneBridge | None = None

def get_drone_bridge(connection_string: str = None) -> MAVLinkDroneBridge:
    global _drone_instance
    if _drone_instance is None:
        conn = connection_string or os.environ.get("DRONE_CONNECTION", "udp://:14540")
        _drone_instance = MAVLinkDroneBridge(conn)
    return _drone_instance
