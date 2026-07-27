"""
CropGuard AI - Drone Fleet Manager (Phase 3 / Phase 5)
Manages multiple drones for large farm operations.

Features:
  - Assign different zones to different drones
  - Battery status monitoring
  - Collision avoidance (altitude separation)
  - Emergency RTL (Return to Launch)
  - Mission handoff when a drone's battery is low
"""
import asyncio
from datetime import datetime
from typing import Optional


class DroneAgent:
    """Represents a single drone in the fleet."""

    def __init__(self, drone_id: str, connection_string: str, zone_name: str = ""):
        self.drone_id          = drone_id
        self.connection_string = connection_string
        self.zone_name         = zone_name
        self.battery_pct       = 100.0
        self.status            = "IDLE"   # IDLE, SCANNING, SPRAYING, RTL, CHARGING, ERROR
        self.current_mission   = None
        self.altitude_slot     = 0        # altitude separation for collision avoidance

        # Bridge instance (lazy-loaded)
        self._bridge = None

    async def get_bridge(self):
        if self._bridge is None:
            from modules.drone.mavlink_bridge import MAVLinkDroneBridge
            self._bridge = MAVLinkDroneBridge(self.connection_string)
            await self._bridge.connect()
        return self._bridge

    def to_dict(self) -> dict:
        return {
            "drone_id":     self.drone_id,
            "zone":         self.zone_name,
            "battery_pct":  self.battery_pct,
            "status":       self.status,
            "connection":   self.connection_string,
            "altitude_slot": self.altitude_slot,
        }


class DroneFleetManager:
    """
    Manages a fleet of drones for large-scale agricultural operations.

    Usage:
        fleet = DroneFleetManager()
        fleet.register_drone("Drone-1", "udp://192.168.1.10:14540", zone="North Field")
        fleet.register_drone("Drone-2", "udp://192.168.1.11:14540", zone="South Field")
        await fleet.assign_scan_mission(field_polygons)
    """

    def __init__(self, min_battery_to_fly: float = 20.0):
        self.drones: dict[str, DroneAgent] = {}
        self.min_battery = min_battery_to_fly
        self.mission_log = []
        print("✅ Drone Fleet Manager initialized")

    # ── Fleet Management ───────────────────────────────────────────────────────

    def register_drone(self, drone_id: str, connection_string: str,
                       zone: str = "", altitude_slot: int = None) -> DroneAgent:
        """Add a drone to the fleet."""
        agent = DroneAgent(drone_id, connection_string, zone)
        agent.altitude_slot = altitude_slot if altitude_slot is not None else len(self.drones)
        self.drones[drone_id] = agent
        print(f"✅ Drone registered: {drone_id} | Zone: {zone} | Altitude slot: {agent.altitude_slot}")
        return agent

    def unregister_drone(self, drone_id: str):
        """Remove a drone from the fleet."""
        self.drones.pop(drone_id, None)

    def get_fleet_status(self) -> dict:
        """Return status of all drones."""
        return {
            "total_drones":  len(self.drones),
            "available":     [d.to_dict() for d in self.drones.values() if d.battery_pct >= self.min_battery],
            "low_battery":   [d.to_dict() for d in self.drones.values() if d.battery_pct < self.min_battery],
            "all":           [d.to_dict() for d in self.drones.values()],
            "timestamp":     datetime.utcnow().isoformat(),
        }

    def get_available_drones(self) -> list[DroneAgent]:
        """Return drones with sufficient battery and IDLE status."""
        return [
            d for d in self.drones.values()
            if d.battery_pct >= self.min_battery and d.status in ("IDLE", "CHARGING")
        ]

    # ── Mission Assignment ─────────────────────────────────────────────────────

    def assign_scan_zones(self, field_polygon: list, zone_count: int = None) -> dict:
        """
        Split a large field into zones and assign each to a drone.

        Args:
            field_polygon: full field boundary as [(lat,lon)...]
            zone_count:    number of zones (defaults to number of available drones)

        Returns:
            zone assignments dict
        """
        available = self.get_available_drones()
        if not available:
            return {"success": False, "error": "No available drones (check battery levels)"}

        zone_count = zone_count or len(available)
        zone_count = min(zone_count, len(available))

        # Split field into horizontal strips
        lats = [p[0] for p in field_polygon]
        lons = [p[1] for p in field_polygon]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        lat_range = max_lat - min_lat
        lat_step  = lat_range / zone_count

        assignments = {}
        for i, drone in enumerate(available[:zone_count]):
            zone_min_lat = min_lat + i * lat_step
            zone_max_lat = min_lat + (i + 1) * lat_step

            zone_polygon = [
                (zone_min_lat, min_lon),
                (zone_min_lat, max_lon),
                (zone_max_lat, max_lon),
                (zone_max_lat, min_lon),
            ]

            drone.zone_name    = f"Zone {chr(65 + i)}"  # Zone A, B, C...
            drone.status       = "ASSIGNED"
            drone.current_mission = {"type": "SCAN", "zone": zone_polygon}

            assignments[drone.drone_id] = {
                "drone_id":     drone.drone_id,
                "zone_name":    drone.zone_name,
                "zone_polygon": zone_polygon,
                "altitude_m":   20 + drone.altitude_slot * 5,  # 20m, 25m, 30m... for separation
                "status":       "ASSIGNED",
            }

        self.mission_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "type":      "MULTI_DRONE_SCAN",
            "zones":     list(assignments.keys()),
        })

        return {
            "success":     True,
            "total_zones": zone_count,
            "assignments": assignments,
            "note": (f"Field split into {zone_count} zones. "
                     f"Each drone flies at different altitude (5m separation) for collision avoidance."),
        }

    def assign_spray_mission(self, disease_coords: list) -> dict:
        """
        Assign spray waypoints across available drones (load balancing).

        Args:
            disease_coords: list of {lat, lon, severity, disease_name}

        Returns:
            spray assignments per drone
        """
        available = self.get_available_drones()
        if not available:
            return {"success": False, "error": "No available drones"}

        # Sort by severity
        severity_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        sorted_coords = sorted(
            disease_coords,
            key=lambda c: severity_order.get(c.get("severity", "Low"), 3)
        )

        # Round-robin assignment
        assignments = {d.drone_id: [] for d in available}
        drone_cycle = [d.drone_id for d in available]
        for i, coord in enumerate(sorted_coords):
            drone_id = drone_cycle[i % len(drone_cycle)]
            assignments[drone_id].append(coord)

        result = {}
        for drone in available:
            pts = assignments[drone.drone_id]
            drone.status = "ASSIGNED_SPRAY"
            result[drone.drone_id] = {
                "drone_id":       drone.drone_id,
                "spray_points":   len(pts),
                "coordinates":    pts,
                "altitude_m":     3 + drone.altitude_slot,  # Very low for spraying
            }

        return {
            "success":       True,
            "total_points":  len(disease_coords),
            "drones_used":   len(available),
            "assignments":   result,
        }

    async def emergency_all_return_to_home(self):
        """Send RTL command to all active drones."""
        print("🚨 EMERGENCY RTL: All drones returning to home!")
        tasks = []
        for drone in self.drones.values():
            if drone.status not in ("IDLE", "CHARGING", "ERROR"):
                tasks.append(self._rtl_drone(drone))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _rtl_drone(self, drone: DroneAgent):
        try:
            bridge = await drone.get_bridge()
            await bridge.return_to_home()
            drone.status = "RTL"
            print(f"  ✅ {drone.drone_id} returning to home")
        except Exception as e:
            drone.status = "ERROR"
            print(f"  ❌ {drone.drone_id} RTL failed: {e}")


# Singleton
_fleet_instance: DroneFleetManager | None = None

def get_fleet_manager() -> DroneFleetManager:
    global _fleet_instance
    if _fleet_instance is None:
        _fleet_instance = DroneFleetManager()
    return _fleet_instance
