"""
CropGuard AI - Autonomous Mission Planner (Phase 3)
Generates optimal flight paths for field scanning and precision spraying.

Features:
  - Lawnmower/boustrophedon scan pattern
  - Disease-targeted spray waypoints
  - Battery-aware mission splitting
  - Overlap calculation for photogrammetry
  - GeoJSON output for Leaflet/Mapbox display
"""
import math
import json
from typing import List, Tuple
from datetime import datetime


# ── Constants ──────────────────────────────────────────────────────────────────
EARTH_RADIUS_M = 6_371_000
DEFAULT_SCAN_ALT_M   = 30.0   # scanning altitude (meters)
DEFAULT_SPRAY_ALT_M  = 3.0    # spraying altitude (very low for coverage)
DEFAULT_SPEED_SCAN   = 5.0    # m/s scan speed
DEFAULT_SPEED_SPRAY  = 2.0    # m/s spray speed (slow for coverage)
DEFAULT_OVERLAP_PCT  = 70     # image overlap for photogrammetry


class MissionPlanner:
    """
    Generates autonomous flight plans for CropGuard AI drone missions.

    Two mission types:
      1. SCAN mission – lawnmower pattern to capture full field imagery
      2. SPRAY mission – targeted waypoints at disease GPS coordinates only
    """

    def __init__(self, altitude_scan: float = DEFAULT_SCAN_ALT_M,
                       altitude_spray: float = DEFAULT_SPRAY_ALT_M,
                       overlap_pct: float = DEFAULT_OVERLAP_PCT):
        self.altitude_scan  = altitude_scan
        self.altitude_spray = altitude_spray
        self.overlap_pct    = overlap_pct

    # ── Public API ─────────────────────────────────────────────────────────────

    def plan_scan_mission(
        self,
        field_polygon: List[Tuple[float, float]],   # list of (lat, lon) points
        home_lat: float, home_lon: float,
        camera_fov_deg: float = 73.0,               # typical drone camera FOV
        scan_speed_ms: float = DEFAULT_SPEED_SCAN,
    ) -> dict:
        """
        Generate a lawnmower scanning mission for the field polygon.

        Args:
            field_polygon:   list of (lat, lon) tuples defining field boundary
            home_lat/lon:    drone home / takeoff point
            camera_fov_deg:  camera field-of-view (determines strip width)
            scan_speed_ms:   flight speed in m/s

        Returns:
            mission dict with waypoints, GeoJSON, estimated time, coverage
        """
        if len(field_polygon) < 3:
            return {"success": False, "error": "Field polygon needs at least 3 points"}

        # Calculate strip width from altitude and FOV
        strip_width_m = 2 * self.altitude_scan * math.tan(math.radians(camera_fov_deg / 2))
        strip_spacing = strip_width_m * (1 - self.overlap_pct / 100)

        # Get bounding box
        lats = [p[0] for p in field_polygon]
        lons = [p[1] for p in field_polygon]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        # Generate lawnmower waypoints
        waypoints = self._lawnmower_pattern(
            min_lat, max_lat, min_lon, max_lon,
            strip_spacing, self.altitude_scan, scan_speed_ms
        )

        # Add home point at start and end
        home_wp = {"lat": home_lat, "lon": home_lon, "alt_m": 0, "action": "HOME"}
        takeoff_wp = {"lat": home_lat, "lon": home_lon, "alt_m": self.altitude_scan, "action": "TAKEOFF"}
        land_wp    = {"lat": home_lat, "lon": home_lon, "alt_m": 0,                 "action": "LAND"}

        full_mission = [home_wp, takeoff_wp] + waypoints + [land_wp]

        # Metrics
        field_area_ha = self._polygon_area_ha(field_polygon)
        flight_dist_m = self._total_distance_m(full_mission)
        est_time_min  = flight_dist_m / (scan_speed_ms * 60)
        battery_pct_needed = min(est_time_min * 1.5, 100)  # ~1.5% battery/min typical

        return {
            "success":            True,
            "mission_type":       "SCAN",
            "waypoints":          full_mission,
            "waypoint_count":     len(full_mission),
            "field_area_ha":      round(field_area_ha, 2),
            "flight_distance_m":  round(flight_dist_m),
            "estimated_time_min": round(est_time_min, 1),
            "battery_needed_pct": round(battery_pct_needed, 1),
            "strip_width_m":      round(strip_width_m, 1),
            "expected_images":    len(waypoints),
            "geojson":            self._to_geojson(full_mission, field_polygon),
            "created_at":         datetime.utcnow().isoformat(),
        }

    def plan_spray_mission(
        self,
        disease_coordinates: List[dict],   # [{lat, lon, severity, disease_name}]
        home_lat: float, home_lon: float,
        buffer_m: float = 2.0,            # spray buffer around each point (m)
        spray_duration_s: float = 3.0,    # seconds to spray per point
    ) -> dict:
        """
        Generate a precision spray mission targeting only diseased areas.

        Args:
            disease_coordinates: GPS points with disease detections
            home_lat/lon:        home point
            buffer_m:            spray coverage radius per waypoint
            spray_duration_s:    how long to spray at each point

        Returns:
            spray mission dict with ordered waypoints
        """
        if not disease_coordinates:
            return {"success": False, "error": "No disease coordinates provided"}

        # Sort by severity (spray critical areas first)
        severity_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        sorted_coords = sorted(
            disease_coordinates,
            key=lambda c: severity_order.get(c.get("severity", "Low"), 3)
        )

        # Optimize order using nearest-neighbor TSP approximation
        optimized = self._nearest_neighbor_order(sorted_coords, home_lat, home_lon)

        waypoints = []
        for coord in optimized:
            waypoints.append({
                "lat":          coord["lat"],
                "lon":          coord["lon"],
                "alt_m":        self.altitude_spray,
                "speed_ms":     DEFAULT_SPEED_SPRAY,
                "action":       "SPRAY",
                "spray":        True,
                "hover":        True,
                "hover_time":   spray_duration_s,
                "disease":      coord.get("disease_name", "Unknown"),
                "severity":     coord.get("severity", "Unknown"),
                "radius_m":     buffer_m,
            })

        home_wp = {"lat": home_lat, "lon": home_lon, "alt_m": 0,                  "action": "HOME"}
        takeoff = {"lat": home_lat, "lon": home_lon, "alt_m": self.altitude_spray, "action": "TAKEOFF"}
        land    = {"lat": home_lat, "lon": home_lon, "alt_m": 0,                  "action": "LAND"}

        full_mission = [home_wp, takeoff] + waypoints + [land]

        flight_dist_m = self._total_distance_m(full_mission)
        spray_time_s  = len(optimized) * spray_duration_s
        total_time_min = (flight_dist_m / (DEFAULT_SPEED_SPRAY * 60)) + (spray_time_s / 60)

        # Estimate pesticide savings vs blanket spraying
        spray_area_m2 = len(optimized) * math.pi * (buffer_m ** 2)
        # Field area estimate from bounding box of disease points
        if len(optimized) > 1:
            all_lats = [c["lat"] for c in optimized]
            all_lons = [c["lon"] for c in optimized]
            field_est_m2 = (
                self._haversine_m(min(all_lats), min(all_lons), max(all_lats), min(all_lons)) *
                self._haversine_m(min(all_lats), min(all_lons), min(all_lats), max(all_lons))
            )
        else:
            field_est_m2 = 10000  # 1 hectare default

        pesticide_savings_pct = max(0, 100 - (spray_area_m2 / max(field_est_m2, 1) * 100))

        return {
            "success":               True,
            "mission_type":          "SPRAY",
            "waypoints":             full_mission,
            "spray_points":          len(optimized),
            "flight_distance_m":     round(flight_dist_m),
            "estimated_time_min":    round(total_time_min, 1),
            "spray_time_s":          spray_time_s,
            "pesticide_savings_pct": round(min(pesticide_savings_pct, 85), 1),
            "chemical_saved_liters": round(pesticide_savings_pct / 100 * len(optimized) * 0.5, 2),
            "geojson":               self._to_geojson(full_mission, []),
            "created_at":            datetime.utcnow().isoformat(),
        }

    # ── Patterns ───────────────────────────────────────────────────────────────

    def _lawnmower_pattern(self, min_lat, max_lat, min_lon, max_lon,
                           strip_spacing_m, alt_m, speed_ms) -> list:
        """Generate east-west lawnmower waypoints."""
        waypoints = []
        # Convert strip spacing from meters to degrees latitude
        lat_step = strip_spacing_m / 111_000   # approx 111km per degree lat

        current_lat = min_lat
        row = 0
        while current_lat <= max_lat:
            if row % 2 == 0:
                start_lon, end_lon = min_lon, max_lon
            else:
                start_lon, end_lon = max_lon, min_lon

            waypoints.append({
                "lat":      current_lat,
                "lon":      start_lon,
                "alt_m":    alt_m,
                "speed_ms": speed_ms,
                "action":   "PHOTO",
                "photo":    True,
            })
            waypoints.append({
                "lat":      current_lat,
                "lon":      end_lon,
                "alt_m":    alt_m,
                "speed_ms": speed_ms,
                "action":   "PHOTO",
                "photo":    True,
            })
            current_lat += lat_step
            row += 1

        return waypoints

    def _nearest_neighbor_order(self, coords: list, home_lat: float, home_lon: float) -> list:
        """Simple nearest-neighbor TSP for spray waypoint ordering."""
        if len(coords) <= 1:
            return coords

        remaining = list(coords)
        ordered   = []
        cur_lat, cur_lon = home_lat, home_lon

        while remaining:
            nearest = min(
                remaining,
                key=lambda c: self._haversine_m(cur_lat, cur_lon, c["lat"], c["lon"])
            )
            ordered.append(nearest)
            remaining.remove(nearest)
            cur_lat, cur_lon = nearest["lat"], nearest["lon"]

        return ordered

    # ── Utilities ──────────────────────────────────────────────────────────────

    @staticmethod
    def _haversine_m(lat1, lon1, lat2, lon2) -> float:
        R = EARTH_RADIUS_M
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2)**2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon/2)**2)
        return 2 * R * math.asin(math.sqrt(a))

    def _total_distance_m(self, waypoints: list) -> float:
        total = 0.0
        for i in range(1, len(waypoints)):
            total += self._haversine_m(
                waypoints[i-1]["lat"], waypoints[i-1]["lon"],
                waypoints[i]["lat"],   waypoints[i]["lon"]
            )
        return total

    def _polygon_area_ha(self, polygon: List[Tuple[float, float]]) -> float:
        """Shoelace formula for polygon area in hectares."""
        n = len(polygon)
        if n < 3:
            return 0.0
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            # Convert to approx meters
            x1 = polygon[i][1] * 111_000 * math.cos(math.radians(polygon[i][0]))
            y1 = polygon[i][0] * 111_000
            x2 = polygon[j][1] * 111_000 * math.cos(math.radians(polygon[j][0]))
            y2 = polygon[j][0] * 111_000
            area += x1 * y2 - x2 * y1
        return abs(area) / 2 / 10_000   # m² → hectares

    def _to_geojson(self, waypoints: list, polygon: list) -> dict:
        """Convert mission to GeoJSON for Leaflet display."""
        features = []

        # Flight path line
        coords = [[wp["lon"], wp["lat"]] for wp in waypoints]
        if len(coords) >= 2:
            features.append({
                "type": "Feature",
                "properties": {"type": "flight_path"},
                "geometry": {"type": "LineString", "coordinates": coords},
            })

        # Waypoint markers
        for i, wp in enumerate(waypoints):
            features.append({
                "type": "Feature",
                "properties": {
                    "type":    "waypoint",
                    "index":   i,
                    "action":  wp.get("action", ""),
                    "alt_m":   wp.get("alt_m", 0),
                },
                "geometry": {
                    "type":        "Point",
                    "coordinates": [wp["lon"], wp["lat"]],
                },
            })

        # Field polygon
        if polygon:
            ring = [[lon, lat] for lat, lon in polygon]
            ring.append(ring[0])   # close ring
            features.append({
                "type": "Feature",
                "properties": {"type": "field_boundary"},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            })

        return {"type": "FeatureCollection", "features": features}


# Singleton
_planner_instance: MissionPlanner | None = None

def get_mission_planner() -> MissionPlanner:
    global _planner_instance
    if _planner_instance is None:
        _planner_instance = MissionPlanner()
    return _planner_instance
