"""
Phase 2: AI-assisted mission generation.

Takes a field boundary polygon and (optionally) a disease severity
value from your existing scan pipeline, and produces:
  - a lawnmower/grid flight path covering the polygon
  - suggested spray parameters scaled to severity

This is real, working geometry code (no hardware/network dependency),
so unlike the DJI/MAVLink adapters, this part is fully testable today.
"""
import math


def _bounding_box(polygon):
    lats = [p[0] for p in polygon]
    lons = [p[1] for p in polygon]
    return min(lats), max(lats), min(lons), max(lons)


def _point_in_polygon(lat, lon, polygon):
    """Ray casting point-in-polygon test."""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        yi, xi = polygon[i]
        yj, xj = polygon[j]
        if ((yi > lat) != (yj > lat)) and \
           (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def generate_grid_path(polygon, swath_width_m=6.0):
    """polygon: list of [lat, lon]. Returns list of [lat, lon] waypoints
    forming a boustrophedon (back-and-forth) coverage path."""
    if len(polygon) < 3:
        return []

    min_lat, max_lat, min_lon, max_lon = _bounding_box(polygon)

    # Rough meters-per-degree at this latitude
    mean_lat = (min_lat + max_lat) / 2
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(mean_lat))

    lat_span_m = (max_lat - min_lat) * m_per_deg_lat
    n_lines = max(1, int(lat_span_m / swath_width_m))
    lat_step = (max_lat - min_lat) / max(1, n_lines)

    waypoints = []
    for i in range(n_lines + 1):
        lat = min_lat + i * lat_step
        # sample across this row to find where it crosses the polygon
        crossings = []
        n_samples = 200
        prev_inside = False
        for s in range(n_samples + 1):
            lon = min_lon + (max_lon - min_lon) * s / n_samples
            inside = _point_in_polygon(lat, lon, polygon)
            if inside != prev_inside:
                crossings.append(lon)
            prev_inside = inside
        if len(crossings) >= 2:
            row = [crossings[0], crossings[-1]]
            if i % 2 == 1:
                row = row[::-1]  # boustrophedon: reverse every other row
            waypoints.append([lat, row[0]])
            waypoints.append([lat, row[1]])

    return waypoints


def generate_mission(polygon, disease_severity_pct=None, chemical='General Fungicide',
                      swath_width_m=6.0, base_dosage_l_per_ha=2.0):
    """Returns a mission dict ready to hand to DroneHubMission."""
    waypoints_2d = generate_grid_path(polygon, swath_width_m=swath_width_m)
    waypoints = [[lat, lon, 3.0] for lat, lon in waypoints_2d]  # default 3m spray altitude

    severity = disease_severity_pct if disease_severity_pct is not None else 30.0
    # scale dosage with severity, capped to a sane agronomic range
    dosage = round(base_dosage_l_per_ha * (0.5 + severity / 100.0), 2)
    dosage = max(0.5, min(dosage, base_dosage_l_per_ha * 2.0))

    speed = 5.0 if severity < 50 else 3.5  # slower, more thorough pass for severe cases

    return {
        'waypoints': waypoints,
        'chemical': chemical,
        'dosage_l_per_ha': dosage,
        'spray_height_m': 2.5 if severity < 50 else 2.0,
        'speed_mps': speed,
        'overlap_pct': 10.0,
        'waypoint_count': len(waypoints),
    }
