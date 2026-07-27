"""
Drone Hub API - Flask Blueprint.

Registered under a distinct URL prefix (/api/dronehub/...) specifically
so it cannot collide with any existing /api/drone/... routes already
in your app.py.

Phases 1-3 (this file's default wiring) use the SIMULATED telemetry
backend - no hardware required, fully testable today.

To move to real hardware later (Phase 4/5), see the "ADAPTER SWITCH"
comment below - it's a one-line change.
"""
import json
from datetime import datetime

from flask import Blueprint, jsonify, request, render_template, Response

from models import db, FarmField
try:
    from models import DroneDevice  # your existing model, if present
except ImportError:
    DroneDevice = None

from .models_drone_ext import DroneHubMission, DroneHubTelemetryLog, DroneHubReport
from . import ai_mission_generator
from . import report_generator

# --- ADAPTER SWITCH ---------------------------------------------------
# Phase 1-3 default: simulated telemetry, no hardware needed.
from . import simulated_telemetry as drone_adapter
# Phase 4 (after verifying against SITL/real hardware), swap to:
#   from . import mavlink_adapter as drone_adapter
# ------------------------------------------------------------------------

drone_bp = Blueprint('dronehub', __name__, url_prefix='/api/dronehub')


# ---------------------------------------------------------------------
# Dashboard page
# ---------------------------------------------------------------------
@drone_bp.route('/dashboard')
def dashboard():
    return render_template('drone_connection.html')


# ---------------------------------------------------------------------
# Detection / connection
# ---------------------------------------------------------------------
@drone_bp.route('/scan', methods=['GET'])
def scan_drones():
    """Phase 1: 'detect' drones. If you already have DroneDevice rows,
    surface those; otherwise return a couple of demo entries so the UI
    has something to show before any real device is registered."""
    devices = []
    if DroneDevice is not None:
        try:
            for d in DroneDevice.query.all():
                devices.append({
                    'id': d.id,
                    'name': getattr(d, 'name', f'Drone {d.id}'),
                    'connection_type': 'registered',
                })
        except Exception:
            pass

    if not devices:
        devices = [
            {'id': 1, 'name': 'Demo Drone (Simulated T40)', 'connection_type': 'simulated'},
        ]
    return jsonify({'devices': devices})


@drone_bp.route('/connect/<int:device_id>', methods=['POST'])
def connect_device(device_id):
    state = drone_adapter.connect(device_id)
    return jsonify({'connected': True, 'device_id': device_id, 'state': state})


@drone_bp.route('/disconnect/<int:device_id>', methods=['POST'])
def disconnect_device(device_id):
    drone_adapter.disconnect(device_id)
    return jsonify({'connected': False, 'device_id': device_id})


# ---------------------------------------------------------------------
# Live telemetry + health
# ---------------------------------------------------------------------
@drone_bp.route('/telemetry/<int:device_id>', methods=['GET'])
def telemetry(device_id):
    data = drone_adapter.get_telemetry(device_id)

    # persist a lightweight log row (optional but useful for reports)
    log = DroneHubTelemetryLog(
        device_id=device_id,
        latitude=data.get('latitude'),
        longitude=data.get('longitude'),
        altitude_m=data.get('altitude_m'),
        speed_mps=data.get('speed_mps'),
        heading_deg=data.get('heading_deg'),
        satellites=data.get('satellites'),
        battery_pct=data.get('battery_pct'),
        signal_strength_pct=data.get('signal_strength_pct'),
        flight_mode=data.get('flight_mode'),
        tank_pct=data.get('tank_pct'),
        flow_rate_lpm=data.get('flow_rate_lpm'),
    )
    db.session.add(log)
    db.session.commit()

    return jsonify(data)


@drone_bp.route('/health/<int:device_id>', methods=['GET'])
def health(device_id):
    if hasattr(drone_adapter, 'health_check'):
        return jsonify(drone_adapter.health_check(device_id))
    return jsonify({'safe_to_fly': True, 'warnings': [], 'note': 'health_check not implemented for this adapter'})


# ---------------------------------------------------------------------
# Weather safety gate
# ---------------------------------------------------------------------
@drone_bp.route('/weather-check', methods=['GET'])
def weather_check():
    """Tries to use your existing weather_intelligence module if present;
    otherwise returns a permissive default so the UI still works."""
    try:
        from modules.weather_intelligence import get_current_weather  # adjust if your function name differs
        weather = get_current_weather(request.args.get('lat'), request.args.get('lon'))
    except Exception:
        weather = {'wind_speed_kmh': 8, 'rain': False, 'visibility_km': 10}

    wind = weather.get('wind_speed_kmh', 0)
    rain = weather.get('rain', False)
    safe = wind < 20 and not rain
    return jsonify({'weather': weather, 'safe_to_spray': safe})


# ---------------------------------------------------------------------
# Mission planning (AI-generated)
# ---------------------------------------------------------------------
@drone_bp.route('/mission/generate', methods=['POST'])
def generate_mission():
    data = request.get_json(force=True) or {}
    polygon = data.get('boundary', [])
    if len(polygon) < 3:
        return jsonify({'error': 'boundary must have at least 3 [lat, lon] points'}), 400

    plan = ai_mission_generator.generate_mission(
        polygon,
        disease_severity_pct=data.get('disease_severity_pct'),
        chemical=data.get('chemical', 'General Fungicide'),
        swath_width_m=data.get('swath_width_m', 6.0),
    )

    mission = DroneHubMission(
        name=data.get('name', 'AI Generated Mission'),
        field_id=data.get('field_id'),
        device_id=data.get('device_id'),
        boundary=json.dumps(polygon),
        waypoints=json.dumps(plan['waypoints']),
        chemical=plan['chemical'],
        dosage_l_per_ha=plan['dosage_l_per_ha'],
        spray_height_m=plan['spray_height_m'],
        speed_mps=plan['speed_mps'],
        overlap_pct=plan['overlap_pct'],
        ai_generated=True,
        source_scan_id=data.get('scan_id'),
        status='draft',
    )
    db.session.add(mission)
    db.session.commit()
    return jsonify(mission.to_dict())


@drone_bp.route('/mission/<int:mission_id>', methods=['GET'])
def get_mission(mission_id):
    m = DroneHubMission.query.get_or_404(mission_id)
    return jsonify(m.to_dict())


@drone_bp.route('/missions', methods=['GET'])
def list_missions():
    missions = DroneHubMission.query.order_by(DroneHubMission.created_at.desc()).all()
    return jsonify([m.to_dict() for m in missions])


# ---------------------------------------------------------------------
# Mission execution
# ---------------------------------------------------------------------
@drone_bp.route('/mission/<int:mission_id>/upload', methods=['POST'])
def upload_mission(mission_id):
    mission = DroneHubMission.query.get_or_404(mission_id)
    mission.status = 'uploaded'
    db.session.commit()
    return jsonify({'status': 'uploaded', 'mission_id': mission_id})


@drone_bp.route('/mission/<int:mission_id>/start', methods=['POST'])
def start_mission(mission_id):
    mission = DroneHubMission.query.get_or_404(mission_id)
    if mission.device_id is not None:
        drone_adapter.set_mission_active(mission.device_id, True, speed_mps=mission.speed_mps or 4.0)
    mission.status = 'running'
    db.session.commit()
    return jsonify({'status': 'running', 'mission_id': mission_id})


@drone_bp.route('/mission/<int:mission_id>/pause', methods=['POST'])
def pause_mission(mission_id):
    mission = DroneHubMission.query.get_or_404(mission_id)
    if mission.device_id is not None:
        drone_adapter.set_mission_active(mission.device_id, False)
    mission.status = 'paused'
    db.session.commit()
    return jsonify({'status': 'paused', 'mission_id': mission_id})


@drone_bp.route('/mission/<int:mission_id>/resume', methods=['POST'])
def resume_mission(mission_id):
    mission = DroneHubMission.query.get_or_404(mission_id)
    if mission.device_id is not None:
        drone_adapter.set_mission_active(mission.device_id, True, speed_mps=mission.speed_mps or 4.0)
    mission.status = 'running'
    db.session.commit()
    return jsonify({'status': 'running', 'mission_id': mission_id})


@drone_bp.route('/mission/<int:mission_id>/abort', methods=['POST'])
def abort_mission(mission_id):
    mission = DroneHubMission.query.get_or_404(mission_id)
    if mission.device_id is not None:
        drone_adapter.set_mission_active(mission.device_id, False)
    mission.status = 'aborted'
    db.session.commit()
    return jsonify({'status': 'aborted', 'mission_id': mission_id})


@drone_bp.route('/mission/<int:mission_id>/complete', methods=['POST'])
def complete_mission(mission_id):
    """Marks a mission done and generates a report from the telemetry
    log collected during the flight (works with simulated OR real data,
    since both write to DroneHubTelemetryLog)."""
    mission = DroneHubMission.query.get_or_404(mission_id)
    if mission.device_id is not None:
        drone_adapter.set_mission_active(mission.device_id, False)
    mission.status = 'completed'

    logs = DroneHubTelemetryLog.query.filter_by(device_id=mission.device_id).order_by(
        DroneHubTelemetryLog.timestamp).all()

    flight_time = 0
    distance = 0.0
    if len(logs) >= 2:
        flight_time = int((logs[-1].timestamp - logs[0].timestamp).total_seconds())
        for a, b in zip(logs, logs[1:]):
            if None not in (a.latitude, a.longitude, b.latitude, b.longitude):
                distance += _haversine_m(a.latitude, a.longitude, b.latitude, b.longitude)

    battery_used = 0.0
    if logs:
        battery_used = max(0.0, (logs[0].battery_pct or 100) - (logs[-1].battery_pct or 100))

    waypoints = json.loads(mission.waypoints) if mission.waypoints else []
    area_ha = _estimate_area_ha(json.loads(mission.boundary)) if mission.boundary else 0.0

    warnings = []
    if battery_used > 80:
        warnings.append('High battery consumption for this mission')

    report = DroneHubReport(
        mission_id=mission.id,
        flight_time_sec=flight_time,
        distance_m=round(distance, 1),
        area_covered_ha=round(area_ha, 2),
        chemical_used_l=round(area_ha * (mission.dosage_l_per_ha or 0), 2),
        battery_used_pct=round(battery_used, 1),
        success=True,
        warnings=json.dumps(warnings),
    )
    db.session.add(report)
    db.session.commit()
    return jsonify({'mission': mission.to_dict(), 'report': report.to_dict()})


@drone_bp.route('/mission/<int:mission_id>/rth', methods=['POST'])
def return_home(mission_id):
    mission = DroneHubMission.query.get_or_404(mission_id)
    if hasattr(drone_adapter, 'return_to_home') and mission.device_id is not None:
        drone_adapter.return_to_home(mission.device_id)
    return jsonify({'status': 'returning_home', 'mission_id': mission_id})


# ---------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------
@drone_bp.route('/mission/<int:mission_id>/report', methods=['GET'])
def get_report(mission_id):
    report = DroneHubReport.query.filter_by(mission_id=mission_id).order_by(
        DroneHubReport.created_at.desc()).first()
    if not report:
        return jsonify({'error': 'No report found for this mission yet. Complete the mission first.'}), 404

    fmt = request.args.get('format', 'json')
    mission = DroneHubMission.query.get(mission_id)

    if fmt == 'csv':
        csv_text = report_generator.report_to_csv(report.to_dict())
        return Response(csv_text, mimetype='text/csv',
                         headers={'Content-Disposition': f'attachment; filename=mission_{mission_id}_report.csv'})
    elif fmt == 'pdf':
        try:
            pdf_bytes = report_generator.report_to_pdf_bytes(report.to_dict(), mission.to_dict() if mission else None)
        except RuntimeError as e:
            return jsonify({'error': str(e)}), 501
        return Response(pdf_bytes, mimetype='application/pdf',
                         headers={'Content-Disposition': f'attachment; filename=mission_{mission_id}_report.pdf'})
    else:
        return jsonify(report.to_dict())


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _haversine_m(lat1, lon1, lat2, lon2):
    import math
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(min(1, math.sqrt(a)))


def _estimate_area_ha(polygon):
    """Shoelace formula on an equirectangular approximation - fine for
    small field-sized polygons."""
    import math
    if len(polygon) < 3:
        return 0.0
    mean_lat = sum(p[0] for p in polygon) / len(polygon)
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(mean_lat))
    pts = [(p[1] * m_per_deg_lon, p[0] * m_per_deg_lat) for p in polygon]
    area = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    area = abs(area) / 2.0
    return area / 10000.0  # m^2 -> hectares
