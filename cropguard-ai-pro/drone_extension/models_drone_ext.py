"""
New, additive models for the Drone Hub extension.

IMPORTANT: this file does NOT define its own `db` instance. It imports
the shared `db` object from your existing models.py so all tables live
on the same SQLAlchemy metadata / same database file. This is required
- do not create a second `SQLAlchemy()` instance.

All class + table names here are deliberately prefixed/distinct
(DroneHub*) so they cannot collide with anything already in your
models.py (DronePlan, DroneDevice, FleetDrone, NoFlyZone, MissionShare,
FarmZone, MaintenanceLog, MissionSchedule, etc).
"""
import json
from datetime import datetime

# This import assumes your project root is on sys.path when Flask runs
# (normal case when you `python app.py` from the project root).
from models import db


class DroneHubMission(db.Model):
    """A mission created via the new Drone Connection dashboard.
    Separate from DronePlan so nothing about your existing DronePlan
    rows/behaviour changes."""
    __tablename__ = 'dronehub_missions'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False, default='Untitled Mission')
    field_id = db.Column(db.Integer, db.ForeignKey('farm_fields.id'), nullable=True)
    device_id = db.Column(db.Integer, nullable=True)  # FK-ish ref to DroneDevice.id (loose, avoids hard dependency)

    boundary = db.Column(db.Text)         # JSON [[lat,lon], ...]
    waypoints = db.Column(db.Text)        # JSON [[lat,lon,alt], ...]
    chemical = db.Column(db.String(120))
    dosage_l_per_ha = db.Column(db.Float, default=0.0)
    spray_height_m = db.Column(db.Float, default=2.0)
    speed_mps = db.Column(db.Float, default=4.0)
    overlap_pct = db.Column(db.Float, default=10.0)

    status = db.Column(db.String(30), default='draft')
    # draft -> uploaded -> running -> paused -> completed / aborted

    ai_generated = db.Column(db.Boolean, default=False)
    source_scan_id = db.Column(db.Integer, nullable=True)  # optional ref to ScanHistory.id

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'field_id': self.field_id,
            'device_id': self.device_id,
            'boundary': json.loads(self.boundary) if self.boundary else [],
            'waypoints': json.loads(self.waypoints) if self.waypoints else [],
            'chemical': self.chemical,
            'dosage_l_per_ha': self.dosage_l_per_ha,
            'spray_height_m': self.spray_height_m,
            'speed_mps': self.speed_mps,
            'overlap_pct': self.overlap_pct,
            'status': self.status,
            'ai_generated': self.ai_generated,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class DroneHubTelemetryLog(db.Model):
    """Rolling telemetry samples per device. In Phase 1-3 these rows are
    written by the simulator; in Phase 4 they'd be written by the real
    MAVLink adapter instead - the schema doesn't change either way."""
    __tablename__ = 'dronehub_telemetry_log'

    id = db.Column(db.Integer, primary_key=True)
    device_id = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    altitude_m = db.Column(db.Float)
    speed_mps = db.Column(db.Float)
    heading_deg = db.Column(db.Float)
    satellites = db.Column(db.Integer)
    battery_pct = db.Column(db.Float)
    signal_strength_pct = db.Column(db.Float)
    flight_mode = db.Column(db.String(40))
    tank_pct = db.Column(db.Float)
    flow_rate_lpm = db.Column(db.Float)

    def to_dict(self):
        return {
            'device_id': self.device_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'altitude_m': self.altitude_m,
            'speed_mps': self.speed_mps,
            'heading_deg': self.heading_deg,
            'satellites': self.satellites,
            'battery_pct': self.battery_pct,
            'signal_strength_pct': self.signal_strength_pct,
            'flight_mode': self.flight_mode,
            'tank_pct': self.tank_pct,
            'flow_rate_lpm': self.flow_rate_lpm,
        }


class DroneHubReport(db.Model):
    """Post-mission report record."""
    __tablename__ = 'dronehub_reports'

    id = db.Column(db.Integer, primary_key=True)
    mission_id = db.Column(db.Integer, db.ForeignKey('dronehub_missions.id'), nullable=False)

    flight_time_sec = db.Column(db.Integer, default=0)
    distance_m = db.Column(db.Float, default=0.0)
    area_covered_ha = db.Column(db.Float, default=0.0)
    chemical_used_l = db.Column(db.Float, default=0.0)
    battery_used_pct = db.Column(db.Float, default=0.0)
    success = db.Column(db.Boolean, default=True)
    warnings = db.Column(db.Text)  # JSON list of strings
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'mission_id': self.mission_id,
            'flight_time_sec': self.flight_time_sec,
            'distance_m': self.distance_m,
            'area_covered_ha': self.area_covered_ha,
            'chemical_used_l': self.chemical_used_l,
            'battery_used_pct': self.battery_used_pct,
            'success': self.success,
            'warnings': json.loads(self.warnings) if self.warnings else [],
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
