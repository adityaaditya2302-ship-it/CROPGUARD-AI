"""
CropGuard AI Pro - Database Models
SQLAlchemy models for scan history, drone plans, user profiles, etc.
"""
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class ScanHistory(db.Model):
    """Store AI scan results"""
    __tablename__ = 'scan_history'

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    crop_name = db.Column(db.String(100), nullable=False)
    crop_icon = db.Column(db.String(10), default='🌱')
    disease_name = db.Column(db.String(200), nullable=False)
    severity = db.Column(db.String(50), default='Unknown')
    confidence = db.Column(db.Float, default=0.0)
    description = db.Column(db.Text)
    symptoms = db.Column(db.Text)  # JSON string
    treatments_chemical = db.Column(db.Text)  # JSON string
    treatments_organic = db.Column(db.Text)  # JSON string
    treatments_prevention = db.Column(db.Text)  # JSON string
    image_path = db.Column(db.String(500))
    source = db.Column(db.String(100), default='YOLOv8')
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)

    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'crop_name': self.crop_name,
            'crop_icon': self.crop_icon,
            'disease_name': self.disease_name,
            'severity': self.severity,
            'confidence': self.confidence,
            'description': self.description,
            'symptoms': json.loads(self.symptoms) if self.symptoms else [],
            'treatments': {
                'chemical': json.loads(self.treatments_chemical) if self.treatments_chemical else [],
                'organic': json.loads(self.treatments_organic) if self.treatments_organic else [],
                'prevention': json.loads(self.treatments_prevention) if self.treatments_prevention else []
            },
            'image_path': self.image_path,
            'source': self.source,
            'location': {'lat': self.latitude, 'lon': self.longitude} if self.latitude else None
        }

class DronePlan(db.Model):
    """Store drone spray/mapping plans"""
    __tablename__ = 'drone_plans'

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    name = db.Column(db.String(200))
    area_hectares = db.Column(db.Float)
    spray_type = db.Column(db.String(50))
    drone_model = db.Column(db.String(100))
    flight_time_minutes = db.Column(db.Integer)
    tank_loads = db.Column(db.Integer)
    chemical_liters = db.Column(db.Float)
    boundary_points = db.Column(db.Text)  # JSON array of lat/lng

    def to_dict(self):
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat(),
            'name': self.name,
            'area_hectares': self.area_hectares,
            'spray_type': self.spray_type,
            'drone_model': self.drone_model,
            'flight_time_minutes': self.flight_time_minutes,
            'tank_loads': self.tank_loads,
            'chemical_liters': self.chemical_liters,
            'boundary_points': json.loads(self.boundary_points) if self.boundary_points else []
        }

class DroneDevice(db.Model):
    """Registry of drones/companion devices that have connected, over
    WiFi or Bluetooth, so the user can see what's currently linked."""
    __tablename__ = 'drone_devices'

    id = db.Column(db.Integer, primary_key=True)
    device_id = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(200), default='Unknown Drone')
    connection_type = db.Column(db.String(20), default='wifi')  # 'wifi' or 'bluetooth'
    first_seen = db.Column(db.DateTime, default=datetime.utcnow)
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)
    images_sent = db.Column(db.Integer, default=0)

    def to_dict(self):
        return {
            'id': self.id,
            'device_id': self.device_id,
            'name': self.name,
            'connection_type': self.connection_type,
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'images_sent': self.images_sent
        }


class UserProfile(db.Model):
    """Farmer profile"""
    __tablename__ = 'user_profiles'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200))
    phone = db.Column(db.String(20))
    village = db.Column(db.String(200))
    district = db.Column(db.String(200))
    state = db.Column(db.String(100))
    farm_size_acres = db.Column(db.Float)
    experience_years = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'phone': self.phone,
            'village': self.village,
            'district': self.district,
            'state': self.state,
            'farm_size_acres': self.farm_size_acres,
            'experience_years': self.experience_years
        }

class FarmField(db.Model):
    """Individual farm fields"""
    __tablename__ = 'farm_fields'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    area_acres = db.Column(db.Float, nullable=False)
    crop = db.Column(db.String(100))
    soil_type = db.Column(db.String(50))
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'area_acres': self.area_acres,
            'crop': self.crop,
            'soil_type': self.soil_type,
            'notes': self.notes
        }

class FarmingTask(db.Model):
    """Calendar tasks"""
    __tablename__ = 'farming_tasks'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    task_date = db.Column(db.Date, nullable=False)
    task_type = db.Column(db.String(50))
    field_name = db.Column(db.String(200))
    notes = db.Column(db.Text)
    completed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'task_date': self.task_date.isoformat(),
            'task_type': self.task_type,
            'field_name': self.field_name,
            'notes': self.notes,
            'completed': self.completed
        }
