"""
CropGuard AI Pro v4.0 - AI Precision Agriculture Platform
Phase 0-5: Ensemble AI + Drone + IoT + Digital Twin + Farm OS

New additions over v3.0:
  Phase 0: EfficientNetV2 + Grad-CAM ensemble pipeline
  Phase 1: Predictive disease alerts (7-10 days before symptoms)
  Phase 2: AI Farm Assistant (Gemini/GPT), Weather Intelligence
  Phase 3: MAVLink drone integration, Precision spray, IoT soil sensors
  Phase 4: NDVI, Yield prediction, Weed/Pest detection
  Phase 5: Farm Memory, Carbon tracker, Marketplace
"""
import os
import io
import json
import uuid
import base64
from datetime import datetime, date, timedelta

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
load_dotenv()

from config import config
from models import db, ScanHistory, DronePlan, UserProfile, FarmField, FarmingTask, DroneDevice, \
    FleetDrone, MaintenanceLog, MissionSchedule, NoFlyZone, MissionShare, FarmZone, NoFlyZone, MissionShare
from disease_database import CROP_DISEASE_DB, MARKET_DATA, MANDI_PREMIUMS, CROP_ICONS, CROP_NAMES
from yolo_detector import get_detector
from image_utils import normalize_image_bytes, normalize_uploaded_file, ImageDecodeError
from discovery import start_discovery_beacon, DiscoveryBeacon

# ── Drone Hub Extension: must be imported before db.create_all() runs
# (below) so its new tables (dronehub_missions, dronehub_telemetry_log,
# dronehub_reports) actually get created. ────────────────────────────────
from drone_extension.routes_drone import drone_bp
from drone_extension.models_drone_ext import DroneHubMission, DroneHubTelemetryLog, DroneHubReport

# ── Phase 0-1: AI Ensemble Pipeline ─────────────────────────────────────────
try:
    from ai_pipeline.ensemble import get_pipeline
    _ensemble_pipeline = None   # lazy-loaded on first use
except ImportError as _e:
    print(f"⚠️  Ensemble pipeline not available: {_e}")
    get_pipeline = None
    _ensemble_pipeline = None

# ── Phase 1: Predictive Disease ──────────────────────────────────────────────
try:
    from modules.predictive_disease import get_predictive_engine
except ImportError:
    get_predictive_engine = None

# ── Phase 2: AI Assistant + Weather ──────────────────────────────────────────
try:
    from modules.ai_assistant import get_assistant
except ImportError:
    get_assistant = None

try:
    from modules.weather_intelligence import get_weather_intelligence
except ImportError:
    get_weather_intelligence = None

# ── Phase 3: Drone + Mission Planner + Sensors ───────────────────────────────
try:
    from modules.drone.mission_planner import get_mission_planner
except ImportError:
    get_mission_planner = None

try:
    from modules.sensors.soil_sensor import get_soil_hub
except ImportError:
    get_soil_hub = None

# ── Drone Command Center: telemetry sim, emergency, copilot, analytics ──────
try:
    from modules.drone import command_center as dcc
except ImportError:
    dcc = None

try:
    from modules.drone import command_center_ext as dcx
except ImportError:
    dcx = None

try:
    from modules.sensors.ndvi_calculator import calculate_ndvi
except ImportError:
    calculate_ndvi = None

# ── Phase 4: Yield + Weed + Pest ─────────────────────────────────────────────
try:
    from modules.yield_predictor import get_yield_predictor
except ImportError:
    get_yield_predictor = None

try:
    from ai_pipeline.weed_detector import get_weed_detector
except ImportError:
    get_weed_detector = None

try:
    from ai_pipeline.pest_detector import get_pest_detector
except ImportError:
    get_pest_detector = None

# ── Phase 5: Farm Memory + Carbon + Marketplace ───────────────────────────────
try:
    from modules.farm_memory import get_farm_memory
except ImportError:
    get_farm_memory = None

try:
    from modules.carbon_tracker import get_carbon_tracker
except ImportError:
    get_carbon_tracker = None

try:
    from modules.marketplace import get_marketplace
except ImportError:
    get_marketplace = None

# Load environment variables


# Create Flask app
def create_app(config_name='development'):
    app = Flask(__name__, static_folder='static', template_folder='templates')
    app.config.from_object(config[config_name])

    # Initialize extensions
    CORS(app)
    db.init_app(app)
    app.register_blueprint(drone_bp)

    # Make sure required folders exist (fresh clones won't have empty dirs
    # from git, and uploads/model loading will fail without them)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.dirname(app.config['MODEL_PATH']) or '.', exist_ok=True)

    # Create tables
    with app.app_context():
        db.create_all()

    return app

app = create_app(os.environ.get('FLASK_ENV', 'development'))

# Initialize detector
detector = get_detector(
    model_path=app.config.get('MODEL_PATH'),
    conf_threshold=app.config.get('CONFIDENCE_THRESHOLD', 0.25),
    iou_threshold=app.config.get('IOU_THRESHOLD', 0.45),
    model_path_v2=app.config.get('MODEL_PATH_V2'),
    model_path_v3=app.config.get('MODEL_PATH_V3')
)

# Start the WiFi auto-discovery beacon so any drone/companion app on the
# same network can find this server without the user typing an IP address.
if app.config.get('ENABLE_WIFI_DISCOVERY', True):
    _discovery_port = int(os.environ.get('PORT', 5000))
    start_discovery_beacon(http_port=_discovery_port)

# In-memory heartbeat cache: device_id -> last-seen datetime.
_recent_devices = {}

# ===================== UTILITIES =====================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def save_uploaded_file(file):
    """Save an uploaded file, converting ANY image format (JPEG, PNG, WEBP,
    BMP, TIFF, GIF, HEIC/HEIF, RAW, etc.) into a normalized JPEG the model
    can always read. Returns (filepath, unique_name) or (None, None)."""
    if not file or not file.filename:
        return None, None
    try:
        filepath, unique_name = normalize_uploaded_file(file, app.config['UPLOAD_FOLDER'])
        return filepath, unique_name
    except ImageDecodeError:
        return None, None

def touch_drone_device(device_id, name=None, connection_type='wifi'):
    """Record/update a heartbeat for a connected drone/companion device."""
    if not device_id:
        return
    device = DroneDevice.query.filter_by(device_id=device_id).first()
    now = datetime.utcnow()
    if device:
        device.last_seen = now
        device.images_sent = (device.images_sent or 0) + 1
        if name:
            device.name = name
    else:
        device = DroneDevice(
            device_id=device_id,
            name=name or f'Drone {device_id[:8]}',
            connection_type=connection_type,
            first_seen=now,
            last_seen=now,
            images_sent=1
        )
        db.session.add(device)
    db.session.commit()
    _recent_devices[device_id] = now

def check_drone_api_key():
    """Optional shared-secret check for the drone endpoints. If DRONE_API_KEY
    is unset (the default), any device can connect - lowest friction for
    getting started. Set it in .env once you want to lock the endpoint down."""
    required_key = app.config.get('DRONE_API_KEY')
    if not required_key:
        return True
    provided = request.headers.get('X-Drone-Api-Key') or request.args.get('api_key')
    return provided == required_key

def get_disease_info(crop_key, disease_name):
    """Get disease information from database"""
    crop_data = CROP_DISEASE_DB.get(crop_key, {})
    diseases = crop_data.get('diseases', {})

    # Try exact match first
    if disease_name in diseases:
        return diseases[disease_name]

    # Try fuzzy match
    for name, info in diseases.items():
        if disease_name.lower() in name.lower() or name.lower() in disease_name.lower():
            return info

    # Return generic info
    return {
        'severity': 'Unknown',
        'description': f'Detected {disease_name} on {crop_data.get("name", "Unknown crop")}.',
        'symptoms': ['Symptoms not available in database'],
        'treatments': {
            'chemical': ['Consult local agricultural extension'],
            'organic': ['Neem oil spray', 'Compost tea application'],
            'prevention': ['Crop rotation', 'Field sanitation']
        }
    }

# ===================== ROUTES =====================

@app.route('/')
def index():
    """Main application page"""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """API health check"""
    return jsonify({
        'status': 'healthy',
        'version': '3.0.0',
        'model_loaded': detector.model_loaded,
        'device': detector.device,
        'timestamp': datetime.utcnow().isoformat()
    })

# ===================== SCAN API =====================

@app.route('/api/scan', methods=['POST'])
def scan_image():
    """Upload and analyze crop image"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filepath, unique_name = save_uploaded_file(file)
    if not filepath:
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Run detection
        crop_group = request.form.get('crop_group', 'auto')
        result = detector.detect(filepath, crop_group=crop_group)

        if not result['success']:
            return jsonify({'error': result.get('error', 'Detection failed')}), 500

        # Get top detection
        detections = result.get('detections', [])
        if not detections:
            return jsonify({
                'success': True,
                'message': 'No diseases detected - plant appears healthy',
                'crop': {'name': 'Unknown', 'icon': '🌱'},
                'disease': {'name': 'Healthy', 'severity': 'Healthy', 'confidence': 0},
                'detections': [],
                'annotated_image': None
            })

        top = detections[0]
        crop_key = top.get('crop', 'unknown').lower()
        disease_name = top.get('disease', 'Unknown')

        # Get disease info from database
        disease_info = get_disease_info(crop_key, disease_name)
        crop_data = CROP_DISEASE_DB.get(crop_key, {'name': crop_key.title(), 'icon': '🌱'})

        # Prefer severity computed directly by the detector (e.g. fallback color
        # analysis already classifies High/Medium/Low/Healthy) over the disease
        # database lookup, which only knows about real, named diseases.
        severity = top.get('severity') or disease_info.get('severity', 'Unknown')

        # Build response
        response = {
            'success': True,
            'crop': {
                'name': crop_data.get('name', crop_key.title()),
                'icon': crop_data.get('icon', '🌱'),
                'key': crop_key
            },
            'disease': {
                'name': disease_name,
                'severity': severity,
                'confidence': top.get('confidence', 0),
                'description': disease_info.get('description', ''),
                'symptoms': disease_info.get('symptoms', []),
                'treatments': disease_info.get('treatments', {})
            },
            'detections': detections,
            'annotated_image': f"/uploads/annotated_{unique_name}" if os.path.exists(
                os.path.join(app.config['UPLOAD_FOLDER'], f"annotated_{unique_name}")
            ) else None,
            'model_used': result.get('model', 'Unknown'),
            'pixel_grid': result.get('pixel_grid', []),
            'analysis': result.get('analysis', {})
        }

        # Save to history
        scan_record = ScanHistory(
            crop_name=crop_data.get('name', crop_key.title()),
            crop_icon=crop_data.get('icon', '🌱'),
            disease_name=disease_name,
            severity=severity,
            confidence=top.get('confidence', 0),
            description=disease_info.get('description', ''),
            symptoms=json.dumps(disease_info.get('symptoms', [])),
            treatments_chemical=json.dumps(disease_info.get('treatments', {}).get('chemical', [])),
            treatments_organic=json.dumps(disease_info.get('treatments', {}).get('organic', [])),
            treatments_prevention=json.dumps(disease_info.get('treatments', {}).get('prevention', [])),
            image_path=unique_name,
            source=result.get('model', 'Unknown')
        )
        db.session.add(scan_record)
        db.session.commit()

        response['scan_id'] = scan_record.id

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===================== HISTORY API =====================

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get scan history"""
    limit = request.args.get('limit', 50, type=int)
    scans = ScanHistory.query.order_by(ScanHistory.timestamp.desc()).limit(limit).all()
    return jsonify({'scans': [s.to_dict() for s in scans]})

@app.route('/api/history/<int:scan_id>', methods=['DELETE'])
def delete_scan(scan_id):
    """Delete a scan record"""
    scan = ScanHistory.query.get_or_404(scan_id)
    db.session.delete(scan)
    db.session.commit()
    return jsonify({'success': True, 'message': 'Scan deleted'})

@app.route('/api/history/export', methods=['GET'])
def export_history():
    """Export history as CSV"""
    scans = ScanHistory.query.order_by(ScanHistory.timestamp.desc()).all()

    import csv
    import io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Date', 'Crop', 'Disease', 'Severity', 'Confidence%', 'Source'])

    for scan in scans:
        writer.writerow([
            scan.timestamp.strftime('%Y-%m-%d %H:%M'),
            scan.crop_name,
            scan.disease_name,
            scan.severity,
            scan.confidence,
            scan.source
        ])

    output.seek(0)
    from flask import Response
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename=cropguard_history_{date.today()}.csv'}
    )

# ===================== MARKET API =====================

@app.route('/api/market', methods=['GET'])
def get_market_prices():
    """Get crop market prices"""
    mandi = request.args.get('mandi', 'national')
    sort_by = request.args.get('sort', 'name')
    currency = request.args.get('currency', 'INR')

    premium = MANDI_PREMIUMS.get(mandi, 1.0)

    # Currency conversion
    rates = {'USD': 0.012, 'INR': 1, 'EUR': 0.011, 'GBP': 0.0095}
    rate = rates.get(currency, 1)
    symbols = {'USD': '$', 'INR': 'Rs', 'EUR': '€', 'GBP': '£'}
    sym = symbols.get(currency, 'Rs')

    items = []
    for key, data in MARKET_DATA.items():
        items.append({
            'key': key,
            'name': CROP_NAMES.get(key, key.title()),
            'icon': CROP_ICONS.get(key, '🌱'),
            'price': round(data['price'] * rate * premium),
            'change': data['change'],
            'currency': sym
        })

    # Sort
    if sort_by == 'price-high':
        items.sort(key=lambda x: x['price'], reverse=True)
    elif sort_by == 'price-low':
        items.sort(key=lambda x: x['price'])
    elif sort_by == 'change':
        items.sort(key=lambda x: x['change'], reverse=True)

    return jsonify({'items': items, 'mandi': mandi, 'currency': currency})

# ===================== FERTILIZER API =====================

@app.route('/api/fertilizer', methods=['POST'])
def calculate_fertilizer():
    """Calculate fertilizer requirements"""
    data = request.get_json()

    crop = data.get('crop')
    area = float(data.get('area', 0))
    stage = data.get('stage', 'vegetative')
    soil = data.get('soil', 'loamy')
    fert_type = data.get('fertilizer_type', 'chemical')
    soil_n = float(data.get('soil_n', 0) or 0)
    soil_p = float(data.get('soil_p', 0) or 0)

    if not crop or area <= 0:
        return jsonify({'error': 'Crop and area required'}), 400

    # Crop NPK requirements (kg/ha)
    crop_needs = {
        'wheat': {'N': 120, 'P': 60, 'K': 40},
        'rice': {'N': 100, 'P': 50, 'K': 50},
        'corn': {'N': 150, 'P': 75, 'K': 60},
        'soybean': {'N': 20, 'P': 60, 'K': 40},
        'cotton': {'N': 120, 'P': 60, 'K': 60},
        'tomato': {'N': 100, 'P': 80, 'K': 120},
        'potato': {'N': 120, 'P': 80, 'K': 150},
        'sugarcane': {'N': 150, 'P': 60, 'K': 80},
        'mustard': {'N': 80, 'P': 40, 'K': 30},
        'groundnut': {'N': 20, 'P': 50, 'K': 40}
    }

    stage_mult = {'seedling': 0.3, 'vegetative': 0.6, 'flowering': 0.9, 'fruiting': 1.0}
    soil_mult = {'loamy': 1.0, 'clay': 0.9, 'sandy': 1.2, 'silty': 1.0, 'peaty': 0.8, 'chalky': 1.1}

    base = crop_needs.get(crop, {'N': 100, 'P': 50, 'K': 50})
    sm = stage_mult.get(stage, 0.6)
    soil_m = soil_mult.get(soil, 1.0)

    # Adjust for soil test values
    n_adj = max(0.3, 1 - soil_n / 200) if soil_n > 0 else 1
    p_adj = max(0.3, 1 - soil_p / 100) if soil_p > 0 else 1

    N = round(base['N'] * area * sm * soil_m * n_adj)
    P = round(base['P'] * area * sm * soil_m * p_adj)
    K = round(base['K'] * area * sm * soil_m)

    result = {
        'crop': crop,
        'area': area,
        'requirements': {'N': N, 'P': P, 'K': K},
        'recommendations': {}
    }

    if fert_type in ['chemical', 'balanced']:
        result['recommendations']['chemical'] = {
            'urea_kg': round(N * 2.17),
            'dap_kg': round(P * 5.43),
            'mop_kg': round(K * 1.67)
        }

    if fert_type in ['organic', 'balanced']:
        result['recommendations']['organic'] = {
            'fym_kg': round(area * 5000 * sm),
            'vermicompost_kg': round(area * 1000 * sm)
        }

    if fert_type in ['bio', 'balanced']:
        result['recommendations']['bio'] = {
            'azotobacter_kg': round(area * 2),
            'psb_kg': round(area * 2)
        }

    return jsonify(result)

# ===================== IRRIGATION API =====================

@app.route('/api/irrigation', methods=['POST'])
def check_irrigation():
    """Check irrigation needs"""
    data = request.get_json()

    crop = data.get('crop')
    moisture = float(data.get('moisture', 0))
    days_since = int(data.get('days_since', 0))
    area = float(data.get('area', 0))
    method = data.get('method', 'drip')
    expected_rain = float(data.get('rain', 0))

    if not crop or moisture < 0:
        return jsonify({'error': 'Crop and moisture required'}), 400

    # Optimal moisture ranges
    optimal = {
        'wheat': {'min': 50, 'max': 70},
        'rice': {'min': 80, 'max': 100},
        'corn': {'min': 55, 'max': 75},
        'tomato': {'min': 60, 'max': 80},
        'potato': {'min': 65, 'max': 85},
        'cotton': {'min': 50, 'max': 70},
        'sugarcane': {'min': 60, 'max': 80},
        'onion': {'min': 55, 'max': 75},
        'groundnut': {'min': 50, 'max': 70}
    }

    range_data = optimal.get(crop, {'min': 55, 'max': 75})
    optimal_val = (range_data['min'] + range_data['max']) / 2

    method_eff = {'flood': 0.4, 'sprinkler': 0.7, 'drip': 0.9, 'furrow': 0.5, 'raingun': 0.6}
    eff = method_eff.get(method, 0.7)

    if moisture < range_data['min']:
        status = 'IRRIGATION NEEDED NOW'
        status_color = '#c1121f'
        water_needed = round(area * 25 * (range_data['min'] - moisture) / 10 / eff)
    elif moisture < optimal_val:
        status = 'IRRIGATE SOON'
        status_color = '#f4a261'
        water_needed = round(area * 15 * (optimal_val - moisture) / 10 / eff)
    elif moisture > range_data['max']:
        status = 'TOO WET'
        status_color = '#457b9d'
        water_needed = 0
    else:
        status = 'OPTIMAL'
        status_color = '#1a5c3a'
        water_needed = 0

    method_tips = {
        'drip': 'Drip irrigation saves 40-60% water compared to flood irrigation.',
        'sprinkler': 'Sprinkler is efficient for field crops. Best in early morning or evening.',
        'flood': 'Flood irrigation is water-intensive. Consider switching to drip for water savings.',
        'furrow': 'Furrow irrigation works well for row crops. Maintain proper slope.',
        'raingun': 'Rain gun covers large areas but has higher evaporation losses.'
    }

    return jsonify({
        'status': status,
        'status_color': status_color,
        'moisture': moisture,
        'optimal': optimal_val,
        'range': range_data,
        'water_needed_liters': water_needed,
        'method': method,
        'method_tip': method_tips.get(method, ''),
        'expected_rain': expected_rain,
        'days_since_irrigation': days_since
    })

# ===================== YIELD CALCULATOR API =====================

@app.route('/api/yield', methods=['POST'])
def calculate_yield():
    """Calculate yield and profit"""
    data = request.get_json()

    crop = data.get('crop')
    area = float(data.get('area', 0))
    yield_per_acre = float(data.get('yield_per_acre', 0))
    price = float(data.get('price', 0))
    cost = float(data.get('cost', 0))
    irrigation = data.get('irrigation', 'rainfed')

    if not all([crop, area, yield_per_acre, price, cost]):
        return jsonify({'error': 'All fields required'}), 400

    irr_costs = {'rainfed': 0, 'canal': 2000, 'tubewell': 5000, 'drip': 8000}
    total_irr = irr_costs.get(irrigation, 0) * area
    total_yield = area * yield_per_acre
    gross = total_yield * price
    total_cost = (cost * area) + total_irr
    net = gross - total_cost
    roi = round((net / total_cost) * 100, 1) if total_cost > 0 else 0

    return jsonify({
        'crop': crop,
        'area': area,
        'total_yield': total_yield,
        'gross_income': gross,
        'total_cost': total_cost,
        'net_profit': net,
        'profit_per_acre': round(net / area, 2),
        'roi_percent': roi,
        'irrigation_cost': total_irr,
        'irrigation_type': irrigation
    })

# ===================== PROFILE API =====================

@app.route('/api/profile', methods=['GET', 'POST'])
def profile():
    """Get or update user profile"""
    if request.method == 'GET':
        prof = UserProfile.query.first()
        if prof:
            return jsonify(prof.to_dict())
        return jsonify({})

    data = request.get_json()
    prof = UserProfile.query.first()

    if not prof:
        prof = UserProfile()
        db.session.add(prof)

    prof.name = data.get('name', prof.name)
    prof.phone = data.get('phone', prof.phone)
    prof.village = data.get('village', prof.village)
    prof.district = data.get('district', prof.district)
    prof.state = data.get('state', prof.state)
    prof.farm_size_acres = data.get('farm_size', prof.farm_size_acres)
    prof.experience_years = data.get('experience', prof.experience_years)

    db.session.commit()
    return jsonify({'success': True, 'profile': prof.to_dict()})

# ===================== FIELDS API =====================

@app.route('/api/fields', methods=['GET', 'POST'])
def fields():
    """Get or add farm fields"""
    if request.method == 'GET':
        fields = FarmField.query.order_by(FarmField.created_at.desc()).all()
        return jsonify({'fields': [f.to_dict() for f in fields]})

    data = request.get_json()
    field = FarmField(
        name=data.get('name'),
        area_acres=data.get('area'),
        crop=data.get('crop'),
        soil_type=data.get('soil'),
        notes=data.get('notes')
    )
    db.session.add(field)
    db.session.commit()
    return jsonify({'success': True, 'field': field.to_dict()})

@app.route('/api/fields/<int:field_id>', methods=['DELETE'])
def delete_field(field_id):
    """Delete a field"""
    field = FarmField.query.get_or_404(field_id)
    db.session.delete(field)
    db.session.commit()
    return jsonify({'success': True})

# ===================== TASKS API =====================

@app.route('/api/tasks', methods=['GET', 'POST'])
def tasks():
    """Get or add farming tasks"""
    if request.method == 'GET':
        tasks = FarmingTask.query.order_by(FarmingTask.task_date).all()
        return jsonify({'tasks': [t.to_dict() for t in tasks]})

    data = request.get_json()
    task = FarmingTask(
        name=data.get('name'),
        task_date=datetime.strptime(data.get('date'), '%Y-%m-%d').date(),
        task_type=data.get('type'),
        field_name=data.get('field'),
        notes=data.get('notes')
    )
    db.session.add(task)
    db.session.commit()
    return jsonify({'success': True, 'task': task.to_dict()})

@app.route('/api/tasks/<int:task_id>', methods=['DELETE', 'PATCH'])
def manage_task(task_id):
    """Delete or update a task"""
    task = FarmingTask.query.get_or_404(task_id)

    if request.method == 'DELETE':
        db.session.delete(task)
        db.session.commit()
        return jsonify({'success': True})

    data = request.get_json()
    task.completed = data.get('completed', task.completed)
    db.session.commit()
    return jsonify({'success': True, 'task': task.to_dict()})

# ===================== DRONE API =====================

@app.route('/api/drone/plans', methods=['GET', 'POST'])
def drone_plans():
    """Get or create drone plans"""
    if request.method == 'GET':
        plans = DronePlan.query.order_by(DronePlan.created_at.desc()).all()
        return jsonify({'plans': [p.to_dict() for p in plans]})

    data = request.get_json()
    plan = DronePlan(
        name=data.get('name'),
        area_hectares=data.get('area'),
        spray_type=data.get('spray_type'),
        drone_model=data.get('drone_model'),
        flight_time_minutes=data.get('flight_time'),
        tank_loads=data.get('tank_loads'),
        chemical_liters=data.get('chemical'),
        boundary_points=json.dumps(data.get('boundary', []))
    )
    db.session.add(plan)
    db.session.commit()
    return jsonify({'success': True, 'plan': plan.to_dict()})

# ===================== LIVE DRONE CONNECTION (WiFi + Bluetooth bridge) =====================
# This is the actual "drone talks straight to the model" pipeline. It's
# deliberately generic (plain HTTP POST) so it works with ANY drone brand -
# a DJI/companion app, a Raspberry Pi on a custom drone, or the Bluetooth
# bridge script (bluetooth_bridge.py) - as long as it can POST bytes here.

@app.route('/api/drone/connect-info', methods=['GET'])
def drone_connect_info():
    """Everything a drone/companion app needs to find and use this server:
    the LAN URL to POST images to, plus a QR code for quick pairing from a
    phone-based companion app."""
    port = int(os.environ.get('PORT', 5000))
    beacon = DiscoveryBeacon(http_port=port)
    local_ip = beacon.get_local_ip()
    upload_url = f"http://{local_ip}:{port}/api/drone/upload"

    qr_base64 = None
    try:
        import qrcode
        qr_img = qrcode.make(upload_url)
        buf = io.BytesIO()
        qr_img.save(buf, format='PNG')
        qr_base64 = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')
    except ImportError:
        pass  # qrcode not installed - URL text still works fine without it

    return jsonify({
        'wifi': {
            'upload_url': upload_url,
            'local_ip': local_ip,
            'port': port,
            'qr_code': qr_base64,
            'discovery_udp_port': 41234,
            'instructions': 'Any device on the same WiFi network can POST an image (any format) '
                             'as multipart/form-data field "image" to upload_url and get results back.'
        },
        'bluetooth': {
            'bridge_script': 'bluetooth_bridge.py',
            'instructions': 'Run "python bluetooth_bridge.py" on this machine (needs Bluetooth '
                             'hardware). It accepts images over classic Bluetooth (RFCOMM/SPP) or '
                             'the browser can pair directly via Web Bluetooth from the dashboard.'
        },
        'api_key_required': bool(app.config.get('DRONE_API_KEY'))
    })

@app.route('/api/drone/upload', methods=['POST'])
def drone_upload():
    """Dedicated ingestion endpoint for drone images.

    Accepts, in order of preference:
      1. multipart/form-data with a file field named 'image' (any format)
      2. application/json with a base64 string in 'image_base64'
      3. a raw binary body (any image bytes, any format) - simplest for
         microcontroller / embedded drone clients that can't build multipart

    Optional fields (form, query string, or JSON body): device_id, device_name,
    lat, lon, alt, battery, crop_group.
    """
    if not check_drone_api_key():
        return jsonify({'error': 'Invalid or missing API key'}), 401

    raw_bytes = None
    meta = {}

    if 'image' in request.files:
        f = request.files['image']
        raw_bytes = f.read()
        meta['filename_hint'] = os.path.splitext(f.filename or 'drone')[0]
        meta.update(request.form.to_dict())
    elif request.is_json:
        body = request.get_json(silent=True) or {}
        b64 = body.get('image_base64', '')
        if ',' in b64 and b64.strip().startswith('data:'):
            b64 = b64.split(',', 1)[1]
        try:
            raw_bytes = base64.b64decode(b64)
        except Exception:
            return jsonify({'error': 'Invalid base64 image data'}), 400
        meta.update(body)
    else:
        raw_bytes = request.get_data()
        meta.update(request.args.to_dict())

    if not raw_bytes:
        return jsonify({'error': 'No image data received'}), 400

    try:
        filepath, unique_name = normalize_image_bytes(
            raw_bytes, app.config['UPLOAD_FOLDER'],
            filename_hint=meta.get('filename_hint', 'drone')
        )
    except ImageDecodeError as e:
        return jsonify({'error': f'Unsupported or corrupted image: {e}'}), 400

    device_id = meta.get('device_id', 'unknown-drone')
    touch_drone_device(device_id, name=meta.get('device_name'), connection_type=meta.get('connection_type', 'wifi'))

    crop_group = meta.get('crop_group', 'auto')
    result = detector.detect(filepath, crop_group=crop_group)

    if not result.get('success'):
        return jsonify({'error': result.get('error', 'Detection failed')}), 500

    detections = result.get('detections', [])
    lat = meta.get('lat')
    lon = meta.get('lon')

    if not detections:
        response = {
            'success': True,
            'device_id': device_id,
            'message': 'No diseases detected - plant appears healthy',
            'crop': {'name': 'Unknown', 'icon': '🌱'},
            'disease': {'name': 'Healthy', 'severity': 'Healthy', 'confidence': 0},
            'detections': [],
            'image_url': f"/uploads/{unique_name}"
        }
        scan_record = ScanHistory(
            crop_name='Unknown', disease_name='Healthy', severity='Healthy',
            confidence=0, image_path=unique_name, source=f'Drone:{device_id}',
            latitude=float(lat) if lat else None, longitude=float(lon) if lon else None
        )
    else:
        top = detections[0]
        crop_key = top.get('crop', 'unknown').lower()
        disease_name = top.get('disease', 'Unknown')
        disease_info = get_disease_info(crop_key, disease_name)
        crop_data = CROP_DISEASE_DB.get(crop_key, {'name': crop_key.title(), 'icon': '🌱'})
        severity = top.get('severity') or disease_info.get('severity', 'Unknown')

        response = {
            'success': True,
            'device_id': device_id,
            'crop': {'name': crop_data.get('name', crop_key.title()), 'icon': crop_data.get('icon', '🌱'), 'key': crop_key},
            'disease': {
                'name': disease_name, 'severity': severity, 'confidence': top.get('confidence', 0),
                'description': disease_info.get('description', ''),
                'symptoms': disease_info.get('symptoms', []),
                'treatments': disease_info.get('treatments', {})
            },
            'detections': detections,
            'image_url': f"/uploads/{unique_name}",
            'location': {'lat': lat, 'lon': lon} if lat and lon else None
        }
        scan_record = ScanHistory(
            crop_name=crop_data.get('name', crop_key.title()), crop_icon=crop_data.get('icon', '🌱'),
            disease_name=disease_name, severity=severity, confidence=top.get('confidence', 0),
            description=disease_info.get('description', ''),
            symptoms=json.dumps(disease_info.get('symptoms', [])),
            treatments_chemical=json.dumps(disease_info.get('treatments', {}).get('chemical', [])),
            treatments_organic=json.dumps(disease_info.get('treatments', {}).get('organic', [])),
            treatments_prevention=json.dumps(disease_info.get('treatments', {}).get('prevention', [])),
            image_path=unique_name, source=f'Drone:{device_id}',
            latitude=float(lat) if lat else None, longitude=float(lon) if lon else None
        )

    db.session.add(scan_record)
    db.session.commit()
    response['scan_id'] = scan_record.id
    return jsonify(response)

@app.route('/api/drone/devices', methods=['GET'])
def drone_devices():
    """List drones/companion devices that have connected recently."""
    timeout = app.config.get('DRONE_DEVICE_TIMEOUT_SECONDS', 30)
    cutoff = datetime.utcnow() - timedelta(seconds=timeout * 20)  # keep a longer visible history
    devices = DroneDevice.query.filter(DroneDevice.last_seen >= cutoff).order_by(DroneDevice.last_seen.desc()).all()
    live_cutoff = datetime.utcnow() - timedelta(seconds=timeout)
    result = []
    for d in devices:
        item = d.to_dict()
        item['online'] = d.last_seen >= live_cutoff
        result.append(item)
    return jsonify({'devices': result})

@app.route('/api/drone/latest', methods=['GET'])
def drone_latest():
    """Most recent drone-sourced scan, for the dashboard to poll and show
    results the instant a drone image comes in - no manual refresh needed."""
    scan = ScanHistory.query.filter(ScanHistory.source.like('Drone:%')).order_by(ScanHistory.timestamp.desc()).first()
    if not scan:
        return jsonify({'scan': None})
    return jsonify({'scan': scan.to_dict()})

# ===================== UPLOADS =====================

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ===================== TRAINING API =====================

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train YOLOv8 model on custom dataset"""
    data = request.get_json()
    data_yaml = data.get('data_yaml')
    epochs = data.get('epochs', 100)
    imgsz = data.get('imgsz', 640)

    if not data_yaml:
        return jsonify({'error': 'data_yaml path required'}), 400

    try:
        results = detector.train(
            data_yaml=data_yaml,
            epochs=epochs,
            imgsz=imgsz
        )
        return jsonify({
            'success': True,
            'message': 'Training completed',
            'results': str(results)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ======================================================================
# PHASE 0-1: ENSEMBLE AI SCAN (replaces /api/scan with richer output)
# ======================================================================

@app.route('/api/scan/ensemble', methods=['POST'])
def scan_ensemble():
    """
    Phase 0+1: Full ensemble AI scan.
    Uses YOLOv8 + EfficientNetV2 + Swin Transformer + Grad-CAM.
    Returns a complete 'Farm Doctor Report'.
    """
    global _ensemble_pipeline

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filepath, unique_name = save_uploaded_file(file)
    if not filepath:
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Lazy-load ensemble pipeline
        if get_pipeline is not None:
            if _ensemble_pipeline is None:
                _ensemble_pipeline = get_pipeline(
                    yolo_model_path=app.config.get('MODEL_PATH'),
                    efficientnet_weights=app.config.get('EFFICIENTNET_WEIGHTS'),
                    swin_weights=app.config.get('SWIN_WEIGHTS'),
                )
            crop_hint = request.form.get('crop', 'auto')
            report = _ensemble_pipeline.analyze(filepath, crop_hint=crop_hint)
        else:
            # Fallback to YOLOv8 original scan
            return scan_image()

        # Save to history
        if report.get('success'):
            crop    = report.get('crop', {})
            disease = report.get('disease', {})
            impact  = report.get('impact', {})
            record  = ScanHistory(
                crop_name     = crop.get('name', 'Unknown'),
                crop_icon     = crop.get('icon', '🌱'),
                disease_name  = disease.get('name', 'Unknown'),
                severity      = disease.get('severity', 'Unknown'),
                confidence    = disease.get('confidence', 0),
                description   = disease.get('description', ''),
                symptoms      = json.dumps(disease.get('symptoms', [])),
                treatments_chemical  = json.dumps(report.get('treatments', {}).get('chemical', [])),
                treatments_organic   = json.dumps(report.get('treatments', {}).get('organic', [])),
                treatments_prevention= json.dumps(report.get('treatments', {}).get('prevention', [])),
                image_path    = unique_name,
                source        = 'ensemble_v4',
            )
            db.session.add(record)
            db.session.commit()
            report['scan_id'] = record.id

            # Auto-record in farm memory (if farm_id provided)
            farm_id = request.form.get('farm_id')
            if farm_id and get_farm_memory:
                mem = get_farm_memory()
                mem.record_disease(
                    farm_id=farm_id,
                    crop=crop.get('key', ''),
                    disease=disease.get('name', ''),
                    severity=disease.get('severity', ''),
                    affected_area_pct=impact.get('yield_loss_estimate_pct', 0),
                )

        return jsonify(report)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ======================================================================
# PHASE 1: PREDICTIVE DISEASE ALERTS
# ======================================================================

@app.route('/api/predict/disease', methods=['POST'])
def predict_disease():
    """
    Phase 1: Predict disease outbreak 7-10 days before symptoms appear.
    POST body: { crop, lat, lon, soil_moisture, historical_outbreaks }
    """
    if get_predictive_engine is None:
        return jsonify({'error': 'Predictive engine not available'}), 503

    data = request.get_json(silent=True) or {}
    crop = data.get('crop', 'tomato')
    lat  = data.get('lat', 20.5)
    lon  = data.get('lon', 78.9)

    # Fetch weather forecast for location
    forecast = []
    if get_weather_intelligence:
        try:
            wx = get_weather_intelligence()
            wx_data  = wx.get_forecast(lat, lon, days=7)
            forecast = wx_data.get('forecast_days', [])
        except Exception:
            pass

    if not forecast:
        # Use dummy forecast for demo
        forecast = [{"humidity_avg": 82, "temp_min": 18, "temp_max": 26,
                     "rainfall_mm": 5, "wind_kmh": 12}] * 7

    engine = get_predictive_engine()
    result = engine.assess_risk(
        crop=crop,
        weather_forecast=forecast,
        soil_moisture_pct=data.get('soil_moisture', 50),
        ndvi_anomaly=data.get('ndvi_anomaly', 0.0),
        historical_outbreaks=data.get('historical_outbreaks', []),
    )

    # Also get spray advice for today
    if forecast:
        result['spray_advice_today'] = engine.get_weather_based_spray_advice(forecast[0])

    return jsonify(result)


# ======================================================================
# PHASE 2: AI FARM ASSISTANT
# ======================================================================

@app.route('/api/assistant/ask', methods=['POST'])
def assistant_ask():
    """
    Phase 2: Ask the AI agricultural assistant a question.
    POST body: { question, context: { crop, disease, weather } }
    """
    if get_assistant is None:
        return jsonify({'error': 'AI assistant not available'}), 503

    data     = request.get_json(silent=True) or {}
    question = data.get('question', '')
    context  = data.get('context', {})

    if not question:
        return jsonify({'error': 'question is required'}), 400

    assistant = get_assistant()
    result    = assistant.ask(question, context)
    return jsonify(result)


@app.route('/api/assistant/clear', methods=['POST'])
def assistant_clear():
    """Clear assistant conversation history."""
    if get_assistant:
        get_assistant().clear_history()
    return jsonify({'success': True, 'message': 'Conversation cleared'})


# ======================================================================
# PHASE 2: WEATHER INTELLIGENCE
# ======================================================================

@app.route('/api/weather', methods=['GET'])
def get_weather():
    """
    Phase 2: Get 7-day weather forecast + farming interpretation.
    Query params: lat, lon
    """
    lat = request.args.get('lat', 20.5, type=float)
    lon = request.args.get('lon', 78.9, type=float)

    if get_weather_intelligence is None:
        return jsonify({'error': 'Weather module not available'}), 503

    wx     = get_weather_intelligence()
    result = wx.get_forecast(lat, lon)
    return jsonify(result)


@app.route('/api/weather/current', methods=['GET'])
def get_current_weather():
    lat = request.args.get('lat', 20.5, type=float)
    lon = request.args.get('lon', 78.9, type=float)

    if get_weather_intelligence is None:
        return jsonify({'error': 'Weather module not available'}), 503

    wx = get_weather_intelligence()
    return jsonify(wx.get_current_weather(lat, lon))


# ======================================================================
# PHASE 3: DRONE MISSION PLANNER
# ======================================================================

@app.route('/api/drone/plan/scan', methods=['POST'])
def plan_scan_mission():
    """
    Phase 3: Generate autonomous scanning mission for a field polygon.
    POST body: {
      field_polygon: [[lat,lon], ...],
      home_lat, home_lon,
      altitude_m, overlap_pct
    }
    """
    if get_mission_planner is None:
        return jsonify({'error': 'Mission planner not available'}), 503

    data    = request.get_json(silent=True) or {}
    polygon = data.get('field_polygon', [])
    if len(polygon) < 3:
        return jsonify({'error': 'field_polygon must have at least 3 points'}), 400

    # Convert [[lat,lon],...] to [(lat,lon),...]
    poly_tuples = [(p[0], p[1]) for p in polygon]

    planner = get_mission_planner()
    result  = planner.plan_scan_mission(
        field_polygon=poly_tuples,
        home_lat=data.get('home_lat', poly_tuples[0][0]),
        home_lon=data.get('home_lon', poly_tuples[0][1]),
    )
    return jsonify(result)


@app.route('/api/drone/plan/spray', methods=['POST'])
def plan_spray_mission():
    """
    Phase 3: Generate precision spray mission from disease GPS coordinates.
    POST body: {
      disease_coordinates: [{lat, lon, severity, disease_name}, ...],
      home_lat, home_lon
    }
    """
    if get_mission_planner is None:
        return jsonify({'error': 'Mission planner not available'}), 503

    data   = request.get_json(silent=True) or {}
    coords = data.get('disease_coordinates', [])
    if not coords:
        return jsonify({'error': 'disease_coordinates required'}), 400

    planner = get_mission_planner()
    result  = planner.plan_spray_mission(
        disease_coordinates=coords,
        home_lat=data.get('home_lat', 0.0),
        home_lon=data.get('home_lon', 0.0),
    )
    return jsonify(result)


# ======================================================================
# PHASE 3: IoT SOIL SENSORS
# ======================================================================

@app.route('/api/sensors/soil', methods=['GET'])
def get_soil_readings():
    """Phase 3: Get latest soil sensor readings."""
    if get_soil_hub is None:
        return jsonify({'error': 'Soil sensor module not available'}), 503

    sensor_id = request.args.get('sensor_id')
    hub    = get_soil_hub()
    result = hub.get_readings(sensor_id)
    return jsonify(result)


# ======================================================================
# PHASE 4: NDVI ANALYSIS
# ======================================================================

@app.route('/api/analyze/ndvi', methods=['POST'])
def analyze_ndvi():
    """
    Phase 4: Compute NDVI from uploaded drone image.
    Returns NDVI map, health zones, and colorized heatmap.
    """
    if calculate_ndvi is None:
        return jsonify({'error': 'NDVI module not available'}), 503

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    filepath, _ = save_uploaded_file(file)
    if not filepath:
        return jsonify({'error': 'Invalid file'}), 400

    result = calculate_ndvi(filepath)
    return jsonify(result)


# ======================================================================
# PHASE 4: WEED DETECTION
# ======================================================================

@app.route('/api/detect/weeds', methods=['POST'])
def detect_weeds():
    """Phase 4: Detect weeds in crop field image."""
    if get_weed_detector is None:
        return jsonify({'error': 'Weed detector not available'}), 503

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    filepath, _ = save_uploaded_file(file)
    if not filepath:
        return jsonify({'error': 'Invalid file'}), 400

    detector_w = get_weed_detector()
    result = detector_w.detect(filepath)
    return jsonify(result)


# ======================================================================
# PHASE 4: PEST DETECTION
# ======================================================================

@app.route('/api/detect/pests', methods=['POST'])
def detect_pests():
    """Phase 4: Detect insects, larvae, and eggs in crop image."""
    if get_pest_detector is None:
        return jsonify({'error': 'Pest detector not available'}), 503

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    filepath, _ = save_uploaded_file(file)
    if not filepath:
        return jsonify({'error': 'Invalid file'}), 400

    detector_p = get_pest_detector()
    result = detector_p.detect(filepath)
    return jsonify(result)


# ======================================================================
# PHASE 4: YIELD PREDICTION
# ======================================================================

@app.route('/api/predict/yield', methods=['POST'])
def predict_yield():
    """
    Phase 4: Predict crop yield.
    POST body: {
      crop, field_area_acres, ndvi_current, ndvi_trend,
      disease_yield_loss_pct, soil_nitrogen, rainfall_mm_7day,
      temp_avg_c, fertilizer_applied, irrigation_adequate, sowing_date
    }
    """
    if get_yield_predictor is None:
        return jsonify({'error': 'Yield predictor not available'}), 503

    data = request.get_json(silent=True) or {}
    if not data.get('crop') or not data.get('field_area_acres'):
        return jsonify({'error': 'crop and field_area_acres are required'}), 400

    predictor = get_yield_predictor()
    result = predictor.predict(
        crop=data['crop'],
        field_area_acres=float(data['field_area_acres']),
        sowing_date=data.get('sowing_date'),
        ndvi_current=data.get('ndvi_current', 0.5),
        ndvi_trend=data.get('ndvi_trend', 0.0),
        disease_yield_loss_pct=data.get('disease_yield_loss_pct', 0.0),
        soil_nitrogen=data.get('soil_nitrogen', 100.0),
        soil_moisture_pct=data.get('soil_moisture_pct', 50.0),
        rainfall_mm_7day=data.get('rainfall_mm_7day', 15.0),
        temp_avg_c=data.get('temp_avg_c', 25.0),
        fertilizer_applied=data.get('fertilizer_applied', True),
        irrigation_adequate=data.get('irrigation_adequate', True),
    )
    return jsonify(result)


# ======================================================================
# PHASE 5: FARM MEMORY
# ======================================================================

@app.route('/api/farm/<farm_id>/insights', methods=['GET'])
def farm_insights(farm_id: str):
    """Phase 5: Get AI insights from farm history."""
    if get_farm_memory is None:
        return jsonify({'error': 'Farm memory not available'}), 503

    crop   = request.args.get('crop')
    memory = get_farm_memory()
    result = memory.get_insights(farm_id, crop=crop)
    return jsonify(result)


@app.route('/api/farm/<farm_id>/history', methods=['GET'])
def farm_history(farm_id: str):
    """Phase 5: Get farm activity history."""
    if get_farm_memory is None:
        return jsonify({'error': 'Farm memory not available'}), 503

    months = request.args.get('months', 6, type=int)
    memory = get_farm_memory()
    result = memory.get_history_summary(farm_id, months=months)
    return jsonify(result)


@app.route('/api/farm/<farm_id>/record/spray', methods=['POST'])
def record_spray(farm_id: str):
    """Phase 5: Record a spray event in farm memory."""
    if get_farm_memory is None:
        return jsonify({'error': 'Farm memory not available'}), 503

    data   = request.get_json(silent=True) or {}
    memory = get_farm_memory()
    memory.record_spray(
        farm_id=farm_id,
        chemical=data.get('chemical', ''),
        dose=data.get('dose', ''),
        area_acres=data.get('area_acres', 0),
        reason=data.get('reason', ''),
        cost_inr=data.get('cost_inr', 0),
    )
    return jsonify({'success': True})


@app.route('/api/farm/<farm_id>/record/yield', methods=['POST'])
def record_yield(farm_id: str):
    """Phase 5: Record harvest yield."""
    if get_farm_memory is None:
        return jsonify({'error': 'Farm memory not available'}), 503

    data   = request.get_json(silent=True) or {}
    memory = get_farm_memory()
    memory.record_yield(
        farm_id=farm_id,
        crop=data.get('crop', ''),
        season=data.get('season', ''),
        yield_tonnes=data.get('yield_tonnes', 0),
        area_acres=data.get('area_acres', 1),
        market_value_inr=data.get('market_value_inr', 0),
    )
    return jsonify({'success': True})


# ======================================================================
# PHASE 5: CARBON TRACKER
# ======================================================================

@app.route('/api/carbon/score', methods=['POST'])
def carbon_score():
    """
    Phase 5: Calculate farm carbon score and sustainability rating.
    POST body: { crop, field_area_acres, precision_spray_pct, ... }
    """
    if get_carbon_tracker is None:
        return jsonify({'error': 'Carbon tracker not available'}), 503

    data    = request.get_json(silent=True) or {}
    tracker = get_carbon_tracker()
    result  = tracker.calculate_farm_carbon_score(
        crop=data.get('crop', 'rice'),
        field_area_acres=data.get('field_area_acres', 1.0),
        precision_spray_pct=data.get('precision_spray_pct', 0.0),
        organic_fertilizer_pct=data.get('organic_fertilizer_pct', 0.0),
        solar_powered_pumps=data.get('solar_powered_pumps', False),
        drip_irrigation=data.get('drip_irrigation', False),
        drone_electric=data.get('drone_electric', True),
        cover_crop=data.get('cover_crop', False),
        crop_rotation=data.get('crop_rotation', False),
    )
    return jsonify(result)


# ======================================================================
# PHASE 5: MARKETPLACE
# ======================================================================

@app.route('/api/marketplace/products', methods=['GET'])
def marketplace_products():
    """Phase 5: Get product recommendations for a disease/pest."""
    if get_marketplace is None:
        return jsonify({'error': 'Marketplace not available'}), 503

    disease = request.args.get('disease', '')
    pest    = request.args.get('pest', '')
    market  = get_marketplace()
    result  = market.get_product_recommendations(disease, pest)
    return jsonify(result)


@app.route('/api/marketplace/dealers', methods=['GET'])
def marketplace_dealers():
    """Phase 5: Get nearby dealer search links."""
    if get_marketplace is None:
        return jsonify({'error': 'Marketplace not available'}), 503

    lat  = request.args.get('lat', 20.5, type=float)
    lon  = request.args.get('lon', 78.9, type=float)
    prod = request.args.get('product', 'pesticide')
    market = get_marketplace()
    result = market.get_dealer_search_url(lat, lon, prod)
    return jsonify(result)


@app.route('/api/marketplace/schemes', methods=['GET'])
def government_schemes():
    """Phase 5: Get applicable government schemes."""
    if get_marketplace is None:
        return jsonify({'error': 'Marketplace not available'}), 503

    crop  = request.args.get('crop')
    state = request.args.get('state')
    market = get_marketplace()
    result = market.get_schemes(crop, state)
    return jsonify(result)


@app.route('/api/marketplace/prices', methods=['GET'])
def market_prices():
    """Phase 5: Get mandi price data sources for a crop."""
    if get_marketplace is None:
        return jsonify({'error': 'Marketplace not available'}), 503

    crop   = request.args.get('crop', 'tomato')
    market = get_marketplace()
    result = market.get_market_prices(crop)
    return jsonify(result)


# ======================================================================
# ENHANCED HEALTH CHECK (shows all module status)
# ======================================================================


# ======================================================================
# DRONE COMMAND CENTER
# ======================================================================

def _get_or_create_default_drone():
    d = FleetDrone.query.first()
    if not d:
        d = FleetDrone(drone_id='drone-1', name='CropGuard Drone 1', model='Generic Quadcopter',
                        status='IDLE', battery_pct=100.0, home_lat=20.5937, home_lon=78.9629)
        db.session.add(d)
        db.session.commit()
    return d


@app.route('/api/drone/fleet', methods=['GET', 'POST'])
def drone_fleet():
    """Fleet Management: list drones, or register a new one.
    A drone with no connection_string is simulated; set one (e.g.
    udp://:14540) to route it through the real MAVLink bridge later."""
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        d = FleetDrone(
            drone_id=data.get('drone_id') or f"drone-{FleetDrone.query.count() + 1}",
            name=data.get('name', 'CropGuard Drone'),
            model=data.get('model', 'Generic Quadcopter'),
            connection_string=data.get('connection_string'),
            zone_name=data.get('zone_name'),
            home_lat=data.get('home_lat', 20.5937),
            home_lon=data.get('home_lon', 78.9629),
        )
        db.session.add(d)
        db.session.commit()
        return jsonify(d.to_dict()), 201

    drones = FleetDrone.query.all()
    if not drones:
        _get_or_create_default_drone()
        drones = FleetDrone.query.all()
    return jsonify({'drones': [d.to_dict() for d in drones]})


@app.route('/api/drone/fleet/<int:drone_pk>', methods=['PATCH', 'DELETE'])
def drone_fleet_item(drone_pk):
    d = FleetDrone.query.get_or_404(drone_pk)
    if request.method == 'DELETE':
        db.session.delete(d)
        db.session.commit()
        return jsonify({'deleted': True})

    data = request.get_json(silent=True) or {}
    for field in ('name', 'model', 'zone_name', 'status', 'connection_string'):
        if field in data:
            setattr(d, field, data[field])
    if 'battery_pct' in data:
        d.battery_pct = float(data['battery_pct'])
    db.session.commit()
    return jsonify(d.to_dict())


@app.route('/api/drone/telemetry', methods=['GET'])
def drone_telemetry():
    """Drone Health Dashboard: simulated live parameters for a drone
    (real telemetry requires a connected MAVLink drone — not available here)."""
    if dcc is None:
        return jsonify({'error': 'Command center module unavailable'}), 503
    drone_id = request.args.get('drone_id')
    d = FleetDrone.query.filter_by(drone_id=drone_id).first() if drone_id else _get_or_create_default_drone()
    if not d:
        return jsonify({'error': 'drone not found'}), 404
    return jsonify(dcc.simulate_telemetry(d))


@app.route('/api/drone/emergency', methods=['GET', 'POST'])
def drone_emergency():
    """Emergency System: GET returns current alerts derived from telemetry;
    POST with {"drone_id":..., "action": "return_home"|"emergency_land"|
    "stop_spraying"|"stop_mission"} simulates issuing that command."""
    if dcc is None:
        return jsonify({'error': 'Command center module unavailable'}), 503

    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        d = FleetDrone.query.filter_by(drone_id=data.get('drone_id')).first() or _get_or_create_default_drone()
        action = data.get('action')
        new_status = dcc.EMERGENCY_ACTIONS.get(action)
        if not new_status:
            return jsonify({'error': f'unknown action {action}'}), 400
        d.status = new_status
        db.session.commit()
        return jsonify({'drone_id': d.drone_id, 'action': action, 'new_status': new_status, 'simulated': not bool(d.connection_string)})

    drone_id = request.args.get('drone_id')
    d = FleetDrone.query.filter_by(drone_id=drone_id).first() if drone_id else _get_or_create_default_drone()
    telemetry = dcc.simulate_telemetry(d)
    alerts = dcc.evaluate_emergency(telemetry)
    return jsonify({'drone_id': d.drone_id, 'alerts': alerts, 'telemetry': telemetry})


@app.route('/api/drone/maintenance', methods=['GET', 'POST'])
def drone_maintenance():
    """Maintenance tracker: propeller/battery/camera/firmware reminders."""
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        log = MaintenanceLog(
            drone_id=data.get('drone_id', 'drone-1'),
            item=data.get('item', 'General'),
            action=data.get('action'),
            due_flight_hours=data.get('due_flight_hours'),
            notes=data.get('notes'),
        )
        db.session.add(log)
        db.session.commit()
        return jsonify(log.to_dict()), 201

    drone_id = request.args.get('drone_id')
    q = MaintenanceLog.query
    if drone_id:
        q = q.filter_by(drone_id=drone_id)
    logs = q.order_by(MaintenanceLog.logged_at.desc()).all()
    return jsonify({'logs': [l.to_dict() for l in logs]})


@app.route('/api/drone/maintenance/<int:log_id>', methods=['PATCH', 'DELETE'])
def drone_maintenance_item(log_id):
    log = MaintenanceLog.query.get_or_404(log_id)
    if request.method == 'DELETE':
        db.session.delete(log)
        db.session.commit()
        return jsonify({'deleted': True})
    data = request.get_json(silent=True) or {}
    if 'completed' in data:
        log.completed = bool(data['completed'])
    db.session.commit()
    return jsonify(log.to_dict())


@app.route('/api/drone/schedule', methods=['GET', 'POST'])
def drone_schedule():
    """Mission Scheduler: every morning / weekly / before rain, etc."""
    if dcc is None:
        return jsonify({'error': 'Command center module unavailable'}), 503

    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        recurrence = data.get('recurrence', 'once')
        sched = MissionSchedule(
            name=data.get('name', 'Scheduled Mission'),
            mission_type=data.get('mission_type', 'scan'),
            field_name=data.get('field_name'),
            recurrence=recurrence,
            next_run=dcc.compute_next_run(recurrence),
        )
        db.session.add(sched)
        db.session.commit()
        return jsonify(sched.to_dict()), 201

    scheds = MissionSchedule.query.order_by(MissionSchedule.next_run.asc()).all()
    return jsonify({'schedules': [s.to_dict() for s in scheds]})


@app.route('/api/drone/schedule/<int:sched_id>', methods=['PATCH', 'DELETE'])
def drone_schedule_item(sched_id):
    sched = MissionSchedule.query.get_or_404(sched_id)
    if request.method == 'DELETE':
        db.session.delete(sched)
        db.session.commit()
        return jsonify({'deleted': True})
    data = request.get_json(silent=True) or {}
    if 'active' in data:
        sched.active = bool(data['active'])
    db.session.commit()
    return jsonify(sched.to_dict())


@app.route('/api/drone/copilot', methods=['POST'])
def drone_copilot():
    """AI Mission Copilot: parse a free-text command into a structured
    mission plan, then (if a field polygon is also supplied) run it
    through the real MissionPlanner to generate actual waypoints."""
    if dcc is None:
        return jsonify({'error': 'Command center module unavailable'}), 503

    data = request.get_json(silent=True) or {}
    text = data.get('command', '')
    if not text.strip():
        return jsonify({'error': 'command text required'}), 400

    parsed = dcc.parse_mission_command(text)

    polygon = data.get('field_polygon')
    if polygon and len(polygon) >= 3 and get_mission_planner:
        poly_tuples = [(p[0], p[1]) for p in polygon]
        planner = get_mission_planner()
        if 'spray' in parsed['actions']:
            coords = data.get('disease_coordinates', [])
            mission = planner.plan_spray_mission(
                disease_coordinates=coords,
                home_lat=data.get('home_lat', poly_tuples[0][0]),
                home_lon=data.get('home_lon', poly_tuples[0][1]),
            ) if coords else None
        else:
            mission = planner.plan_scan_mission(
                field_polygon=poly_tuples,
                home_lat=data.get('home_lat', poly_tuples[0][0]),
                home_lon=data.get('home_lon', poly_tuples[0][1]),
            )
        parsed['generated_mission'] = mission
    else:
        parsed['generated_mission'] = None
        parsed['note'] = 'Provide field_polygon (and disease_coordinates for spray) to auto-generate real waypoints.'

    return jsonify(parsed)


@app.route('/api/drone/analytics/heatmap', methods=['GET'])
def drone_analytics_heatmap():
    """Analytics: disease/pest heatmap grid built from real scan history."""
    if dcc is None:
        return jsonify({'error': 'Command center module unavailable'}), 503
    scans = ScanHistory.query.filter(ScanHistory.latitude.isnot(None)).all()
    return jsonify({'cells': dcc.build_heatmap(scans), 'total_scans_with_location': len(scans)})


@app.route('/api/drone/analytics/timeline', methods=['GET'])
def drone_analytics_timeline():
    """Historical Timeline: scans grouped by day for a growth/disease
    time-lapse view."""
    if dcc is None:
        return jsonify({'error': 'Command center module unavailable'}), 503
    scans = ScanHistory.query.order_by(ScanHistory.timestamp.asc()).all()
    return jsonify({'timeline': dcc.build_timeline(scans)})


@app.route('/api/drone/sustainability', methods=['GET'])
def drone_sustainability():
    """Sustainability Dashboard: estimated water/chemical savings from
    precision spraying vs. a blanket-spray baseline (assumption-based —
    see 'basis' field in the response)."""
    if dcc is None:
        return jsonify({'error': 'Command center module unavailable'}), 503
    plans = DronePlan.query.all()
    scans = ScanHistory.query.all()
    return jsonify(dcc.estimate_sustainability(plans, scans))


@app.route('/api/drone/flight-safety', methods=['GET'])
def drone_flight_safety():
    """Weather Intelligence: flight safety score derived from current
    weather at the given lat/lon."""
    if dcc is None or get_weather_intelligence is None:
        return jsonify({'error': 'Weather/command center module unavailable'}), 503
    lat = request.args.get('lat', 20.5937, type=float)
    lon = request.args.get('lon', 78.9629, type=float)
    wx_engine = get_weather_intelligence()
    wx = wx_engine.get_current_weather(lat, lon) if hasattr(wx_engine, 'get_current_weather') else {}
    return jsonify(dcc.flight_safety_score(wx if isinstance(wx, dict) else {}))



# ======================================================================
# DRONE COMMAND CENTER — EXTENSION PACK
# (AI Scan Modes, Interactive Farm Map, Sensor Hub, 3D Mapping,
#  Compliance Assistant, AI Collaboration Mode)
# ======================================================================

@app.route('/api/drone/scan-modes', methods=['GET'])
def list_scan_modes():
    if dcx is None:
        return jsonify({'error': 'Extension module unavailable'}), 503
    return jsonify({'modes': dcx.SCAN_MODES})


@app.route('/api/drone/scan-modes/analyze', methods=['POST'])
def analyze_scan_mode_route():
    """Multi-mode field analysis. Disease/pest/weed route through the real
    trained models you already have; every other mode uses a labeled
    image-statistics heuristic (see 'heuristic': true in the response)."""
    if dcx is None:
        return jsonify({'error': 'Extension module unavailable'}), 503
    if 'image' not in request.files:
        return jsonify({'error': 'image file required'}), 400
    mode = request.form.get('mode', 'disease')
    image_bytes = request.files['image'].read()

    if mode == 'disease':
        return jsonify({'mode': 'disease', 'redirect': '/api/scan', 'note': 'Use /api/scan (multipart image) for the real trained disease detector.'})
    if mode == 'weed':
        return jsonify({'mode': 'weed', 'redirect': '/api/detect/weeds', 'note': 'Use /api/detect/weeds for the real weed detector.'})
    if mode == 'pest':
        return jsonify({'mode': 'pest', 'redirect': '/api/detect/pests', 'note': 'Use /api/detect/pests for the real pest detector.'})

    if mode not in dcx.SCAN_MODES:
        return jsonify({'error': f'unknown mode {mode}'}), 400
    return jsonify(dcx.analyze_scan_mode(mode, image_bytes))


@app.route('/api/drone/zones', methods=['GET', 'POST'])
def drone_zones():
    """Interactive Farm Map: named zones/boundaries, drawn or imported."""
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        zone = FarmZone(
            name=data.get('name', 'Zone'),
            zone_type=data.get('zone_type', 'field'),
            boundary_points=json.dumps(data.get('boundary_points', [])),
            source=data.get('source', 'drawn'),
        )
        db.session.add(zone)
        db.session.commit()
        return jsonify(zone.to_dict()), 201
    return jsonify({'zones': [z.to_dict() for z in FarmZone.query.all()]})


@app.route('/api/drone/zones/<int:zone_id>', methods=['DELETE'])
def drone_zone_delete(zone_id):
    zone = FarmZone.query.get_or_404(zone_id)
    db.session.delete(zone)
    db.session.commit()
    return jsonify({'deleted': True})


@app.route('/api/drone/zones/import', methods=['POST'])
def drone_zones_import():
    """Import zone boundaries from pasted KML or GeoJSON text."""
    if dcx is None:
        return jsonify({'error': 'Extension module unavailable'}), 503
    data = request.get_json(silent=True) or {}
    text = data.get('text', '')
    fmt = data.get('format', 'geojson')
    try:
        polygons = dcx.parse_kml(text) if fmt == 'kml' else dcx.parse_geojson(text)
    except (ValueError, json.JSONDecodeError) as e:
        return jsonify({'error': f'Could not parse {fmt}: {e}'}), 400

    created = []
    for i, poly in enumerate(polygons):
        zone = FarmZone(name=data.get('name', f'Imported Zone {i+1}'), zone_type='field',
                         boundary_points=json.dumps(poly), source=fmt)
        db.session.add(zone)
        created.append(zone)
    db.session.commit()
    return jsonify({'imported': [z.to_dict() for z in created]}), 201


@app.route('/api/drone/nofly-zones', methods=['GET', 'POST'])
def nofly_zones():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        zone = NoFlyZone(name=data.get('name', 'No-Fly Zone'),
                          boundary_points=json.dumps(data.get('boundary_points', [])),
                          reason=data.get('reason'))
        db.session.add(zone)
        db.session.commit()
        return jsonify(zone.to_dict()), 201
    return jsonify({'zones': [z.to_dict() for z in NoFlyZone.query.all()]})


@app.route('/api/drone/nofly-zones/<int:zone_id>', methods=['DELETE'])
def nofly_zone_delete(zone_id):
    zone = NoFlyZone.query.get_or_404(zone_id)
    db.session.delete(zone)
    db.session.commit()
    return jsonify({'deleted': True})


@app.route('/api/drone/geofence-check', methods=['POST'])
def geofence_check():
    """Warn if a planned mission polygon overlaps a registered no-fly zone.
    Vertex-based check — a helpful warning, not a substitute for verified
    airspace clearance."""
    if dcx is None:
        return jsonify({'error': 'Extension module unavailable'}), 503
    data = request.get_json(silent=True) or {}
    polygon = data.get('field_polygon', [])
    zones = NoFlyZone.query.all()
    violations = dcx.check_geofence(polygon, zones)
    return jsonify({'safe': len(violations) == 0, 'violations': violations})


@app.route('/api/drone/sensors/hub', methods=['GET'])
def sensors_hub():
    """Expanded Sensor Hub: RGB/soil/weather are real where available;
    thermal/multispectral/hyperspectral/LiDAR/gas sensors are simulated
    placeholders (see 'simulated': true fields) since no such hardware is
    connected in this deployment."""
    if dcx is None:
        return jsonify({'error': 'Extension module unavailable'}), 503
    soil_reading = None
    if get_soil_hub:
        try:
            hub = get_soil_hub()
            soil_reading = hub.get_readings() if hasattr(hub, 'get_readings') else None
        except Exception:
            soil_reading = None
    weather = None
    if get_weather_intelligence:
        try:
            wx = get_weather_intelligence()
            weather = wx.get_current_weather(20.5937, 78.9629) if hasattr(wx, 'get_current_weather') else None
        except Exception:
            weather = None
    return jsonify(dcx.simulate_sensor_hub(soil_reading, weather))


@app.route('/api/drone/mapping/3d', methods=['POST'])
def mapping_3d():
    """3D Mapping: DEM/DSM/orthomosaic/point-cloud summary. Real outputs
    need a photogrammetry pipeline (e.g. OpenDroneMap) processing
    overlapping images — this returns a labeled placeholder summary."""
    if dcx is None:
        return jsonify({'error': 'Extension module unavailable'}), 503
    data = request.get_json(silent=True) or {}
    polygon = data.get('field_polygon', [])
    if len(polygon) < 3:
        return jsonify({'error': 'field_polygon must have at least 3 points'}), 400
    return jsonify(dcx.simulate_3d_mapping(polygon))


@app.route('/api/drone/compliance/report', methods=['GET'])
def compliance_report():
    """Regulatory & Compliance Assistant: pre-flight checklist + logged
    maintenance/mission summary. General guidance, not legal advice."""
    if dcx is None:
        return jsonify({'error': 'Extension module unavailable'}), 503
    fleet = FleetDrone.query.all()
    maint = MaintenanceLog.query.all()
    mission_count = DronePlan.query.count()
    return jsonify(dcx.build_compliance_report(fleet, maint, mission_count))


@app.route('/api/drone/collaboration/share', methods=['GET', 'POST'])
def collaboration_share():
    """AI Collaboration Mode: create a role-scoped share link for a
    mission (agronomist, farm_manager, government_agency, researcher,
    insurance_company). No real authentication — a shareable token, not
    an access-controlled account."""
    if dcx is None:
        return jsonify({'error': 'Extension module unavailable'}), 503
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        try:
            share = dcx.create_share_token(data.get('mission_name', 'Mission'), data.get('role', ''))
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        row = MissionShare(mission_name=share['mission_name'], role=share['role'],
                            token=share['token'], visible_sections=json.dumps(share['visible_sections']))
        db.session.add(row)
        db.session.commit()
        return jsonify(row.to_dict()), 201
    return jsonify({'shares': [s.to_dict() for s in MissionShare.query.order_by(MissionShare.created_at.desc()).all()],
                     'available_roles': dcx.ROLES})


@app.route('/api/drone/collaboration/share/<int:share_id>', methods=['DELETE'])
def collaboration_share_delete(share_id):
    row = MissionShare.query.get_or_404(share_id)
    db.session.delete(row)
    db.session.commit()
    return jsonify({'deleted': True})


@app.route('/api/drone/scan-mode', methods=['POST'])
def drone_scan_mode():
    """AI Automatic Detection Modes: dispatches to a real trained detector
    when one exists for the mode (disease/pest/weed/ndvi), otherwise runs
    a transparent image-statistics heuristic (see 'heuristic_estimate' in
    the response) for modes with no trained model yet."""
    mode = request.form.get('mode', 'disease')
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    filepath, _ = save_uploaded_file(file)
    if not filepath:
        return jsonify({'error': 'Invalid file'}), 400

    if mode == 'disease':
        result = detector.detect(filepath)
        result['mode'] = 'disease'
        return jsonify(result)
    if mode == 'pest' and get_pest_detector:
        result = get_pest_detector().detect(filepath)
        result['mode'] = 'pest'
        return jsonify(result)
    if mode == 'weed' and get_weed_detector:
        result = get_weed_detector().detect(filepath)
        result['mode'] = 'weed'
        return jsonify(result)
    if mode == 'ndvi' and calculate_ndvi:
        result = calculate_ndvi(filepath)
        result['mode'] = 'ndvi'
        return jsonify(result)

    if dcc is None:
        return jsonify({'error': 'Command center module unavailable'}), 503
    return jsonify(dcc.heuristic_image_scan(filepath, mode))


@app.route('/api/drone/sensor-hub', methods=['GET'])
def drone_sensor_hub():
    """Sensor Dashboard: real soil sensor readings + simulated placeholders
    for the sensors this deployment doesn't have connected yet."""
    if dcc is None:
        return jsonify({'error': 'Command center module unavailable'}), 503
    soil = None
    if get_soil_hub:
        try:
            soil = get_soil_hub().get_latest_readings() if hasattr(get_soil_hub(), 'get_latest_readings') else None
        except Exception:
            soil = None
    hub = dcc.simulate_sensor_hub()
    hub['soil'] = soil
    return jsonify(hub)


@app.route('/api/drone/mapping', methods=['POST'])
def drone_mapping():
    """3D Mapping: real DEM/DSM/orthomosaic generation needs photogrammetry
    software processing actual overlapping flight photos — this returns a
    job placeholder describing what that would take (see 'note')."""
    if dcc is None:
        return jsonify({'error': 'Command center module unavailable'}), 503
    data = request.get_json(silent=True) or {}
    return jsonify(dcc.mapping_job_stub(data.get('field_polygon'), data.get('area_ha')))


@app.route('/api/drone/spray/dose', methods=['POST'])
def drone_spray_dose():
    """Smart Spraying: real dose/duration calculation from severity, via
    the existing SprayController."""
    try:
        from modules.drone.spray_controller import get_spray_controller
    except ImportError:
        return jsonify({'error': 'Spray controller not available'}), 503
    data = request.get_json(silent=True) or {}
    severity = data.get('severity', 'Medium')
    disease_name = data.get('disease_name', 'unknown')
    controller = get_spray_controller()
    return jsonify(controller.calculate_dose(severity, disease_name))


@app.route('/api/drone/fields/<int:field_id>/boundary', methods=['POST'])
def set_field_boundary(field_id):
    """Interactive Farm Map: save a drawn/imported polygon boundary for a field."""
    field = FarmField.query.get_or_404(field_id)
    data = request.get_json(silent=True) or {}
    polygon = data.get('boundary')
    if not polygon or len(polygon) < 3:
        return jsonify({'error': 'boundary must have at least 3 [lat,lon] points'}), 400
    field.boundary_geojson = json.dumps(polygon)
    db.session.commit()
    return jsonify(field.to_dict())


@app.route('/api/drone/import/kml', methods=['POST'])
def import_kml():
    """Interactive Farm Map: import a field boundary from an uploaded KML file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No KML file provided'}), 400
    import xml.etree.ElementTree as ET
    try:
        content = request.files['file'].read().decode('utf-8', errors='ignore')
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        root = ET.fromstring(content)
        coords_el = root.find('.//kml:coordinates', ns) or root.find('.//coordinates')
        if coords_el is None:
            return jsonify({'error': 'No <coordinates> found in KML'}), 400
        points = []
        for triple in coords_el.text.strip().split():
            lon, lat, *_ = triple.split(',')
            points.append([float(lat), float(lon)])
        return jsonify({'boundary': points, 'point_count': len(points)})
    except Exception as e:
        return jsonify({'error': f'Could not parse KML: {e}'}), 400


@app.route('/api/drone/import/geojson', methods=['POST'])
def import_geojson():
    """Interactive Farm Map: import a field boundary from uploaded GeoJSON."""
    data = request.get_json(silent=True) or {}
    try:
        geom = data.get('geometry', data)
        coords = geom['coordinates'][0]  # first ring of a Polygon
        points = [[c[1], c[0]] for c in coords]  # GeoJSON is [lon,lat] -> flip to [lat,lon]
        return jsonify({'boundary': points, 'point_count': len(points)})
    except Exception as e:
        return jsonify({'error': f'Could not parse GeoJSON: {e}'}), 400


@app.route('/api/drone/geofence', methods=['GET', 'POST'])
def drone_geofence():
    """Drone Security: manage no-fly zones, or check a point against them."""
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        if 'check_lat' in data:
            if dcc is None:
                return jsonify({'error': 'Command center module unavailable'}), 503
            zones = [z.to_dict() for z in NoFlyZone.query.all()]
            return jsonify(dcc.check_geofence(data['check_lat'], data['check_lon'], zones))
        zone = NoFlyZone(name=data.get('name', 'No-Fly Zone'), polygon=json.dumps(data.get('polygon', [])))
        db.session.add(zone)
        db.session.commit()
        return jsonify(zone.to_dict()), 201

    zones = NoFlyZone.query.all()
    return jsonify({'zones': [z.to_dict() for z in zones]})


@app.route('/api/drone/geofence/<int:zone_id>', methods=['DELETE'])
def drone_geofence_delete(zone_id):
    zone = NoFlyZone.query.get_or_404(zone_id)
    db.session.delete(zone)
    db.session.commit()
    return jsonify({'deleted': True})


@app.route('/api/drone/compliance-report', methods=['GET'])
def drone_compliance_report():
    """Regulatory & Compliance Assistant: exportable activity summary +
    operator checklist (not legal advice — see 'note' in the response)."""
    if dcc is None:
        return jsonify({'error': 'Command center module unavailable'}), 503
    fleet = FleetDrone.query.all()
    maint = MaintenanceLog.query.all()
    scans = ScanHistory.query.all()
    scheds = MissionSchedule.query.all()
    return jsonify(dcc.build_compliance_report(fleet, maint, scans, scheds))


@app.route('/api/drone/collaboration/share', methods=['GET', 'POST'])
def drone_collaboration_share():
    """AI Collaboration Mode: create/list read-only share tokens for a
    field, tagged with a role (agronomist/farm_manager/government/
    researcher/insurer). No auth on the recipient side is implemented yet
    — treat tokens as capability links, not access control."""
    import secrets as _secrets
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        share = MissionShare(
            token=_secrets.token_urlsafe(24),
            field_id=data.get('field_id'),
            role=data.get('role', 'agronomist'),
        )
        db.session.add(share)
        db.session.commit()
        return jsonify(share.to_dict()), 201

    shares = MissionShare.query.all()
    return jsonify({'shares': [s.to_dict() for s in shares]})


@app.route('/api/health/full', methods=['GET'])
def full_health():
    """Complete health check for all Phase 0-5 modules."""
    return jsonify({
        'status':   'healthy',
        'version':  '4.0.0',
        'phases': {
            'phase_0_ensemble':       get_pipeline is not None,
            'phase_1_predictive':     get_predictive_engine is not None,
            'phase_2_assistant':      get_assistant is not None,
            'phase_2_weather':        get_weather_intelligence is not None,
            'phase_3_drone_planner':  get_mission_planner is not None,
            'phase_3_soil_sensors':   get_soil_hub is not None,
            'phase_3_ndvi':           calculate_ndvi is not None,
            'phase_4_yield':          get_yield_predictor is not None,
            'phase_4_weed':           get_weed_detector is not None,
            'phase_4_pest':           get_pest_detector is not None,
            'phase_5_farm_memory':    get_farm_memory is not None,
            'phase_5_carbon':         get_carbon_tracker is not None,
            'phase_5_marketplace':    get_marketplace is not None,
        },
        'yolo_model_loaded': detector.model_loaded,
        'device':            detector.device,
        'timestamp':         datetime.utcnow().isoformat(),
    })


# ===================== MAIN =====================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=app.config.get('DEBUG', False))

