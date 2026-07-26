"""
CropGuard AI Pro v3.0 - Python Flask Application
Advanced crop disease detection with YOLOv8
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
from models import db, ScanHistory, DronePlan, UserProfile, FarmField, FarmingTask, DroneDevice
from disease_database import CROP_DISEASE_DB, MARKET_DATA, MANDI_PREMIUMS, CROP_ICONS, CROP_NAMES
from yolo_detector import get_detector
from image_utils import normalize_image_bytes, normalize_uploaded_file, ImageDecodeError
from discovery import start_discovery_beacon, DiscoveryBeacon

# Load environment variables


# Create Flask app
def create_app(config_name='development'):
    app = Flask(__name__, static_folder='static', template_folder='templates')
    app.config.from_object(config[config_name])

    # Initialize extensions
    CORS(app)
    db.init_app(app)

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

# ===================== WEATHER API =====================

@app.route('/api/weather', methods=['GET'])
def get_weather():
    """Get weather data from Open-Meteo"""
    import requests

    lat = request.args.get('lat', 20.5937)
    lon = request.args.get('lon', 78.9629)

    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&"
            f"current=temperature_2m,relative_humidity_2m,apparent_temperature,"
            f"is_day,precipitation,rain,weather_code,cloud_cover,wind_speed_10m,wind_direction_10m&"
            f"daily=precipitation_probability_max,uv_index_max,temperature_2m_max,temperature_2m_min&"
            f"timezone=auto&forecast_days=7"
        )

        resp = requests.get(url, timeout=10)
        data = resp.json()

        # Add farming advice
        current = data.get('current', {})
        advice = []

        humidity = current.get('relative_humidity_2m', 50)
        temp = current.get('temperature_2m', 25)
        wind = current.get('wind_speed_10m', 5)
        rain_prob = data.get('daily', {}).get('precipitation_probability_max', [0])[0]

        if humidity > 85 and temp > 20:
            advice.append("⚠️ High humidity alert: Fungal disease risk is elevated. Avoid overhead irrigation.")
        if rain_prob > 60:
            advice.append("🌧️ Rain expected: Delay spraying operations. Good time for soil fertilizer application.")
        if temp > 38:
            advice.append("🔥 Extreme heat: Ensure adequate irrigation. Spray only early morning/evening.")
        if wind > 20:
            advice.append("💨 High winds: Avoid drone spraying and foliar applications today.")
        if temp < 5:
            advice.append("❄️ Cold weather: Protect sensitive crops. Delay sowing until temperatures rise.")
        if not advice:
            advice.append("✅ Good conditions: Weather is favorable for most farming operations.")

        data['farming_advice'] = advice
        return jsonify(data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

# ===================== MAIN =====================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=app.config.get('DEBUG', False))
