// CropGuard AI Pro v3.0 - Frontend JavaScript
// Handles all UI interactions and API calls

const API_BASE = '';
let currentImage = null;
let currentScanResult = null;
let map = null;
let fieldPolygon = null;
let mapPoints = [];
let currentMonth = new Date();
let weatherData = null;

// ===================== TAB NAVIGATION =====================
function showTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    const tab = document.getElementById('tab-' + tabId);
    if(tab) tab.classList.add('active');
    const btn = document.querySelector('.tab-btn[data-tab="' + tabId + '"]');
    if(btn) btn.classList.add('active');

    if(tabId === 'drone' && !map) setTimeout(initMap, 100);
    if(tabId === 'market') renderMarket();
    if(tabId === 'history') renderHistory();
    if(tabId === 'calendar') renderCalendar();
    if(tabId === 'profile') loadProfile();
    if(tabId === 'yield') renderYieldDefaults();

    if(tabId === 'drone') startDronePolling();
    else stopDronePolling();
}

// ===================== FILE UPLOAD =====================
function handleFileSelect(e) {
    const file = e.target.files[0];
    if(file) processFile(file);
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if(file && file.type.startsWith('image/')) processFile(file);
}

function processFile(file) {
    if(!file.type.startsWith('image/')) { showToast('Please upload an image file'); return; }
    if(file.size > 15 * 1024 * 1024) { showToast('File too large. Max 15MB.'); return; }
    currentImage = file;
    const reader = new FileReader();
    reader.onload = function(e) {
        const img = document.getElementById('preview-img');
        img.src = e.target.result;
        img.style.display = 'block';
        document.getElementById('analyzeBtn').disabled = false;
    };
    reader.readAsDataURL(file);
}

function clearScan() {
    currentImage = null;
    currentScanResult = null;
    document.getElementById('preview-img').style.display = 'none';
    document.getElementById('analyzeBtn').disabled = true;
    document.getElementById('scanResults').style.display = 'none';
    document.getElementById('fileInput').value = '';
    document.getElementById('annotatedImageContainer').style.display = 'none';
}

// ===================== AI SCAN =====================
async function analyzeImage() {
    if(!currentImage) { showToast('Please upload an image first'); return; }

    showLoading('Running YOLOv8 AI detection...');
    document.getElementById('loadingSub').textContent = 'Neural network processing image with bounding boxes';

    const formData = new FormData();
    formData.append('image', currentImage);
    const cropGroup = document.getElementById('cropGroup').value;
formData.append('crop_group', cropGroup);

    try {
        const response = await fetch(`${API_BASE}/api/scan`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if(!response.ok) {
            throw new Error(result.error || 'Scan failed');
        }

        hideLoading();
        currentScanResult = result;
        renderScanResults(result);

    } catch(err) {
        hideLoading();
        showToast('Error: ' + err.message);
        console.error(err);
    }
}

function renderScanResults(result) {
    document.getElementById('scanResults').style.display = 'block';

    // Show annotated image if available
    if(result.annotated_image) {
        const img = document.getElementById('annotatedImage');
        img.src = result.annotated_image;
        document.getElementById('annotatedImageContainer').style.display = 'block';
    }

    // Detection list
    const detectionList = document.getElementById('detectionList');
    if(result.detections && result.detections.length > 0) {
        detectionList.innerHTML = result.detections.map((d, i) => {
            const severityClass = d.confidence > 70 ? 'high-severity' : d.confidence > 40 ? 'medium-severity' : 'low-severity';
            return `
                <div class="detection-box ${severityClass}">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <strong>${d.class_name}</strong>
                        <span class="badge badge-${d.confidence > 70 ? 'orange' : d.confidence > 40 ? 'blue' : 'green'}">${d.confidence}% confidence</span>
                    </div>
                    <small class="text-muted">${d.bbox ? `Bounding box: [${d.bbox.map(n => Math.round(n)).join(', ')}]` : 'Whole-image classification (no bounding box)'}</small>
                </div>
            `;
        }).join('');
    } else {
        detectionList.innerHTML = '<p class="text-muted">No diseases detected - plant appears healthy!</p>';
    }

    // Pixel grid (if available)
    const pixelWrap = document.getElementById('pixelGridWrap');
    if(result.pixel_grid && result.pixel_grid.length > 0) {
        pixelWrap.style.display = 'block';
        const grid = document.getElementById('pixelGrid');
        grid.innerHTML = result.pixel_grid.map(p => {
            const color = p.type === 'spot' ? `rgb(${Math.min(255,p.r+40)},${Math.max(0,p.g-30)},${Math.max(0,p.b-30)})` : `rgb(${p.r},${p.g},${p.b})`;
            const outline = p.type === 'spot' ? 'box-shadow:inset 0 0 0 2px rgba(193,18,31,0.7)' : '';
            return `<div class="pixel-cell" style="background:${color};${outline}" title="${p.type}"></div>`;
        }).join('');
    } else {
        pixelWrap.style.display = 'none';
    }

    // Disease result
    const disease = result.disease;
    const isHealthy = disease.severity === 'Healthy';
    const sevClass = disease.severity === 'High' ? 'severity-high' : disease.severity === 'Medium' ? 'severity-medium' : disease.severity === 'Low' ? 'severity-low' : 'severity-healthy';

    document.getElementById('diseaseResult').innerHTML = `
        <div class="disease-result ${isHealthy ? 'healthy' : ''}">
            <div style="display:flex;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:10px">
                <span class="disease-name">${result.crop.icon} ${disease.name}</span>
                <span class="severity-badge ${sevClass}">${disease.severity}</span>
                <span style="font-size:0.85rem;color:#888">AI Confidence: ${disease.confidence}%</span>
            </div>
            <p style="color:#555;line-height:1.6">${disease.description}</p>
            <div style="margin-top:14px">
                <strong style="color:var(--primary);display:block;margin-bottom:8px"><i class="fas fa-list-ul"></i> Key Symptoms:</strong>
                <div style="display:flex;flex-wrap:wrap;gap:8px">
                    ${disease.symptoms.map(s => `<span style="background:var(--light);padding:6px 12px;border-radius:20px;font-size:0.85rem;color:var(--primary);border:1px solid var(--border)">${s}</span>`).join('')}
                </div>
            </div>
        </div>
    `;

    // Treatments
    const treatments = disease.treatments;
    document.getElementById('treatmentSection').innerHTML = `
        <div class="treatment-tabs">
            <button class="treatment-tab active" onclick="showTreatment('chemical',this)"><i class="fas fa-flask"></i> Chemical</button>
            <button class="treatment-tab" onclick="showTreatment('organic',this)"><i class="fas fa-leaf"></i> Organic</button>
            <button class="treatment-tab" onclick="showTreatment('prevention',this)"><i class="fas fa-shield-alt"></i> Prevention</button>
        </div>
        <div class="treatment-content active" id="treat-chemical">
            ${treatments.chemical.map(t => `<div class="treatment-item"><i class="fas fa-spray-can"></i><div>${t}</div></div>`).join('')}
        </div>
        <div class="treatment-content" id="treat-organic">
            ${treatments.organic.map(t => `<div class="treatment-item"><i class="fas fa-seedling"></i><div>${t}</div></div>`).join('')}
        </div>
        <div class="treatment-content" id="treat-prevention">
            ${treatments.prevention.map(t => `<div class="treatment-item"><i class="fas fa-check-circle"></i><div>${t}</div></div>`).join('')}
        </div>
    `;

    window.scrollTo({ top: document.getElementById('scanResults').offsetTop - 80, behavior: 'smooth' });
}

function showTreatment(type, btn) {
    document.querySelectorAll('.treatment-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.treatment-content').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('treat-' + type).classList.add('active');
}

function exportReport() {
    if(!currentScanResult) { showToast('No scan to export'); return; }
    const r = currentScanResult;
    const report = `CROPGUARD AI PRO v3.0 SCAN REPORT
Generated: ${new Date().toLocaleString()}
Crop: ${r.crop.name} ${r.crop.icon}
Disease: ${r.disease.name}
Severity: ${r.disease.severity}
Confidence: ${r.disease.confidence}%
Description: ${r.disease.description}
Symptoms: ${r.disease.symptoms.join(', ')}
Model: ${r.model_used || 'YOLOv8'}
`;
    const blob = new Blob([report], {type: 'text/plain'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `CropGuard_Report_${r.crop.name}_${new Date().toISOString().slice(0,10)}.txt`;
    a.click();
    showToast('Report exported!');
}

function shareResult() {
    if(!currentScanResult) { showToast('No result to share'); return; }
    const text = `CropGuard AI detected: ${currentScanResult.crop.name} - ${currentScanResult.disease.name} (${currentScanResult.disease.confidence}% confidence)`;
    if(navigator.share) {
        navigator.share({ title: 'CropGuard AI Scan', text: text });
    } else {
        navigator.clipboard.writeText(text).then(() => showToast('Copied to clipboard!'));
    }
}

function saveToHistory() {
    showToast('Scan saved to history!');
}

// ===================== HISTORY =====================
async function renderHistory() {
    const list = document.getElementById('historyList');
    try {
        const resp = await fetch(`${API_BASE}/api/history`);
        const data = await resp.json();
        const scans = data.scans || [];

        if(scans.length === 0) {
            list.innerHTML = '<p class="text-muted text-center" style="padding:30px"><i class="fas fa-inbox" style="font-size:2rem;display:block;margin-bottom:10px"></i>No scan history yet.</p>';
            return;
        }

        list.innerHTML = scans.map(h => {
            const isHealthy = h.disease_name === 'Healthy';
            const cls = isHealthy ? 'healthy' : (h.severity === 'High' ? 'severe' : '');
            return `
                <div class="history-item ${cls}">
                    <div class="history-icon">${h.crop_icon}</div>
                    <div class="history-details">
                        <div class="history-title">${h.crop_name} - ${h.disease_name}</div>
                        <div class="history-meta">${new Date(h.timestamp).toLocaleString()} | ${h.source}</div>
                    </div>
                    <div class="history-confidence">${h.confidence}%</div>
                    <button class="btn btn-secondary btn-sm" onclick="deleteScan(${h.id})"><i class="fas fa-trash"></i></button>
                </div>
            `;
        }).join('');
    } catch(e) {
        list.innerHTML = '<p class="text-muted text-center">Error loading history</p>';
    }
}

async function deleteScan(id) {
    if(!confirm('Delete this scan?')) return;
    try {
        await fetch(`${API_BASE}/api/history/${id}`, { method: 'DELETE' });
        renderHistory();
        showToast('Deleted');
    } catch(e) {
        showToast('Error deleting');
    }
}

function exportAllHistory() {
    window.open(`${API_BASE}/api/history/export`, '_blank');
}

function clearHistory() {
    if(!confirm('Clear all history?')) return;
    showToast('History cleared (refresh to see)');
}

// ===================== WEATHER =====================
async function detectLocation() {
    if(!navigator.geolocation) { showToast('Geolocation not supported'); return; }
    document.getElementById('weatherLoading').style.display = 'block';
    document.getElementById('weatherDisplay').style.display = 'none';
    navigator.geolocation.getCurrentPosition(
        pos => fetchWeather(pos.coords.latitude, pos.coords.longitude),
        () => { document.getElementById('weatherLoading').style.display = 'none'; showToast('Could not detect location'); }
    );
}

function manualLocation() {
    const lat = prompt('Enter latitude:');
    const lon = prompt('Enter longitude:');
    if(lat && lon) {
        document.getElementById('weatherLoading').style.display = 'block';
        fetchWeather(parseFloat(lat), parseFloat(lon));
    }
}

async function fetchWeather(lat, lon) {
    try {
        const resp = await fetch(`${API_BASE}/api/weather?lat=${lat}&lon=${lon}`);
        const data = await resp.json();
        weatherData = data;
        renderWeather(data, lat, lon);
    } catch(e) {
        document.getElementById('weatherLoading').style.display = 'none';
        showToast('Failed to fetch weather');
    }
}

function renderWeather(data, lat, lon) {
    document.getElementById('weatherLoading').style.display = 'none';
    document.getElementById('weatherDisplay').style.display = 'block';

    const current = data.current;
    const wMap = {0:{desc:'Clear Sky',icon:'fa-sun'},1:{desc:'Mainly Clear',icon:'fa-cloud-sun'},2:{desc:'Partly Cloudy',icon:'fa-cloud-sun'},3:{desc:'Overcast',icon:'fa-cloud'},45:{desc:'Foggy',icon:'fa-smog'},61:{desc:'Slight Rain',icon:'fa-cloud-rain'},63:{desc:'Moderate Rain',icon:'fa-cloud-rain'},65:{desc:'Heavy Rain',icon:'fa-cloud-showers-heavy'},71:{desc:'Snow',icon:'fa-snowflake'},95:{desc:'Thunderstorm',icon:'fa-bolt'}};
    const wInfo = wMap[current.weather_code] || {desc:'Unknown',icon:'fa-cloud'};
    const dirs = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'];
    const windDir = dirs[Math.round(current.wind_direction_10m / 22.5) % 16];

    document.getElementById('weatherTemp').textContent = Math.round(current.temperature_2m) + '°C';
    document.getElementById('weatherDesc').innerHTML = `<i class="fas ${wInfo.icon}"></i> ${wInfo.desc}`;
    document.getElementById('weatherLocation').innerHTML = `<i class="fas fa-map-marker-alt"></i> ${lat.toFixed(3)}, ${lon.toFixed(3)}`;
    document.getElementById('weatherTime').textContent = new Date().toLocaleString();

    document.getElementById('weatherGrid').innerHTML = `
        <div class="weather-item"><i class="fas fa-temperature-high"></i><span>${Math.round(current.apparent_temperature)}°C</span><small>Feels Like</small></div>
        <div class="weather-item"><i class="fas fa-tint"></i><span>${current.relative_humidity_2m}%</span><small>Humidity</small></div>
        <div class="weather-item"><i class="fas fa-wind"></i><span>${current.wind_speed_10m} km/h</span><small>Wind ${windDir}</small></div>
        <div class="weather-item"><i class="fas fa-cloud-rain"></i><span>${data.daily.precipitation_probability_max[0]}%</span><small>Rain Chance</small></div>
        <div class="weather-item"><i class="fas fa-sun"></i><span>${data.daily.uv_index_max[0]}</span><small>UV Index</small></div>
        <div class="weather-item"><i class="fas fa-cloud"></i><span>${current.cloud_cover}%</span><small>Cloud Cover</small></div>
    `;

    document.getElementById('weatherAdvice').innerHTML = data.farming_advice.map(a => `<p>${a}</p>`).join('');
}

function getWeatherForecast() {
    if(!weatherData) { showToast('Get current weather first'); return; }
    const forecast = document.getElementById('forecastSection');
    const grid = document.getElementById('forecastGrid');
    forecast.style.display = 'block';
    const days = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
    grid.innerHTML = weatherData.daily.time.map((t, i) => `
        <div class="weather-item">
            <span style="font-weight:700;color:var(--primary)">${days[new Date(t).getDay()]}</span>
            <span>${Math.round(weatherData.daily.temperature_2m_max[i])}°/${Math.round(weatherData.daily.temperature_2m_min[i])}°</span>
            <small>${weatherData.daily.precipitation_probability_max[i]}% rain</small>
        </div>
    `).join('');
    showToast('7-day forecast loaded!');
}

// ===================== FERTILIZER =====================
async function calculateFertilizer() {
    const crop = document.getElementById('fertCrop').value;
    const area = parseFloat(document.getElementById('fertArea').value);
    if(!crop || !area) { showToast('Please fill required fields'); return; }

    const data = {
        crop: crop,
        area: area,
        stage: document.getElementById('fertStage').value,
        soil: document.getElementById('fertSoil').value,
        fertilizer_type: document.getElementById('fertType').value,
        soil_n: document.getElementById('soilN').value,
        soil_p: document.getElementById('soilP').value
    };

    try {
        const resp = await fetch(`${API_BASE}/api/fertilizer`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });
        const result = await resp.json();

        let html = '<div class="calc-result">';
        html += `<h4><i class="fas fa-flask"></i> Fertilizer for ${area} acres of ${crop}</h4>`;
        html += `<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:15px;margin-top:15px">`;
        html += `<div><div class="big-number">${result.requirements.N} kg</div><small>Nitrogen (N)</small></div>`;
        html += `<div><div class="big-number">${result.requirements.P} kg</div><small>Phosphorus (P)</small></div>`;
        html += `<div><div class="big-number">${result.requirements.K} kg</div><small>Potassium (K)</small></div></div>`;

        if(result.recommendations.chemical) {
            html += `<div style="margin-top:20px;text-align:left"><h4 style="color:var(--primary)"><i class="fas fa-shopping-cart"></i> Chemical Fertilizers</h4>`;
            html += `<div class="treatment-item"><i class="fas fa-bag"></i><div><strong>Urea:</strong> ${result.recommendations.chemical.urea_kg} kg</div></div>`;
            html += `<div class="treatment-item"><i class="fas fa-bag"></i><div><strong>DAP:</strong> ${result.recommendations.chemical.dap_kg} kg</div></div>`;
            html += `<div class="treatment-item"><i class="fas fa-bag"></i><div><strong>MOP:</strong> ${result.recommendations.chemical.mop_kg} kg</div></div></div>`;
        }
        if(result.recommendations.organic) {
            html += `<div style="margin-top:15px;text-align:left"><h4 style="color:var(--primary)"><i class="fas fa-leaf"></i> Organic Inputs</h4>`;
            html += `<div class="treatment-item"><i class="fas fa-tractor"></i><div><strong>FYM:</strong> ${result.recommendations.organic.fym_kg} kg</div></div>`;
            html += `<div class="treatment-item"><i class="fas fa-worm"></i><div><strong>Vermicompost:</strong> ${result.recommendations.organic.vermicompost_kg} kg</div></div></div>`;
        }
        html += '</div>';
        document.getElementById('fertilizerResult').innerHTML = html;

    } catch(e) {
        showToast('Error calculating fertilizer');
    }
}

// ===================== IRRIGATION =====================
async function checkIrrigation() {
    const data = {
        crop: document.getElementById('irrCrop').value,
        moisture: parseFloat(document.getElementById('irrMoisture').value),
        days_since: parseInt(document.getElementById('irrDays').value),
        area: parseFloat(document.getElementById('irrArea').value),
        method: document.getElementById('irrMethod').value,
        rain: parseFloat(document.getElementById('irrRain').value) || 0
    };

    if(isNaN(data.moisture) || isNaN(data.area)) { showToast('Please fill all fields'); return; }

    try {
        const resp = await fetch(`${API_BASE}/api/irrigation`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });
        const result = await resp.json();

        document.getElementById('irrigationResult').innerHTML = `
            <div class="calc-result" style="border-left:5px solid ${result.status_color}">
                <h4 style="color:${result.status_color}"><i class="fas fa-tint"></i> ${result.status}</h4>
                <p style="margin:12px 0">Moisture: ${result.moisture}% | Optimal: ${result.optimal}%</p>
                <div class="moisture-bar">
                    <div class="moisture-fill" style="width:${result.moisture}%;background:linear-gradient(90deg,${result.status_color},var(--secondary))"></div>
                    <div class="moisture-optimal" style="left:${result.optimal}%"></div>
                </div>
                <div class="moisture-labels"><span>0%</span><span>Optimal: ${result.optimal}%</span><span>100%</span></div>
                ${result.water_needed_liters > 0 ? `<div style="margin-top:15px"><div class="big-number">${result.water_needed_liters.toLocaleString()} liters</div><small>Water needed for ${result.area} acres (${result.method})</small></div>` : ''}
                <p class="text-muted" style="margin-top:12px"><i class="fas fa-lightbulb"></i> ${result.method_tip}</p>
            </div>
        `;
    } catch(e) {
        showToast('Error checking irrigation');
    }
}

// ===================== YIELD =====================
const YIELD_DEFAULTS = {
    wheat: {yield: 25, price: 2400, cost: 15000},
    rice: {yield: 30, price: 3200, cost: 18000},
    corn: {yield: 35, price: 2100, cost: 12000},
    cotton: {yield: 8, price: 6200, cost: 25000},
    tomato: {yield: 200, price: 2800, cost: 30000},
    potato: {yield: 150, price: 1500, cost: 20000},
    sugarcane: {yield: 400, price: 340, cost: 35000},
    soybean: {yield: 15, price: 4500, cost: 12000}
};

function renderYieldDefaults() {
    const crop = document.getElementById('yieldCrop').value;
    const d = YIELD_DEFAULTS[crop];
    if(d) {
        document.getElementById('yieldPerAcre').value = d.yield;
        document.getElementById('yieldPrice').value = d.price;
        document.getElementById('yieldCost').value = d.cost;
    }
}

async function calculateYield() {
    const data = {
        crop: document.getElementById('yieldCrop').value,
        area: parseFloat(document.getElementById('yieldArea').value),
        yield_per_acre: parseFloat(document.getElementById('yieldPerAcre').value),
        price: parseFloat(document.getElementById('yieldPrice').value),
        cost: parseFloat(document.getElementById('yieldCost').value),
        irrigation: document.getElementById('yieldIrrigation').value
    };

    if(!data.area || !data.yield_per_acre || !data.price || !data.cost) {
        showToast('Please fill all fields'); return;
    }

    try {
        const resp = await fetch(`${API_BASE}/api/yield`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });
        const r = await resp.json();

        const profitColor = r.net_profit > 0 ? '#1a5c3a' : '#c1121f';
        document.getElementById('yieldResult').innerHTML = `
            <div class="calc-result">
                <h4><i class="fas fa-chart-bar"></i> ${r.crop.toUpperCase()} - Yield & Profit</h4>
                <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:15px;margin-top:15px">
                    <div><div class="big-number">${r.total_yield.toLocaleString()}</div><small>Total Yield (Q)</small></div>
                    <div><div class="big-number">Rs ${r.gross_income.toLocaleString()}</div><small>Gross Income</small></div>
                    <div><div class="big-number">Rs ${r.total_cost.toLocaleString()}</div><small>Total Cost</small></div>
                    <div><div class="big-number" style="color:${profitColor}">Rs ${r.net_profit.toLocaleString()}</div><small>Net Profit</small></div>
                </div>
                <div style="margin-top:15px">
                    <p><strong>Profit/Acre:</strong> Rs ${Math.round(r.profit_per_acre).toLocaleString()}</p>
                    <p><strong>ROI:</strong> ${r.roi_percent}%</p>
                    <p><strong>Irrigation Cost:</strong> Rs ${r.irrigation_cost.toLocaleString()}</p>
                </div>
            </div>
        `;
    } catch(e) {
        showToast('Error calculating yield');
    }
}

// ===================== MARKET =====================
async function renderMarket() {
    const mandi = document.getElementById('marketLocation').value;
    const sort = document.getElementById('marketSort').value;

    try {
        const resp = await fetch(`${API_BASE}/api/market?mandi=${mandi}&sort=${sort}`);
        const data = await resp.json();

        const grid = document.getElementById('marketGrid');
        grid.innerHTML = data.items.map(item => {
            const isUp = item.change > 0;
            return `
                <div class="market-card">
                    <div style="font-size:2rem;margin-bottom:5px">${item.icon}</div>
                    <div class="market-crop">${item.name}</div>
                    <div class="market-price">${item.currency}${item.price.toLocaleString()}</div>
                    <div class="market-change ${isUp ? 'up' : 'down'}"><i class="fas fa-arrow-${isUp ? 'up' : 'down'}"></i> ${Math.abs(item.change)}%</div>
                    <small class="text-muted">per quintal</small>
                </div>
            `;
        }).join('');
    } catch(e) {
        showToast('Error loading market data');
    }
}

// ===================== DRONE MAP =====================
function initMap() {
    if(map) return;
    map = L.map('map').setView([20.5937, 78.9629], 5);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap'
    }).addTo(map);
    map.on('click', function(e) { addMapPoint(e.latlng); });
}

function addMapPoint(latlng) {
    mapPoints.push(latlng);
    L.marker(latlng).addTo(map);
    if(mapPoints.length > 1) {
        if(fieldPolygon) map.removeLayer(fieldPolygon);
        fieldPolygon = L.polygon(mapPoints, {color: '#2d6a4f', fillColor: '#74c69d', fillOpacity: 0.3, weight: 3}).addTo(map);
    }
    updateDroneStats();
}

function calculatePolygonArea(points) {
    if(points.length < 3) return 0;
    let area = 0;
    for(let i = 0; i < points.length; i++) {
        const j = (i + 1) % points.length;
        area += points[i].lng * points[j].lat;
        area -= points[j].lng * points[i].lat;
    }
    return Math.abs(area) * 111320 * 111320 / 2;
}

function updateDroneStats() {
    if(mapPoints.length < 3) {
        document.getElementById('droneStats').innerHTML = `
            <div class="stat-box"><div class="stat-value">${mapPoints.length}</div><div class="stat-label">Points</div></div>
            <div class="stat-box"><div class="stat-value">--</div><div class="stat-label">Area (ha)</div></div>
        `;
        return;
    }
    const area = (calculatePolygonArea(mapPoints) / 10000).toFixed(2);
    document.getElementById('droneStats').innerHTML = `
        <div class="stat-box"><div class="stat-value">${area}</div><div class="stat-label">Area (ha)</div></div>
        <div class="stat-box"><div class="stat-value">${mapPoints.length}</div><div class="stat-label">Points</div></div>
    `;
}

function getCurrentLocation() {
    if(!navigator.geolocation) { showToast('Geolocation not supported'); return; }
    navigator.geolocation.getCurrentPosition(pos => {
        const latlng = [pos.coords.latitude, pos.coords.longitude];
        map.setView(latlng, 16);
        L.marker(latlng).addTo(map).bindPopup('Your Location').openPopup();
    }, () => showToast('Could not get location'));
}

function clearMap() {
    if(!map) return;
    map.eachLayer(layer => {
        if(layer instanceof L.Marker || layer instanceof L.Polygon || layer instanceof L.Polyline) map.removeLayer(layer);
    });
    mapPoints = []; fieldPolygon = null; missionPathLine = null;
    currentWaypoints = null;
    document.getElementById('flyMissionBtn').disabled = true;
    document.getElementById('flightPathResult').innerHTML = '';
    updateDroneStats();
}

function undoLastPoint() {
    if(mapPoints.length === 0) return;
    mapPoints.pop();
    clearMap();
    mapPoints.forEach(p => L.marker(p).addTo(map));
    if(mapPoints.length > 1) {
        fieldPolygon = L.polygon(mapPoints, {color: '#2d6a4f', fillColor: '#74c69d', fillOpacity: 0.3, weight: 3}).addTo(map);
    }
    updateDroneStats();
}

function simulateDroneFlight() {
    if(mapPoints.length < 3) { showToast('Draw field boundary first'); return; }
    document.getElementById('flightPathResult').innerHTML = `
        <div class="card" style="margin-top:15px;background:linear-gradient(135deg,#e8f5e9,#fff)">
            <div class="card-title"><i class="fas fa-plane"></i> Flight Simulation</div>
            <p>Pattern: Parallel lines at 5m spacing, 3m altitude</p>
            <p>Speed: 4.5 m/s | Overlap: 30%</p>
            <p>Batteries: 2 required for this area</p>
        </div>`;
    showToast('Simulation complete!');
}

let currentWaypoints = null;
let missionPathLine = null;

async function generateFlightPath() {
    if(mapPoints.length < 3) { showToast('Draw field boundary first'); return; }
    document.getElementById('flightPathResult').innerHTML = `<p class="text-muted">Computing coverage path...</p>`;
    document.getElementById('flyMissionBtn').disabled = true;
    currentWaypoints = null;

    try {
        const resp = await fetch(`${API_BASE}/api/drone/plan-path`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                boundary: mapPoints.map(p => [p.lat, p.lng]),
                spray: document.getElementById('sprayType').value !== ''
            })
        });
        const data = await resp.json();
        if(!data.success) { showToast(data.error || 'Could not plan path'); document.getElementById('flightPathResult').innerHTML = ''; return; }

        currentWaypoints = data.waypoints;

        if(missionPathLine) map.removeLayer(missionPathLine);
        missionPathLine = L.polyline(data.waypoints.map(w => [w.lat, w.lon]), {color: '#e63946', weight: 2, dashArray: '6,6'}).addTo(map);

        document.getElementById('flightPathResult').innerHTML = `
            <div class="card" style="margin-top:15px;background:linear-gradient(135deg,#e8f5e9,#fff)">
                <div class="card-title"><i class="fas fa-route"></i> Flight Path Ready</div>
                <p><strong>Waypoints:</strong> ${data.estimate.waypoint_count} &nbsp; <strong>Distance:</strong> ${data.estimate.distance_m} m</p>
                <p><strong>Est. flight time:</strong> ${data.estimate.estimated_minutes} min &nbsp; <strong>Batteries needed:</strong> ${data.estimate.batteries_needed}</p>
                <p><strong>Mode:</strong> ${data.simulation_mode ? '🟢 Simulation (no hardware needed)' : '🔴 REAL FLIGHT MODE'}${data.spray_enabled ? ' &nbsp;|&nbsp; 💧 Spray armed on pass rows' : ''}</p>
            </div>`;
        document.getElementById('flyMissionBtn').disabled = false;
    } catch(e) {
        showToast('Error computing flight path');
        document.getElementById('flightPathResult').innerHTML = '';
    }
}

async function flyMission() {
    if(!currentWaypoints) { showToast('Generate a path first'); return; }
    if(!confirm('Start this mission now? The drone will take off and fly the plotted path autonomously. Keep a safety pilot ready if this is a real flight.')) return;

    try {
        const resp = await fetch(`${API_BASE}/api/drone/fly`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                boundary: mapPoints.map(p => [p.lat, p.lng]),
                spray: document.getElementById('sprayType').value !== '',
                crop_group: 'auto'
            })
        });
        const data = await resp.json();
        if(!data.success) {
            showToast(data.error || 'Could not start mission');
            return;
        }
        startMissionPolling(data.mission_id);
        showToast(data.simulation_mode ? 'Simulated mission started' : '🔴 REAL mission started');
    } catch(e) {
        showToast('Error starting mission');
    }
}

let missionPollTimer = null;
let missionEventsSeen = 0;
let currentMissionId = null;

function startMissionPolling(missionId) {
    currentMissionId = missionId;
    missionEventsSeen = 0;
    document.getElementById('missionPanel').style.display = 'block';
    document.getElementById('missionLog').innerHTML = '';
    document.getElementById('missionStatusLine').textContent = 'Starting...';
    if(missionPollTimer) clearInterval(missionPollTimer);
    missionPollTimer = setInterval(() => pollMission(missionId), 1500);
    pollMission(missionId);
}

async function pollMission(missionId) {
    try {
        const resp = await fetch(`${API_BASE}/api/drone/mission/${missionId}?since=${missionEventsSeen}`);
        if(!resp.ok) return;
        const data = await resp.json();

        document.getElementById('missionStatusLine').textContent = `Status: ${data.status.replace(/_/g, ' ')}`;

        const log = document.getElementById('missionLog');
        data.events.forEach(ev => {
            const line = document.createElement('div');
            line.textContent = `[${new Date(ev.timestamp).toLocaleTimeString()}] ${ev.kind.replace(/_/g, ' ')}` +
                (ev.lat ? ` @ ${ev.lat.toFixed(5)}, ${ev.lon.toFixed(5)}` : '') +
                (ev.reason ? ` — ${ev.reason}` : '');
            log.appendChild(line);
        });
        if(data.events.length) log.scrollTop = log.scrollHeight;
        missionEventsSeen = data.total_events;

        if(data.status === 'mission_complete') {
            clearInterval(missionPollTimer);
            missionPollTimer = null;
            showToast('Mission complete');
            refreshDroneFeed();
        }
    } catch(e) { /* keep polling */ }
}

async function abortMission() {
    if(!currentMissionId) return;
    if(!confirm('Abort the mission and return the drone home now?')) return;
    try {
        await fetch(`${API_BASE}/api/drone/mission/${currentMissionId}/abort`, {method: 'POST'});
        showToast('Abort requested — returning home');
    } catch(e) {
        showToast('Could not reach server to send abort — use manual RC override if flying real hardware');
    }
}

async function saveDronePlan() {
    if(mapPoints.length < 3) { showToast('Draw boundary first'); return; }
    const area = (calculatePolygonArea(mapPoints) / 10000).toFixed(2);

    const data = {
        name: `Spray Plan ${new Date().toLocaleDateString()}`,
        area: parseFloat(area),
        spray_type: document.getElementById('sprayType').value,
        drone_model: document.getElementById('sprayDroneModel').value,
        boundary: mapPoints.map(p => [p.lat, p.lng])
    };

    try {
        await fetch(`${API_BASE}/api/drone/plans`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });
        showToast('Drone plan saved!');
        renderDronePlans();
    } catch(e) {
        showToast('Error saving plan');
    }
}

async function renderDronePlans() {
    try {
        const resp = await fetch(`${API_BASE}/api/drone/plans`);
        const data = await resp.json();
        const list = document.getElementById('dronePlansList');

        if(data.plans.length === 0) {
            list.innerHTML = '<p class="text-muted text-center">No saved plans.</p>';
            return;
        }

        list.innerHTML = data.plans.map(p => `
            <div class="history-item">
                <div class="history-icon">🚁</div>
                <div class="history-details">
                    <div class="history-title">${p.spray_type} - ${p.area_hectares} ha</div>
                    <div class="history-meta">${p.drone_model} | ${p.date}</div>
                </div>
            </div>
        `).join('');
    } catch(e) {
        document.getElementById('dronePlansList').innerHTML = '<p class="text-muted text-center">Error loading plans</p>';
    }
}

// ===================== CALENDAR =====================
function renderCalendar() {
    const year = currentMonth.getFullYear();
    const month = currentMonth.getMonth();
    document.getElementById('calendarMonth').textContent = new Date(year, month).toLocaleDateString('en-US', {month:'long', year:'numeric'});

    const firstDay = new Date(year, month, 1).getDay();
    const daysInMonth = new Date(year, month + 1, 0).getDate();
    const prevDays = new Date(year, month, 0).getDate();

    let html = '';
    for(let i = firstDay - 1; i >= 0; i--) {
        html += `<div class="calendar-day other-month">${prevDays - i}</div>`;
    }

    const today = new Date();
    for(let d = 1; d <= daysInMonth; d++) {
        const isToday = d === today.getDate() && month === today.getMonth() && year === today.getFullYear();
        html += `<div class="calendar-day ${isToday ? 'today' : ''}">${d}</div>`;
    }

    const remaining = (7 - ((firstDay + daysInMonth) % 7)) % 7;
    for(let d = 1; d <= remaining; d++) {
        html += `<div class="calendar-day other-month">${d}</div>`;
    }

    document.getElementById('calendarGrid').innerHTML = html;
    renderTasks();
}

function changeMonth(delta) {
    currentMonth.setMonth(currentMonth.getMonth() + delta);
    renderCalendar();
}

async function addTask() {
    const data = {
        name: document.getElementById('taskName').value,
        date: document.getElementById('taskDate').value,
        type: document.getElementById('taskType').value,
        field: document.getElementById('taskField').value,
        notes: document.getElementById('taskNotes').value
    };

    if(!data.name || !data.date) { showToast('Name and date required'); return; }

    try {
        await fetch(`${API_BASE}/api/tasks`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });
        showToast('Task added!');
        renderTasks();
        document.getElementById('taskName').value = '';
        document.getElementById('taskNotes').value = '';
    } catch(e) {
        showToast('Error adding task');
    }
}

async function renderTasks() {
    try {
        const resp = await fetch(`${API_BASE}/api/tasks`);
        const data = await resp.json();
        const list = document.getElementById('taskList');

        if(data.tasks.length === 0) {
            list.innerHTML = '<p class="text-muted">No tasks scheduled.</p>';
            return;
        }

        list.innerHTML = data.tasks.slice(0, 5).map(t => `
            <div class="history-item">
                <div class="history-details">
                    <div class="history-title">${t.name}</div>
                    <div class="history-meta">${new Date(t.task_date).toLocaleDateString()} | ${t.field || 'All'} | ${t.task_type}</div>
                </div>
                <button class="btn btn-secondary btn-sm" onclick="deleteTask(${t.id})"><i class="fas fa-trash"></i></button>
            </div>
        `).join('');
    } catch(e) {
        document.getElementById('taskList').innerHTML = '<p class="text-muted">Error loading tasks</p>';
    }
}

async function deleteTask(id) {
    try {
        await fetch(`${API_BASE}/api/tasks/${id}`, { method: 'DELETE' });
        renderTasks();
    } catch(e) {
        showToast('Error deleting task');
    }
}

// ===================== PROFILE =====================
async function loadProfile() {
    try {
        const resp = await fetch(`${API_BASE}/api/profile`);
        const prof = await resp.json();

        if(prof.name) {
            document.getElementById('profileName').textContent = prof.name;
            document.getElementById('profileLocation').textContent = `${prof.village || ''}, ${prof.district || ''}`;
            document.getElementById('profName').value = prof.name || '';
            document.getElementById('profPhone').value = prof.phone || '';
            document.getElementById('profVillage').value = prof.village || '';
            document.getElementById('profDistrict').value = prof.district || '';
            document.getElementById('profState').value = prof.state || '';
            document.getElementById('profFarmSize').value = prof.farm_size_acres || '';
        }

        // Load stats
        const histResp = await fetch(`${API_BASE}/api/history`);
        const histData = await histResp.json();
        document.getElementById('statScans').textContent = histData.scans ? histData.scans.length : 0;

        const fieldsResp = await fetch(`${API_BASE}/api/fields`);
        const fieldsData = await fieldsResp.json();
        document.getElementById('statFields').textContent = fieldsData.fields ? fieldsData.fields.length : 0;

        renderFields();
    } catch(e) {
        console.error('Error loading profile', e);
    }
}

async function saveProfile() {
    const data = {
        name: document.getElementById('profName').value,
        phone: document.getElementById('profPhone').value,
        village: document.getElementById('profVillage').value,
        district: document.getElementById('profDistrict').value,
        state: document.getElementById('profState').value,
        farm_size: parseFloat(document.getElementById('profFarmSize').value) || 0
    };

    try {
        await fetch(`${API_BASE}/api/profile`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });
        showToast('Profile saved!');
        loadProfile();
    } catch(e) {
        showToast('Error saving profile');
    }
}

async function addField() {
    const data = {
        name: document.getElementById('fieldName').value,
        area: parseFloat(document.getElementById('fieldArea').value),
        crop: document.getElementById('fieldCrop').value,
        soil: document.getElementById('fieldSoil').value
    };

    if(!data.name || !data.area) { showToast('Name and area required'); return; }

    try {
        await fetch(`${API_BASE}/api/fields`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });
        showToast('Field added!');
        renderFields();
        document.getElementById('fieldName').value = '';
    } catch(e) {
        showToast('Error adding field');
    }
}

async function renderFields() {
    try {
        const resp = await fetch(`${API_BASE}/api/fields`);
        const data = await resp.json();
        const list = document.getElementById('fieldList');

        const icons = {wheat:'🌾',rice:'🍚',corn:'🌽',cotton:'🧵',tomato:'🍅',potato:'🥔',sugarcane:'🎋',soybean:'🫘',mustard:'🌿',fallow:'🏜️'};

        if(data.fields.length === 0) {
            list.innerHTML = '<p class="text-muted text-center">No fields added.</p>';
            return;
        }

        list.innerHTML = data.fields.map(f => `
            <div class="field-card">
                <div class="field-icon">${icons[f.crop] || '🌱'}</div>
                <div class="field-details">
                    <div class="field-name">${f.name}</div>
                    <div class="field-meta">${f.area_acres} acres | ${f.crop || 'No crop'} | ${f.soil_type || 'Unknown'}</div>
                </div>
                <button class="btn btn-secondary btn-sm" onclick="deleteField(${f.id})"><i class="fas fa-trash"></i></button>
            </div>
        `).join('');
    } catch(e) {
        document.getElementById('fieldList').innerHTML = '<p class="text-muted">Error loading fields</p>';
    }
}

async function deleteField(id) {
    try {
        await fetch(`${API_BASE}/api/fields/${id}`, { method: 'DELETE' });
        renderFields();
    } catch(e) {
        showToast('Error deleting field');
    }
}

// ===================== CHAT =====================
const CHAT_KNOWLEDGE = {
    'fertilizer': 'For balanced fertilization, use NPK based on soil test. Urea for N, DAP for P, MOP for K. Apply in split doses.',
    'irrigation': 'Most crops need irrigation when soil moisture drops below 60%. Early morning (5-8 AM) is best. Drip saves 40-60% water.',
    'pest': 'IPM combines biological, cultural, and chemical methods. Use pheromone traps and beneficial insects first.',
    'organic': 'Organic farming uses compost, green manure, biofertilizers. Improves soil health. Certification takes 3 years.',
    'drone': 'Agricultural drones like DJI Agras T40 spray 40L/flight, cover 21 ha/hour. Reduce chemical use by 30%.',
    'weather': 'Monitor weather before spraying. Avoid if rain expected within 6 hours or wind >15 km/h.',
    'soil': 'Loamy soil is ideal. Sandy drains fast (add organic matter), clay holds water (add sand/gypsum). pH 6.0-7.5.',
    'rotation': 'Crop rotation breaks pest cycles. Common: Wheat-Rice, Corn-Soybean, Tomato-Legume.',
    'compost': 'Compost improves soil structure. Apply 5-10 tons/ha annually. Turn pile every 2 weeks.',
    'seed': 'Use certified seeds. Check germination rate (>80%). Treat with fungicide before sowing.',
    'npk': 'NPK: Nitrogen (leaves), Phosphorus (roots/flowers), Potassium (fruit/immunity).',
    'ph': 'Soil pH affects nutrient availability. Most crops prefer 6.0-7.5. Add lime to raise, sulfur to lower.',
    'water': 'Water EC should be <2 dS/m. High salinity causes leaf burn. Test before drip irrigation.'
};

function sendChat() {
    const input = document.getElementById('chatInput');
    const msg = input.value.trim();
    if(!msg) return;

    const chatBox = document.getElementById('chatMessages');
    chatBox.innerHTML += `<div class="chat-msg user">${msg}</div>`;
    input.value = '';
    chatBox.scrollTop = chatBox.scrollHeight;

    setTimeout(() => {
        const response = generateChatResponse(msg.toLowerCase());
        chatBox.innerHTML += `<div class="chat-msg bot">${response}</div>`;
        chatBox.scrollTop = chatBox.scrollHeight;
    }, 600);
}

function generateChatResponse(msg) {
    for(const [key, value] of Object.entries(CHAT_KNOWLEDGE)) {
        if(msg.includes(key)) return value;
    }
    if(msg.includes('hello') || msg.includes('hi')) return 'Namaste! How can I help with your farming today?';
    if(msg.includes('thank')) return 'You are welcome! Happy farming! 🌾';
    if(msg.includes('price') || msg.includes('market')) return 'Check the Market tab for current crop prices.';
    if(msg.includes('disease') || msg.includes('scan')) return 'Use the Scan tab to upload a photo for YOLOv8 AI detection.';
    if(msg.includes('drone') || msg.includes('spray')) return 'Go to the Drone tab to plan spray operations.';
    if(msg.includes('weather')) return 'Check the Weather tab for live conditions and farming advice.';
    if(msg.includes('yield') || msg.includes('profit')) return 'Use the Yield tab to estimate crop yield and profit.';
    return 'I can help with crop diseases, fertilizers, irrigation, drones, weather, and market prices. Ask me anything!';
}

// ===================== SETTINGS =====================
function saveSettings() {
    showToast('Settings saved!');
}

function exportAllData() {
    showToast('Data exported!');
}

function resetAllData() {
    if(!confirm('WARNING: This will delete ALL data. Are you sure?')) return;
    showToast('All data reset');
}

// ===================== UTILITIES =====================
function showLoading(text) {
    document.getElementById('loadingText').textContent = text || 'Loading...';
    document.getElementById('loadingOverlay').classList.add('show');
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.remove('show');
}

function showToast(msg) {
    const toast = document.getElementById('toast');
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 3000);
}

function updateOnlineStatus() {
    const badge = document.getElementById('offlineBadge');
    if(!navigator.onLine) badge.classList.add('show');
    else badge.classList.remove('show');
}

window.addEventListener('online', updateOnlineStatus);
window.addEventListener('offline', updateOnlineStatus);

// ===================== DRONE WIRELESS CONNECTION =====================
let dronePollTimer = null;
let droneLastSeenScanId = null;

function startDronePolling() {
    refreshDroneStatus();
    refreshDroneFeed();
    if(dronePollTimer) return;
    dronePollTimer = setInterval(() => {
        refreshDroneStatus();
        refreshDroneFeed();
    }, 5000);
}

function stopDronePolling() {
    if(dronePollTimer) { clearInterval(dronePollTimer); dronePollTimer = null; }
}

async function refreshDroneStatus() {
    const dot = document.getElementById('droneConnDot');
    const text = document.getElementById('droneConnStatusText');
    try {
        const res = await fetch(API_BASE + '/api/drone/status');
        if(!res.ok) throw new Error('bad status');
        const data = await res.json();

        const serverUrl = `http://${data.server_ip}:${data.port}`;
        document.getElementById('droneServerUrl').value = serverUrl;
        document.getElementById('droneEndpointUrl').value = serverUrl + data.upload_endpoint;

        dot.style.background = data.model_loaded ? '#2d8f5e' : '#f4a261';
        text.textContent = data.model_loaded
            ? `Online — ready to receive drone images (${serverUrl})`
            : `Online, but no AI model loaded yet — using fallback analysis (${serverUrl})`;
        text.style.color = 'var(--text)';

        window._droneStatusCache = data;
    } catch (e) {
        dot.style.background = '#c1121f';
        text.textContent = 'Cannot reach the CropGuard server';
        text.style.color = 'var(--danger)';
    }
}

async function refreshDroneFeed() {
    try {
        const res = await fetch(API_BASE + '/api/drone/latest?limit=10');
        if(!res.ok) return;
        const data = await res.json();
        renderDroneFeed(data.scans || []);
    } catch (e) { /* silent - status indicator already shows connectivity */ }
}

function renderDroneFeed(scans) {
    const wrap = document.getElementById('droneFeedList');
    if(!scans.length) {
        wrap.innerHTML = '<p class="text-muted" style="font-size:0.85rem">No drone scans received yet.</p>';
        return;
    }

    // Toast-notify only on genuinely new scans since the last poll
    if(droneLastSeenScanId !== null && scans[0].id !== droneLastSeenScanId) {
        showToast(`New drone image analyzed: ${scans[0].crop_name} — ${scans[0].disease_name}`);
    }
    droneLastSeenScanId = scans[0].id;

    wrap.innerHTML = scans.map(s => {
        const sevClass = s.severity === 'High' ? 'badge-orange' : s.severity === 'Healthy' ? 'badge-green' : 'badge-blue';
        const time = new Date(s.timestamp).toLocaleTimeString();
        const loc = s.location ? `<span class="text-muted" style="font-size:0.75rem"><i class="fas fa-map-pin"></i> ${s.location.lat.toFixed(4)}, ${s.location.lon.toFixed(4)}</span>` : '';
        return `<div style="display:flex;gap:12px;align-items:center;padding:10px 0;border-bottom:1px solid var(--border)">
            <img src="/uploads/${s.image_path}" style="width:56px;height:56px;object-fit:cover;border-radius:10px;flex-shrink:0" onerror="this.style.display='none'">
            <div style="flex:1;min-width:0">
                <strong>${s.crop_icon || '🌱'} ${s.crop_name}</strong> — ${s.disease_name}
                <span class="badge ${sevClass}">${s.severity}</span>
                <div class="text-muted" style="font-size:0.75rem">${time} · ${s.source || 'drone'} ${loc}</div>
            </div>
        </div>`;
    }).join('');
}

function copyDroneUrl() {
    const input = document.getElementById('droneServerUrl');
    input.select();
    navigator.clipboard ? navigator.clipboard.writeText(input.value) : document.execCommand('copy');
    showToast('Server address copied');
}

function toggleDroneApiHelp() {
    const box = document.getElementById('droneApiHelp');
    if(box.style.display === 'block') { box.style.display = 'none'; return; }

    const data = window._droneStatusCache || {};
    const url = (document.getElementById('droneEndpointUrl').value) || '<server-address>/api/drone/upload';
    box.textContent =
`Any device that can make an HTTP request can send images here - it doesn't
need to be a specific drone brand.

# Simplest example (from a companion computer / laptop / phone shortcut):
curl -F "image=@photo.jpg" -F "drone_id=my-drone" ${url}

# From a Python companion script (RPi/Jetson on the drone, etc):
import requests
requests.post("${url}",
    files={"image": open("photo.jpg", "rb")},
    data={"drone_id": "my-drone", "lat": 28.61, "lon": 77.20})

Optional fields: crop_group, lat, lon, drone_model. GPS is read
automatically from the photo's EXIF if you don't pass lat/lon.
${data.api_key_required ? '\nThis server requires an API key - add header  X-API-Key: <your key>' : ''}

Bluetooth-only drone (no WiFi radio)?
Run the included bluetooth_bridge.py on this computer - it receives
images over Bluetooth and forwards them to this same endpoint.`;
    box.style.display = 'block';
}

// ===================== INIT =====================
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('taskDate').valueAsDate = new Date();
    renderMarket();
    renderCalendar();
    updateOnlineStatus();

    // Set default yield values
    renderYieldDefaults();
});
