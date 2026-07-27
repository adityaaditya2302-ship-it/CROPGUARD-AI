const API = '/api/dronehub';

let map, drawnItems, currentBoundary = null;
let connectedDeviceId = null;
let currentMissionId = null;
let telemetryTimer = null;

function initMap() {
  map = L.map('map').setView([28.6139, 77.2090], 15);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(map);

  drawnItems = new L.FeatureGroup();
  map.addLayer(drawnItems);

  const drawControl = new L.Control.Draw({
    draw: {
      polygon: true, polyline: false, rectangle: false,
      circle: false, marker: false, circlemarker: false
    },
    edit: { featureGroup: drawnItems }
  });
  map.addControl(drawControl);

  map.on(L.Draw.Event.CREATED, function (e) {
    drawnItems.clearLayers();
    drawnItems.addLayer(e.layer);
    const latlngs = e.layer.getLatLngs()[0];
    currentBoundary = latlngs.map(p => [p.lat, p.lng]);
  });
}

function setStatus(state) {
  const dot = document.getElementById('connDot');
  const text = document.getElementById('connText');
  dot.className = 'dot ' + (state === 'connected' ? 'dot-green' : state === 'connecting' ? 'dot-yellow' : 'dot-red');
  text.textContent = state === 'connected' ? 'Connected' : state === 'connecting' ? 'Connecting...' : 'Disconnected';
}

async function scanDrones() {
  setStatus('connecting');
  const res = await fetch(`${API}/scan`);
  const data = await res.json();
  const list = document.getElementById('deviceList');
  list.innerHTML = '';
  data.devices.forEach(d => {
    const card = document.createElement('div');
    card.className = 'dh-device-card';
    card.innerHTML = `<b>${d.name}</b><span>Type: ${d.connection_type}</span><br>
      <button class="dh-btn" onclick="connectDevice(${d.id})">Connect</button>`;
    list.appendChild(card);
  });
  setStatus('disconnected');
}

async function connectDevice(deviceId) {
  setStatus('connecting');
  const res = await fetch(`${API}/connect/${deviceId}`, { method: 'POST' });
  if (res.ok) {
    connectedDeviceId = deviceId;
    setStatus('connected');
    startTelemetryPolling();
    checkWeather();
  } else {
    setStatus('disconnected');
    alert('Failed to connect to device ' + deviceId);
  }
}

function startTelemetryPolling() {
  if (telemetryTimer) clearInterval(telemetryTimer);
  telemetryTimer = setInterval(async () => {
    if (!connectedDeviceId) return;
    try {
      const res = await fetch(`${API}/telemetry/${connectedDeviceId}`);
      const t = await res.json();
      document.getElementById('t_battery').textContent = (t.battery_pct ?? '--') + '%';
      document.getElementById('t_sats').textContent = t.satellites ?? '--';
      document.getElementById('t_alt').textContent = (t.altitude_m ?? '--') + ' m';
      document.getElementById('t_speed').textContent = (t.speed_mps ?? '--') + ' m/s';
      document.getElementById('t_signal').textContent = (t.signal_strength_pct ?? '--') + '%';
      document.getElementById('t_mode').textContent = t.flight_mode ?? '--';
      document.getElementById('t_tank').textContent = (t.tank_pct ?? '--') + '%';
      document.getElementById('t_flow').textContent = (t.flow_rate_lpm ?? '--') + ' L/min';

      const hres = await fetch(`${API}/health/${connectedDeviceId}`);
      const h = await hres.json();
      document.getElementById('healthBox').textContent =
        h.safe_to_fly ? 'All systems nominal' : 'WARNING: ' + (h.warnings || []).join(', ');
    } catch (e) { /* silent - keep polling */ }
  }, 1000);
}

async function checkWeather() {
  const res = await fetch(`${API}/weather-check`);
  const data = await res.json();
  document.getElementById('weatherBox').textContent =
    data.safe_to_spray
      ? `Safe to fly. Wind ${data.weather.wind_speed_kmh ?? '?'} km/h`
      : `NOT SAFE - Wind ${data.weather.wind_speed_kmh ?? '?'} km/h / Rain: ${data.weather.rain}`;
}

async function generateMission() {
  if (!currentBoundary || currentBoundary.length < 3) {
    alert('Draw a field boundary on the map first (use the polygon tool).');
    return;
  }
  const res = await fetch(`${API}/mission/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      name: 'AI Mission ' + new Date().toLocaleTimeString(),
      boundary: currentBoundary,
      device_id: connectedDeviceId,
      disease_severity_pct: 40
    })
  });
  const mission = await res.json();
  if (mission.error) { alert(mission.error); return; }
  currentMissionId = mission.id;
  document.getElementById('missionSummary').textContent =
    `Mission #${mission.id}\nChemical: ${mission.chemical}\nDosage: ${mission.dosage_l_per_ha} L/ha\n` +
    `Speed: ${mission.speed_mps} m/s\nWaypoints: ${mission.waypoints.length}`;
}

async function missionAction(action) {
  if (!currentMissionId) { alert('Generate a mission first.'); return; }
  const res = await fetch(`${API}/mission/${currentMissionId}/${action}`, { method: 'POST' });
  const data = await res.json();
  if (action === 'complete') {
    document.getElementById('reportBox').textContent = JSON.stringify(data.report, null, 2);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  initMap();
  document.getElementById('scanBtn').onclick = scanDrones;
  document.getElementById('clearBoundaryBtn').onclick = () => { drawnItems.clearLayers(); currentBoundary = null; };
  document.getElementById('generateMissionBtn').onclick = generateMission;
  document.getElementById('uploadBtn').onclick = () => missionAction('upload');
  document.getElementById('startBtn').onclick = () => missionAction('start');
  document.getElementById('pauseBtn').onclick = () => missionAction('pause');
  document.getElementById('resumeBtn').onclick = () => missionAction('resume');
  document.getElementById('abortBtn').onclick = () => missionAction('abort');
  document.getElementById('rthBtn').onclick = () => missionAction('rth');
  document.getElementById('completeBtn').onclick = () => missionAction('complete');
});
