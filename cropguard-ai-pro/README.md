# CropGuard AI Pro v3.0

A Flask-based crop disease detection web app powered by YOLOv8, with weather advice, market prices, fertilizer/irrigation/yield calculators, drone spray planning, and farm task management.

This copy has been debugged and restructured into a proper Flask project (see **"What was fixed"** at the bottom).

---

## 1. What you need first

- **Python 3.10 or 3.11** (recommended — Ultralytics/PyTorch are most stable on these)
- pip
- ~5 GB free disk space (PyTorch + Ultralytics are large downloads)
- Optional but recommended: a GPU with CUDA if you want fast training/inference. CPU works fine for inference, just slower.

Check your Python version:
```bash
python3 --version
```

---

## 2. Project structure

```
cropguard-ai-pro/
├── app.py                  # Main Flask app & all API routes
├── config.py                # Environment-based configuration
├── models.py                 # SQLAlchemy database models
├── disease_database.py       # Crop/disease/market reference data
├── yolo_detector.py          # YOLOv8 wrapper (load, detect, train, export)
├── wsgi.py                   # Production entry point (gunicorn)
├── requirements.txt
├── Procfile                  # For Heroku/Railway-style deploys
├── .env.example               # Copy this to .env and edit
├── .gitignore
├── templates/
│   └── index.html             # Single-page app UI
└── static/
    ├── js/app.js               # Frontend logic (calls the API)
    ├── uploads/                 # Scanned images land here (auto-created)
    └── models/                   # Put your trained best.pt here
```

---

## 3. Install (step by step)

```bash
# 1. Unzip and enter the folder
cd cropguard-ai-pro

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

This installs Flask, PyTorch, Ultralytics (YOLOv8), OpenCV, and the rest. The PyTorch/Ultralytics install alone can take several minutes on the first run.

> If you don't have a GPU and want a smaller/faster install, you can install the CPU-only PyTorch build instead:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

```bash
# 4. Set up your environment file
cp .env.example .env
```
Open `.env` and at minimum change `SECRET_KEY` to a random string before deploying anywhere public. Generate one with:
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

---

## 4. Add your YOLOv8 model

The app looks for a trained model at `static/models/best.pt` (configurable via `MODEL_PATH` in `.env`).

- **If you already trained a model**, copy your `best.pt` weights file into `static/models/`.
- **If you don't have one yet**, the app still runs — it automatically falls back to a color/HSV-based image analysis (not real disease detection, but a working demo path) whenever no model is found. You'll see this in the terminal:
  ```
  ⚠️ Model not found at static/models/best.pt. Using fallback detection.
  ```
- **If you want to start from a general pretrained YOLOv8 checkpoint** (not crop-disease-specific, but useful for testing that inference works):
  ```bash
  python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
  cp yolov8n.pt static/models/best.pt
  ```
  Note: this won't produce meaningful *disease* labels since it's trained on general objects, not crop diseases — it's just useful to confirm the pipeline runs.

---

## 5. Run it locally

```bash
python3 app.py
```

Then open **http://localhost:5000** in your browser.

Health check:
```bash
curl http://localhost:5000/api/health
```

---

## 6. Training your own disease-detection model

1. Prepare a YOLO-format dataset (images + label `.txt` files + a `data.yaml`). Class names should follow the pattern `crop_disease`, e.g. `tomato_early_blight`, `rice_blast`, `wheat_rust` — the app splits on `_`/`-` to identify crop vs. disease, and matches this against `disease_database.py`. If your class names don't match a known crop key there, the app will still show the detection but with generic treatment advice.

2. Train via the API:
   ```bash
   curl -X POST http://localhost:5000/api/train \
     -H "Content-Type: application/json" \
     -d '{"data_yaml": "/path/to/data.yaml", "epochs": 100, "imgsz": 640}'
   ```
   Or directly in Python:
   ```python
   from yolo_detector import get_detector
   d = get_detector('static/models/best.pt')
   d.train(data_yaml='/path/to/data.yaml', epochs=100)
   ```

3. After training, Ultralytics saves the best weights under `runs/train/cropguard/weights/best.pt` — copy that into `static/models/best.pt` and restart the app to use it.

4. To keep `disease_database.py` in sync with new classes, add entries for each new `crop_disease` under `CROP_DISEASE_DB` (severity, description, symptoms, chemical/organic/prevention treatments) so the app shows real advice instead of the generic fallback text.

---

## 7. Deploying to production

**Don't use `python3 app.py` in production** — it's Flask's development server. Use gunicorn:

```bash
gunicorn --bind 0.0.0.0:8000 --workers 2 --timeout 120 wsgi:app
```

The `--timeout 120` matters: YOLOv8 inference on CPU can take a few seconds per image, and the default gunicorn timeout (30s) can kill slow requests under load.

Set `FLASK_ENV=production` in your `.env` — this enables secure cookies and tighter security settings from `config.py`.

**Platforms this works well on out of the box (via the included `Procfile`):**
- Railway, Render, Heroku-style platforms: push the repo, set `.env` values in their dashboard as environment variables, done.
- A VPS (DigitalOcean, EC2, etc.): run gunicorn behind nginx as a reverse proxy, use a process manager like `systemd` or `supervisor` to keep it alive.

**A note on scale:** the free `flask-limiter` rate limiting in `config.py` defaults to in-memory storage, which won't work correctly across multiple gunicorn workers/instances. If you deploy with more than 1 worker, set `REDIS_URL` in `.env` so rate limiting is shared correctly.

---

## 8. API reference (quick overview)

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | Server + model status |
| `/api/scan` | POST | Upload image (`image` field), get disease detection |
| `/api/history` | GET | Past scans |
| `/api/history/<id>` | DELETE | Delete a scan |
| `/api/history/export` | GET | Download history as CSV |
| `/api/weather` | GET | Weather + farming advice (`lat`, `lon` params) |
| `/api/market` | GET | Crop market prices |
| `/api/fertilizer` | POST | NPK fertilizer recommendation |
| `/api/irrigation` | POST | Irrigation status/advice |
| `/api/yield` | POST | Yield & profit calculator |
| `/api/profile` | GET/POST | Farmer profile |
| `/api/fields` | GET/POST | Farm fields |
| `/api/fields/<id>` | DELETE | Delete a field |
| `/api/tasks` | GET/POST | Farming calendar tasks |
| `/api/tasks/<id>` | DELETE/PATCH | Delete/update a task |
| `/api/drone/plans` | GET/POST | Drone spray planning |
| `/api/train` | POST | Kick off YOLOv8 training |
| `/api/drone/connect-info` | GET | WiFi upload URL + QR code for any drone/companion app |
| `/api/drone/upload` | POST | Live drone image ingestion → instant AI result (any image format) |
| `/api/drone/devices` | GET | Currently/recently connected drones (WiFi or Bluetooth) |
| `/api/drone/latest` | GET | Most recent drone-sourced scan result (for live polling) |

---

## 8b. Live drone connectivity (WiFi + Bluetooth)

The **Drone** tab now has a "Live Drone Connect" panel. This works with **any drone brand** — there's no proprietary SDK lock-in, just a plain HTTP endpoint any drone, companion app, or onboard computer can POST an image to.

**WiFi (primary path):**
1. Open the Drone tab — it shows this server's local network address and a QR code.
2. Have your drone's companion app, or an onboard computer (e.g. a Raspberry Pi on the drone) POST the photo as `multipart/form-data` field `image` to that URL. Raw binary bodies and base64 JSON (`{"image_base64": "..."}`) are also accepted, for devices that can't build multipart requests.
3. The server also broadcasts a small UDP "here I am" beacon every few seconds on port `41234` (payload: `CROPGUARD_SERVER:<ip>:<port>`), so a device can auto-discover the server instead of you typing in an IP manually.
4. Results appear on the dashboard automatically within a few seconds — no page refresh needed.

**Bluetooth (fallback path):**
- From the dashboard, click **"Connect via Bluetooth"** — this uses the browser's Web Bluetooth API to pair directly with drones/controllers that expose a BLE service (not supported in every browser, notably iOS Safari).
- For classic Bluetooth (RFCOMM/SPP) devices — more broadly compatible with companion boards and flight controllers — run `python bluetooth_bridge.py` in a separate terminal (needs `pip install pybluez2` and a Bluetooth adapter). It receives images over Bluetooth and forwards them into the same pipeline, so WiFi and Bluetooth results end up in the same history either way.

**Any image format:** `image_utils.py` normalizes whatever comes in — JPEG, PNG, WEBP, BMP, TIFF, GIF, HEIC/HEIF (iPhone-style captures), CMYK/paletted/grayscale/RGBA images, EXIF-rotated photos, even RAW sensor formats if `rawpy` is installed — into a clean RGB JPEG before it reaches the model. Corrupted or undecodable data returns a clear `400` error instead of crashing.

**Security note:** by default any device on the network can POST to `/api/drone/upload` (lowest friction for getting started). To lock it down, set `DRONE_API_KEY` in your `.env` file — the endpoint will then require an `X-Drone-Api-Key` header matching it.

---

## 9. What was fixed in this copy

I ran the app end-to-end (booted the server, hit every endpoint, uploaded a test image) rather than just reading the code, and found/fixed:

1. **Missing `templates/` and `static/js/` folders.** Your `index.html` and `app.js` were flat files, but `index.html` references `app.js` via Flask's `url_for('static', ...)` and `app.py` calls `render_template('index.html')` — both require the standard Flask folder layout. Restructured accordingly.
2. **Upload folder never auto-created.** On a fresh install, the first image scan would crash with `FileNotFoundError` because `static/uploads/` didn't exist yet and nothing created it. Fixed: `app.py` now creates it (and the models folder) on startup.
3. **Severity mismatch bug.** In fallback detection mode (no trained model), the detector correctly computes a severity (`High`/`Medium`/`Low`/`Healthy`), but the API response ignored it and pulled `"Unknown"` from the disease database lookup instead. Fixed so the real computed severity is used.
4. **Two frontend crash bugs in `app.js`:**
   - It referenced an `offlineBadge` element that didn't exist in the HTML, which threw an error on every page load and silently broke code that ran after it (the yield calculator's default values never got set). Added the missing element + styling.
   - It referenced `soilN`/`soilP` input fields for the fertilizer calculator that didn't exist in the form, so clicking "Calculate" always crashed. Added the missing (optional) soil-test input fields — the backend already supported them.
5. Removed unused imports (`Path`, unused `session`) for cleanliness.
6. Added `wsgi.py`, `Procfile`, and `.gitignore` for a smoother production deployment, since `gunicorn` was already in `requirements.txt` but had no entry point to run.

Everything above was verified by actually running the server and calling `/api/scan`, `/api/fertilizer`, `/api/irrigation`, `/api/yield`, `/api/market`, `/api/history`, and `/api/health`, and loading the page — all pass.

---

## 10. Known limitations (not bugs, just things to know)

- **Single-user profile**: `/api/profile` stores one global farmer profile, not per-account. There's no login system. If you want multiple farmers using the same deployment with separate data, that needs an auth layer added.
- **Weather API** (Open-Meteo) needs outbound internet access from wherever you host this — no API key required, but it must be able to reach `api.open-meteo.com`.
- **Fallback detection is not real disease detection.** It's a rough HSV color-based heuristic for demo purposes when no trained model is present. Real accuracy requires training on an actual labeled crop-disease dataset.
- **CORS is wide open** (`CORS(app)` allows all origins). Fine for a single-origin deployment; restrict it in `app.py` if you expose the API separately from the frontend.
