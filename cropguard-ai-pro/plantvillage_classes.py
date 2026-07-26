"""
CropGuard AI Pro - PlantVillage Classification Mapping
Maps the 38 raw class names from the PlantVillage-trained classification
model (yolov8n-cls) to clean crop/disease/severity info, since PlantVillage's
naming convention ("Corn_(maize)___Common_rust_") doesn't split cleanly with
simple string parsing the way the old detection model's classes did.
"""

# Maps: raw_class_name -> (crop_key, crop_display_name, crop_icon, disease_name, severity)
# severity is a starting default; real severity/treatment detail comes from
# disease_database.py's CROP_DISEASE_DB when a matching entry exists there.
PLANTVILLAGE_CLASS_MAP = {
    "Apple___Apple_scab": ("apple", "Apple", "🍎", "Apple Scab", "Medium"),
    "Apple___Black_rot": ("apple", "Apple", "🍎", "Black Rot", "High"),
    "Apple___Cedar_apple_rust": ("apple", "Apple", "🍎", "Cedar Apple Rust", "Medium"),
    "Apple___healthy": ("apple", "Apple", "🍎", "Healthy", "Healthy"),

    "Blueberry___healthy": ("blueberry", "Blueberry", "🫐", "Healthy", "Healthy"),

    "Cherry_(including_sour)___Powdery_mildew": ("cherry", "Cherry", "🍒", "Powdery Mildew", "Medium"),
    "Cherry_(including_sour)___healthy": ("cherry", "Cherry", "🍒", "Healthy", "Healthy"),

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("corn", "Corn", "🌽", "Gray Leaf Spot", "Medium"),
    "Corn_(maize)___Common_rust_": ("corn", "Corn", "🌽", "Common Rust", "Medium"),
    "Corn_(maize)___Northern_Leaf_Blight": ("corn", "Corn", "🌽", "Northern Leaf Blight", "High"),
    "Corn_(maize)___healthy": ("corn", "Corn", "🌽", "Healthy", "Healthy"),

    "Grape___Black_rot": ("grape", "Grape", "🍇", "Black Rot", "High"),
    "Grape___Esca_(Black_Measles)": ("grape", "Grape", "🍇", "Esca (Black Measles)", "High"),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ("grape", "Grape", "🍇", "Leaf Blight", "Medium"),
    "Grape___healthy": ("grape", "Grape", "🍇", "Healthy", "Healthy"),

    "Orange___Haunglongbing_(Citrus_greening)": ("orange", "Orange", "🍊", "Citrus Greening (HLB)", "High"),

    "Peach___Bacterial_spot": ("peach", "Peach", "🍑", "Bacterial Spot", "Medium"),
    "Peach___healthy": ("peach", "Peach", "🍑", "Healthy", "Healthy"),

    "Pepper,_bell___Bacterial_spot": ("pepper", "Bell Pepper", "🫑", "Bacterial Spot", "Medium"),
    "Pepper,_bell___healthy": ("pepper", "Bell Pepper", "🫑", "Healthy", "Healthy"),

    "Potato___Early_blight": ("potato", "Potato", "🥔", "Early Blight", "Medium"),
    "Potato___Late_blight": ("potato", "Potato", "🥔", "Late Blight", "High"),
    "Potato___healthy": ("potato", "Potato", "🥔", "Healthy", "Healthy"),

    "Raspberry___healthy": ("raspberry", "Raspberry", "🍇", "Healthy", "Healthy"),

    "Soybean___healthy": ("soybean", "Soybean", "🌱", "Healthy", "Healthy"),

    "Squash___Powdery_mildew": ("squash", "Squash", "🎃", "Powdery Mildew", "Medium"),

    "Strawberry___Leaf_scorch": ("strawberry", "Strawberry", "🍓", "Leaf Scorch", "Medium"),
    "Strawberry___healthy": ("strawberry", "Strawberry", "🍓", "Healthy", "Healthy"),

    "Tomato___Bacterial_spot": ("tomato", "Tomato", "🍅", "Bacterial Spot", "Medium"),
    "Tomato___Early_blight": ("tomato", "Tomato", "🍅", "Early Blight", "Medium"),
    "Tomato___Late_blight": ("tomato", "Tomato", "🍅", "Late Blight", "High"),
    "Tomato___Leaf_Mold": ("tomato", "Tomato", "🍅", "Leaf Mold", "Medium"),
    "Tomato___Septoria_leaf_spot": ("tomato", "Tomato", "🍅", "Septoria Leaf Spot", "Medium"),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("tomato", "Tomato", "🍅", "Spider Mites", "Medium"),
    "Tomato___Target_Spot": ("tomato", "Tomato", "🍅", "Target Spot", "Medium"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("tomato", "Tomato", "🍅", "Yellow Leaf Curl Virus", "High"),
    "Tomato___Tomato_mosaic_virus": ("tomato", "Tomato", "🍅", "Mosaic Virus", "High"),
    "Tomato___healthy": ("tomato", "Tomato", "🍅", "Healthy", "Healthy"),
}


def get_plantvillage_info(raw_class_name):
    """Look up crop/disease info for a raw PlantVillage class name.
    Returns a dict, or a safe generic fallback if the class name isn't recognized
    (e.g. if the model gets retrained later with different/extra classes)."""
    entry = PLANTVILLAGE_CLASS_MAP.get(raw_class_name)
    if entry is None:
        # Fallback: best-effort parse so the app doesn't break on unknown classes
        parts = raw_class_name.replace("___", "_").split("_")
        crop_guess = parts[0].lower() if parts else "unknown"
        return {
            "crop_key": crop_guess,
            "crop_name": parts[0].title() if parts else "Unknown",
            "crop_icon": "🌱",
            "disease_name": raw_class_name.replace("___", " ").replace("_", " "),
            "severity": "Unknown",
        }
    crop_key, crop_name, crop_icon, disease_name, severity = entry
    return {
        "crop_key": crop_key,
        "crop_name": crop_name,
        "crop_icon": crop_icon,
        "disease_name": disease_name,
        "severity": severity,
    }
