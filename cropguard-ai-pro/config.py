"""
CropGuard AI Pro - Configuration
Production-ready configuration for Flask application
"""
import os
from datetime import timedelta

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'cropguard-dev-secret-key-2024'

    # File Upload
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 32 * 1024 * 1024))  # 32MB (drone photos can be large)
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'static/uploads')
    # Wide format list - image_utils.py does the real work of decoding/
    # normalizing whatever comes in, this list is just a first-pass filter.
    ALLOWED_EXTENSIONS = set(os.environ.get(
        'ALLOWED_EXTENSIONS',
        'png,jpg,jpeg,webp,bmp,tif,tiff,gif,heic,heif,dng,cr2,nef,arw'
    ).split(','))

    # Drone connectivity
    DRONE_API_KEY = os.environ.get('DRONE_API_KEY', '')  # optional; empty = no auth required
    DRONE_DEVICE_TIMEOUT_SECONDS = int(os.environ.get('DRONE_DEVICE_TIMEOUT_SECONDS', 30))
    ENABLE_WIFI_DISCOVERY = os.environ.get('ENABLE_WIFI_DISCOVERY', 'true').lower() == 'true'

    # Model Settings
    MODEL_PATH = os.environ.get('MODEL_PATH', 'static/models/best.pt')
    MODEL_PATH_V2 = os.getenv('MODEL_PATH_V2')
    MODEL_PATH_V3 = os.getenv('MODEL_PATH_V3')
    CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', 0.25))
    IOU_THRESHOLD = float(os.environ.get('IOU_THRESHOLD', 0.45))

    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///cropguard.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Session
    PERMANENT_SESSION_LIFETIME = timedelta(days=30)

    # API Settings
    WEATHER_API_URL = os.environ.get('WEATHER_API_URL', 'https://api.open-meteo.com/v1/forecast')

    # Drone Processing
    DRONE_MAX_IMAGES = int(os.environ.get('DRONE_MAX_IMAGES', 100))
    GPS_LOG_MAX_SIZE = int(os.environ.get('GPS_LOG_MAX_SIZE', 50 * 1024 * 1024))

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

    # Security headers
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

    # Rate limiting
    RATELIMIT_STORAGE_URI = os.environ.get('REDIS_URL', 'memory://')
    RATELIMIT_STRATEGY = 'fixed-window'
    RATELIMIT_DEFAULT = "200 per hour"

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
