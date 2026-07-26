"""
CropGuard AI Pro - WSGI entry point
Used by production servers (gunicorn, uWSGI, etc).

Run in production with:
    gunicorn --bind 0.0.0.0:8000 --workers 2 --timeout 120 wsgi:app
"""
import os
from app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
