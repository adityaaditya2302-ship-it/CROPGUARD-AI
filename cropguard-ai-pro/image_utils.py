"""
CropGuard AI Pro - Universal Image Normalization
=================================================
Drones send images in wildly different formats depending on brand/camera:
JPEG, PNG, WEBP, BMP, TIFF (multi-page), GIF, HEIC/HEIF (iPhone-style
companion apps), 16-bit thermal/multispectral captures, CMYK exports,
paletted PNGs, images with rotated EXIF orientation, etc.

This module gives the rest of the app ONE guarantee: no matter what bytes
come in, `normalize_image_bytes()` either returns a clean, upright, RGB
JPEG file ready for YOLO, or raises a clear ImageDecodeError explaining
why it couldn't (e.g. corrupted transfer, empty file).

Supported inputs out of the box:
  - Anything Pillow can decode: JPEG, PNG, BMP, TIFF, GIF, PPM, ICO, WEBP...
  - HEIC/HEIF (iPhone photos) - via pillow-heif, registered below
  - 16-bit / high-bit-depth captures - auto-normalized to 8-bit
  - CMYK, palette (P), grayscale (L), RGBA - all converted to clean RGB
  - Multi-frame TIFF/GIF - first frame used
  - Images with EXIF rotation tags - auto-rotated upright
  - Raw camera formats (.dng, .cr2, .nef, .arw, ...) - via optional
    `rawpy`, if installed (best-effort; not required)
"""
import io
import os
import uuid

from PIL import Image, ImageOps, UnidentifiedImageError

# Register HEIC/HEIF support with Pillow if the plugin is available.
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_SUPPORTED = True
except ImportError:
    HEIF_SUPPORTED = False

# Optional: RAW sensor formats (DNG/CR2/NEF/ARW/...) from professional
# drone camera payloads. Not required - degrades gracefully if absent.
try:
    import rawpy
    RAW_SUPPORTED = True
except ImportError:
    RAW_SUPPORTED = False

MAX_DIMENSION = 4096  # downscale absurdly large drone captures for speed
JPEG_QUALITY = 92


class ImageDecodeError(Exception):
    """Raised when the given bytes cannot be turned into a usable image."""
    pass


def _open_with_pillow(raw_bytes):
    return Image.open(io.BytesIO(raw_bytes))


def _open_with_rawpy(raw_bytes):
    with rawpy.imread(io.BytesIO(raw_bytes)) as raw:
        rgb = raw.postprocess()
    return Image.fromarray(rgb)


def _to_clean_rgb(img):
    """Flatten any color mode down to plain RGB, upright, 8-bit."""
    # Auto-rotate based on EXIF orientation tag (very common with phone-
    # style drone companion apps that don't pre-rotate images).
    img = ImageOps.exif_transpose(img)

    # Multi-frame formats (animated GIF, multi-page TIFF): use first frame.
    if getattr(img, 'is_animated', False):
        img.seek(0)

    if img.mode in ('RGBA', 'LA'):
        # Flatten transparency onto a white background instead of losing it.
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode == 'P':
        img = img.convert('RGBA')
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode == 'CMYK':
        img = img.convert('RGB')
    elif img.mode in ('L', 'I', 'I;16', 'I;16B', 'F'):
        # Grayscale / high-bit-depth / thermal-style single channel captures.
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    # Downscale extremely large sensor images (common on professional
    # drone payloads / multispectral cameras) to keep inference fast.
    if max(img.size) > MAX_DIMENSION:
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)

    return img


def normalize_image_bytes(raw_bytes, output_dir, filename_hint='drone_image'):
    """
    Convert arbitrary image bytes into a normalized JPEG on disk.

    Returns (filepath, unique_filename).
    Raises ImageDecodeError if the bytes are not a decodable image.
    """
    if not raw_bytes or len(raw_bytes) < 16:
        raise ImageDecodeError('Empty or truncated image data received.')

    img = None
    errors = []

    # 1. Try normal Pillow decode (covers JPEG/PNG/WEBP/BMP/TIFF/GIF/HEIC...)
    try:
        img = _open_with_pillow(raw_bytes)
        img.load()
    except (UnidentifiedImageError, OSError, ValueError) as e:
        errors.append(str(e))
        img = None

    # 2. Fall back to RAW sensor decode if available and Pillow failed.
    if img is None and RAW_SUPPORTED:
        try:
            img = _open_with_rawpy(raw_bytes)
        except Exception as e:
            errors.append(str(e))
            img = None

    if img is None:
        hint = '' if HEIF_SUPPORTED else ' (install pillow-heif for HEIC/HEIF support)'
        raise ImageDecodeError(
            f'Could not decode image{hint}: {"; ".join(errors) or "unknown format"}'
        )

    img = _to_clean_rgb(img)

    os.makedirs(output_dir, exist_ok=True)
    safe_hint = ''.join(c for c in filename_hint if c.isalnum() or c in ('-', '_'))[:40] or 'image'
    unique_name = f"{uuid.uuid4().hex}_{safe_hint}.jpg"
    filepath = os.path.join(output_dir, unique_name)

    img.save(filepath, 'JPEG', quality=JPEG_QUALITY)
    return filepath, unique_name


def normalize_uploaded_file(file_storage, output_dir):
    """Convenience wrapper for a Flask FileStorage object (from request.files)."""
    raw_bytes = file_storage.read()
    hint = os.path.splitext(file_storage.filename or 'image')[0]
    return normalize_image_bytes(raw_bytes, output_dir, filename_hint=hint)
