"""
Phase 3: Mission reports - CSV and JSON always work (stdlib only).
PDF export works if `reportlab` is installed (pip install reportlab);
otherwise the PDF route returns a clear error instead of crashing.
"""
import csv
import io
import json


def report_to_json(report_dict, mission_dict=None):
    payload = dict(report_dict)
    if mission_dict:
        payload['mission'] = mission_dict
    return json.dumps(payload, indent=2)


def report_to_csv(report_dict):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['Field', 'Value'])
    for k, v in report_dict.items():
        writer.writerow([k, v])
    return buf.getvalue()


def report_to_pdf_bytes(report_dict, mission_dict=None):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError:
        raise RuntimeError(
            "reportlab is not installed. Run: pip install reportlab"
        )

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "CropGuard AI - Drone Mission Report")
    y -= 30

    c.setFont("Helvetica", 11)
    if mission_dict:
        c.drawString(50, y, f"Mission: {mission_dict.get('name', 'Untitled')}")
        y -= 20

    for k, v in report_dict.items():
        if k == 'warnings' and isinstance(v, list) and v:
            c.drawString(50, y, f"Warnings: {', '.join(v)}")
        elif k != 'warnings':
            c.drawString(50, y, f"{k.replace('_', ' ').title()}: {v}")
        y -= 18
        if y < 50:
            c.showPage()
            y = height - 50

    c.save()
    buf.seek(0)
    return buf.read()
