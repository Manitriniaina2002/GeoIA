"""
Stadium Detection Script for QGIS
===================================
Uses Claude Vision API to detect stadiums in a GeoTIFF raster.
Run from QGIS Python Console:
    exec(open(r'C:/Users/MaZik/GeoIA/scripts/detect_stadiums.py').read())
"""

import os, sys, json, base64, math, urllib.request
from pathlib import Path
from qgis.core import QgsProject, QgsVectorLayer, QgsMapSettings, QgsRectangle, QgsMapRendererCustomPainterJob
from PyQt5.QtGui import QImage, QPainter
from PyQt5.QtCore import QSize

# ── CONFIG ────────────────────────────────────────────────────
ANTHROPIC_API_KEY = "sk-ant-XXXXXXXXXXXXXXXXXXXXXXXX"  # ← YOUR KEY HERE
RASTER_LAYER_NAME = "Madagascar Raster"
OUTPUT_GEOJSON    = r"C:\Users\MaZik\GeoIA\output\detected_stadiums.geojson"
TILE_SIZE_PX      = 1024
TILE_OVERLAP_PX   = 64
CONFIDENCE_FILTER = "medium"
# ─────────────────────────────────────────────────────────────

def render_tile(layer_name, x0, y0, x1, y1, out_path, px=TILE_SIZE_PX):
    layers = QgsProject.instance().mapLayersByName(layer_name)
    settings = QgsMapSettings()
    settings.setLayers(layers)
    settings.setOutputSize(QSize(px, px))
    settings.setExtent(QgsRectangle(x0, y0, x1, y1))
    settings.setDestinationCrs(layers[0].crs())
    img = QImage(QSize(px, px), QImage.Format_ARGB32_Premultiplied)
    img.fill(0)
    painter = QPainter(img)
    job = QgsMapRendererCustomPainterJob(settings, painter)
    job.start(); job.waitForFinished(); painter.end()
    img.save(out_path)

def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def call_claude_vision(b64_image, tile_info=""):
    prompt = (
        f"This is a satellite image tile {tile_info}. "
        "Find any stadiums, athletics tracks, or large sports arenas. "
        "Return ONLY a JSON array, no markdown:\n"
        '[{"name":"...","cx":<px>,"cy":<px>,"confidence":"high|medium|low","type":"stadium|athletics_track|football_pitch"}]\n'
        "Image is 1024x1024 pixels, origin top-left. If none found return: []"
    )
    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_image}},
            {"type": "text", "text": prompt}
        ]}]
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=payload,
        headers={"Content-Type": "application/json",
                 "anthropic-version": "2023-06-01",
                 "x-api-key": ANTHROPIC_API_KEY}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    raw = "".join(b["text"] for b in result["content"] if b["type"] == "text").strip()
    return json.loads(raw.replace("```json","").replace("```","").strip())

def px_to_geo(px, py, t_xmin, t_ymax, geo_w, geo_h, tile_px=TILE_SIZE_PX):
    return t_xmin + (px/tile_px)*geo_w, t_ymax - (py/tile_px)*geo_h

def deduplicate(dets, threshold=0.001):
    rank = {"high":3,"medium":2,"low":1}
    kept = []
    for d in sorted(dets, key=lambda x: -rank.get(x["confidence"],0)):
        if not any(abs(k["lon"]-d["lon"])<threshold and abs(k["lat"]-d["lat"])<threshold for k in kept):
            kept.append(d)
    return kept

# ── MAIN ──────────────────────────────────────────────────────
print("="*50)
print("  Stadium Detection — Antananarivo, Madagascar")
print("="*50)

layers = QgsProject.instance().mapLayersByName(RASTER_LAYER_NAME)
if not layers: raise ValueError(f"Layer not found: {RASTER_LAYER_NAME}")
lyr = layers[0]
ext = lyr.extent()
xmin,ymin,xmax,ymax = ext.xMinimum(),ext.yMinimum(),ext.xMaximum(),ext.yMaximum()
rw, rh = lyr.width(), lyr.height()
print(f"Raster: {rw}x{rh}px | {lyr.crs().authid()}")

geo_w = xmax-xmin; geo_h = ymax-ymin
tile_deg_x = TILE_SIZE_PX/(rw/geo_w)
tile_deg_y = TILE_SIZE_PX/(rh/geo_h)
step_x = tile_deg_x - TILE_OVERLAP_PX/(rw/geo_w)
step_y = tile_deg_y - TILE_OVERLAP_PX/(rh/geo_h)
cols = math.ceil(geo_w/step_x); rows = math.ceil(geo_h/step_y)
total = cols*rows
print(f"Tiling: {cols}x{rows} = {total} tiles")

tmp = Path(OUTPUT_GEOJSON).parent/"_tiles_tmp"
tmp.mkdir(parents=True, exist_ok=True)

all_dets = []; idx = 0
rank = {"high":3,"medium":2,"low":1}

for row in range(rows):
    for col in range(cols):
        idx += 1
        tx0 = xmin+col*step_x; tx1 = min(tx0+tile_deg_x,xmax)
        ty1 = ymax-row*step_y;  ty0 = max(ty1-tile_deg_y,ymin)
        label = f"tile_{row:02d}_{col:02d}"
        path  = str(tmp/f"{label}.png")
        print(f"\n[{idx}/{total}] {label}")
        try:
            render_tile(RASTER_LAYER_NAME, tx0, ty0, tx1, ty1, path)
            dets = call_claude_vision(image_to_base64(path), f"(tile {idx}/{total})")
            print(f"  → {len(dets)} detection(s)")
            for d in dets:
                if rank.get(d.get("confidence","low"),0) >= rank.get(CONFIDENCE_FILTER,0):
                    lon,lat = px_to_geo(d["cx"],d["cy"],tx0,ty1,tx1-tx0,ty1-ty0)
                    d.update({"lon":lon,"lat":lat,"tile":label})
                    all_dets.append(d)
                    print(f"     + {d['name']} ({d['confidence']}) @ ({lat:.5f},{lon:.5f})")
        except Exception as e:
            print(f"  ✗ {e}")

all_dets = deduplicate(all_dets)
features = [{"type":"Feature",
             "geometry":{"type":"Point","coordinates":[d["lon"],d["lat"]]},
             "properties":{"id":i+1,"name":d.get("name","Stadium"),
                           "type":d.get("type","stadium"),"confidence":d.get("confidence","?"),
                           "tile":d.get("tile","")}}
            for i,d in enumerate(all_dets)]
geojson = {"type":"FeatureCollection",
           "crs":{"type":"name","properties":{"name":"EPSG:4326"}},
           "features":features}
Path(OUTPUT_GEOJSON).parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_GEOJSON,"w",encoding="utf-8") as f:
    json.dump(geojson,f,indent=2,ensure_ascii=False)
print(f"\n✓ GeoJSON: {OUTPUT_GEOJSON} ({len(all_dets)} stadium(s))")

vl = QgsVectorLayer(OUTPUT_GEOJSON,"Stadium Detections","ogr")
if vl.isValid():
    for old in QgsProject.instance().mapLayersByName("Stadium Detections"):
        QgsProject.instance().removeMapLayer(old.id())
    QgsProject.instance().addMapLayer(vl)
    print("✓ Layer added to QGIS")

print("="*50)
print(f"  DONE — {len(all_dets)} stadium(s) detected")
print("="*50)
