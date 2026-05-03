# GeoIA — Détection stade

Simple starter project for stadium detection using PyQGIS and optional OpenCV processing.

Files created:
- [README.md](README.md)
- [requirements.txt](requirements.txt)
- [src/detect_stadium.py](src/detect_stadium.py)
- [scripts/run_detect.py](scripts/run_detect.py)
- [project/detection_project.qgs](project/detection_project.qgs)

Quick setup

1. This project is intended to run inside the QGIS Python environment (QGIS Python console or `python` with PyQGIS initialized). If you want to use OpenCV-based standalone detection, install the extra Python deps below and run the runner script.

Install optional Python dependencies (outside QGIS):

```bash
pip install -r requirements.txt
```

Usage (inside QGIS Python console):

```python
import sys
sys.path.append(r"C:\Users\MaZik\GeoIA")

from src.detect_stadium import run_stadium_detection

# Select a raster or vector layer in QGIS, then run this.
result = run_stadium_detection(iface.activeLayer(), min_area=5000)
print(result)
```

QGIS 3.44.8 test workflow

1. Open QGIS 3.44.8.
2. Load a GeoTIFF or vector file with a stadium-like area.
3. Click the target layer in the Layers panel so it becomes the active layer.
4. Open `Plugins -> Python Console`.
5. Paste the snippet above.
6. For raster layers, a new polygon layer named like `YourLayer stadium detections` is added to the project.
7. Zoom to the new layer and inspect the bounding boxes.

If you want to test the PNG overlay path outside QGIS first:

```bash
python scripts/run_detect.py --image data/sample.tif --min-area 100 --output-png output/detections.png
```

Usage (standalone with OpenCV):

```bash
python scripts/run_detect.py --image path/to/aerial_image.tif
```

To save a PNG overlay with detected candidates:

```bash
python scripts/run_detect.py --image data/sample.tif --output-png output/detections.png
```

For the Madagascar test, the output is now a map-style PNG with longitude/latitude axes and detection boxes:

```bash
python scripts/test_madagascar.py --output-dir data/madagascar_test
```

The map export is written to [data/madagascar_test/madagascar_map.png](data/madagascar_test/madagascar_map.png).

Next steps

- Replace the sample data paths in [project/detection_project.qgs](project/detection_project.qgs) with your local layers.
- Tune [src/detect_stadium.py](src/detect_stadium.py) parameters for your imagery (resolution, thresholds).

Downloading sample data from OpenGeoAI

You can download NAIP aerial imagery and building footprints using the OpenGeoAI utilities (requires `geoai-py` and `leafmap`). A helper script is provided at [scripts/download_opengeoai_data.py](scripts/download_opengeoai_data.py).

Example:

```bash
pip install geoai-py leafmap
python scripts/download_opengeoai_data.py --out data --max-items 1
```

Pass `--bbox MINX MINY MAXX MAXY` to specify your ROI in lon/lat.
