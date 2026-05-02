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
from src.detect_stadium import detect_from_qgis_layer
# Provide a vector or raster layer selected in QGIS
result = detect_from_qgis_layer(iface.activeLayer())
print(result)
```

Usage (standalone with OpenCV):

```bash
python scripts/run_detect.py --image path/to/aerial_image.tif
```

To save a PNG overlay with detected candidates:

```bash
python scripts/run_detect.py --image data/sample.tif --output-png output/detections.png
```

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
