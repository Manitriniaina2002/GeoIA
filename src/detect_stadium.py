"""Stadium detection helpers for PyQGIS and OpenCV-based processing.

Two main entry points:
- `detect_from_qgis_layer(layer, ...)` — run inside QGIS (requires `qgis` Python modules).
- `detect_stadium_opencv(image_path, ...)` — standalone using OpenCV on an aerial image.

Output: list of detections with bounding boxes and simple scores.
"""
from typing import List, Dict, Tuple, Optional
import os
import re
import math
from io import BytesIO
from urllib.request import Request, urlopen

try:
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import rasterio
    from PIL import Image
except Exception:
    cv2 = None
    np = None
    plt = None
    Rectangle = None
    rasterio = None
    Image = None


def detect_stadium_opencv(image_path: str, min_area: int = 5000) -> List[Dict]:
    """Detect candidate stadium-like shapes in an aerial image using OpenCV.

    This is a heuristic starter implementation: it finds large contours and fits ellipses/rects.
    """
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and numpy are required for this function. Install from requirements.txt")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur + adaptive threshold to highlight large structures
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphology to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        detection = {
            "bbox": (int(x), int(y), int(w), int(h)),
            "area": float(area),
        }
        # try ellipse fit for oval stadiums
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            detection["ellipse"] = tuple(map(float, ellipse[0])) + tuple(map(float, ellipse[1])) + (float(ellipse[2]),)
        detections.append(detection)
    # sort by area descending
    detections.sort(key=lambda d: d["area"], reverse=True)
    return detections


def _normalize_raster_source(source: str) -> str:
    """Return a local file path when the QGIS raster source contains URI fragments."""
    if not source:
        return source
    return re.split(r"\|", source, maxsplit=1)[0]


def _bbox_to_polygon(extent, width: int, height: int, bbox: Tuple[int, int, int, int]):
    """Convert a pixel-space bbox to a QgsGeometry polygon using raster extent."""
    from qgis.core import QgsGeometry, QgsPointXY

    x, y, w, h = bbox
    # Default pixel->map approximation using extent and pixel dimensions
    min_x = extent.xMinimum() + (x / width) * extent.width()
    max_x = extent.xMinimum() + ((x + w) / width) * extent.width()
    max_y = extent.yMaximum() - (y / height) * extent.height()
    min_y = extent.yMaximum() - ((y + h) / height) * extent.height()

    ring = [
        QgsPointXY(min_x, min_y),
        QgsPointXY(max_x, min_y),
        QgsPointXY(max_x, max_y),
        QgsPointXY(min_x, max_y),
        QgsPointXY(min_x, min_y),
    ]
    return QgsGeometry.fromPolygonXY([ring])


def _bbox_to_polygon_with_transform(transform, bbox: Tuple[int, int, int, int]):
    """Convert a pixel-space bbox to a QgsGeometry polygon using a rasterio affine transform.

    `transform` is the Affine transform from rasterio (src.transform).
    bbox is (x, y, w, h) in pixel coordinates where (0,0) is top-left.
    """
    from qgis.core import QgsGeometry, QgsPointXY

    x, y, w, h = bbox
    # rasterio affine: transform * (col, row) -> (x, y) in map coords
    try:
        lon1, lat1 = transform * (x, y + h)
        lon2, lat2 = transform * (x + w, y)

        ring = [
            QgsPointXY(lon1, lat1),
            QgsPointXY(lon2, lat1),
            QgsPointXY(lon2, lat2),
            QgsPointXY(lon1, lat2),
            QgsPointXY(lon1, lat1),
        ]
        return QgsGeometry.fromPolygonXY([ring])
    except Exception:
        # Fallback to None to let caller use alternate method
        return None


def _create_detection_layer(name: str, crs_authid: str):
    """Create an in-memory polygon layer for detection results."""
    from qgis.core import QgsVectorLayer, QgsField, QgsProject
    from PyQt5.QtCore import QVariant

    layer = QgsVectorLayer(f"Polygon?crs={crs_authid}", name, "memory")
    provider = layer.dataProvider()
    provider.addAttributes([
        QgsField("source", QVariant.String),
        QgsField("area_px", QVariant.Double),
        QgsField("bbox_x", QVariant.Int),
        QgsField("bbox_y", QVariant.Int),
        QgsField("bbox_w", QVariant.Int),
        QgsField("bbox_h", QVariant.Int),
    ])
    layer.updateFields()
    QgsProject.instance().addMapLayer(layer)
    return layer


def _append_detection_features(layer, detections: List[Dict], extent, raster_width: int, raster_height: int, source_label: str):
    from qgis.core import QgsFeature

    provider = layer.dataProvider()
    features = []
    for detection in detections:
        # Attempt to use rasterio transform when possible for precise georeferencing
        geometry = None
        try:
            source = getattr(layer, 'dataProvider')().dataSourceUri()
            # dataSourceUri may include a |; normalize
            source_path = re.split(r"\|", source, maxsplit=1)[0]
            try:
                import rasterio
                with rasterio.open(source_path) as src:
                    transform = src.transform
                    geometry = _bbox_to_polygon_with_transform(transform, detection["bbox"])
            except Exception:
                geometry = None
        except Exception:
            geometry = None

        if geometry is None:
            geometry = _bbox_to_polygon(extent, raster_width, raster_height, detection["bbox"])
        x, y, w, h = detection["bbox"]
        feature = QgsFeature(layer.fields())
        feature.setGeometry(geometry)
        feature["source"] = source_label
        feature["area_px"] = float(detection["area"])
        feature["bbox_x"] = int(x)
        feature["bbox_y"] = int(y)
        feature["bbox_w"] = int(w)
        feature["bbox_h"] = int(h)
        features.append(feature)

    provider.addFeatures(features)
    layer.updateExtents()
    return layer


def draw_detections(image_path: str, detections: List[Dict], output_path: str) -> str:
    """Draw detection boxes and ellipses on the input image and save as PNG."""
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and numpy are required for this function. Install from requirements.txt")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    overlay = img.copy()
    for detection in detections:
        x, y, w, h = detection["bbox"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"area={int(detection['area'])}"
        cv2.putText(
            overlay,
            label,
            (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        ellipse = detection.get("ellipse")
        if ellipse:
            center_x, center_y, axis_x, axis_y, angle = ellipse
            cv2.ellipse(
                overlay,
                (int(center_x), int(center_y)),
                (max(1, int(axis_x / 2)), max(1, int(axis_y / 2))),
                float(angle),
                0,
                360,
                (0, 0, 255),
                2,
            )

    # PNG output preserves the overlay for quick sharing or inspection.
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if not cv2.imwrite(output_path, overlay):
        raise RuntimeError(f"Failed to write overlay PNG: {output_path}")
    return output_path


def draw_georeferenced_map(raster_path: str, detections: List[Dict], output_path: str, title: str = "Madagascar Stadium Detection", raster_alpha: float = 0.12) -> str:
    """Render a georeferenced map-style PNG with lon/lat axes and detection boxes."""
    if plt is None or Rectangle is None or rasterio is None:
        raise RuntimeError("matplotlib and rasterio are required for map export")

    with rasterio.open(raster_path) as src:
        rgb = src.read([1, 2, 3])
        bounds = src.bounds
        transform = src.transform

    # convert RGB from band-first to image-first
    image = np.moveaxis(rgb, 0, -1)

    fig, ax = plt.subplots(figsize=(11, 8), dpi=180)

    basemap = _fetch_osm_basemap(bounds.left, bounds.bottom, bounds.right, bounds.top)
    if basemap is not None:
        ax.imshow(basemap[0], extent=basemap[1], origin="upper")

    # Draw the raster lightly so the OpenStreetMap basemap stays readable.
    ax.imshow(image, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top), origin="upper", alpha=raster_alpha)

    for detection in detections:
        x, y, w, h = detection["bbox"]
        # pixel corners -> geographic bounds using rasterio-like affine math
        lon1, lat1 = transform * (x, y + h)
        lon2, lat2 = transform * (x + w, y)
        rect = Rectangle((lon1, lat1), lon2 - lon1, lat2 - lat1, linewidth=2, edgecolor="#00ff66", facecolor="none")
        ax.add_patch(rect)

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, color="white", linewidth=0.4, alpha=0.18)
    ax.set_xlim(bounds.left, bounds.right)
    ax.set_ylim(bounds.bottom, bounds.top)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _deg2num(lon_deg: float, lat_deg: float, zoom: int) -> Tuple[int, int]:
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return xtile, ytile


def _num2deg(xtile: int, ytile: int, zoom: int) -> Tuple[float, float]:
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lon_deg, lat_deg


def _fetch_osm_basemap(min_lon: float, min_lat: float, max_lon: float, max_lat: float, zoom: int = 15):
    """Fetch and mosaic OpenStreetMap tiles for the requested geographic extent."""
    if Image is None:
        return None

    x_min, y_max = _deg2num(min_lon, min_lat, zoom)
    x_max, y_min = _deg2num(max_lon, max_lat, zoom)
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    tiles = []
    for y in range(y_min, y_max + 1):
        row = []
        for x in range(x_min, x_max + 1):
            url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
            try:
                request = Request(url, headers={"User-Agent": "GeoIA/1.0"})
                with urlopen(request, timeout=30) as response:
                    data = response.read()
                tile = Image.open(BytesIO(data)).convert("RGB")
            except Exception:
                return None
            row.append(np.asarray(tile))
        tiles.append(np.concatenate(row, axis=1))

    mosaic = np.concatenate(tiles, axis=0)
    west, north = _num2deg(x_min, y_min, zoom)
    east, south = _num2deg(x_max + 1, y_max + 1, zoom)
    return mosaic, (west, east, south, north)


def detect_from_qgis_layer(layer, min_area: float = 1000.0, create_output_layer: bool = True) -> List[Dict]:
    """Detect stadium candidates from a QGIS vector or raster layer.

    This function is intended to be run from the QGIS Python console where `qgis` is available.
    For a raster layer it will run a simple thresholding + polygonize pipeline; for vector layers
    it filters by area and shape.
    """
    try:
        from qgis.core import QgsVectorLayer, QgsFeatureRequest
    except Exception as e:
        raise RuntimeError("This function must be run in a QGIS Python environment with qgis modules available") from e

    detections = []
    if layer.type() == layer.VectorLayer:
        # vector: iterate features and check area/shape
        for feat in layer.getFeatures(QgsFeatureRequest()):
            geom = feat.geometry()
            if not geom:
                continue
            area = geom.area()
            if area < min_area:
                continue
            bbox = geom.boundingBox()
            detections.append({
                "id": feat.id(),
                "area_m2": float(area),
                "bbox": (bbox.xMinimum(), bbox.yMinimum(), bbox.xMaximum(), bbox.yMaximum()),
            })
    else:
        raster_path = _normalize_raster_source(layer.source())
        detections = detect_stadium_opencv(raster_path, min_area=int(min_area))
        if create_output_layer:
            # attach georeferenced polygons when the layer is a raster
            try:
                crs_authid = layer.crs().authid() or "EPSG:4326"
                output_layer = _create_detection_layer(f"{layer.name()} stadium detections", crs_authid)
                _append_detection_features(
                    output_layer,
                    detections,
                    layer.extent(),
                    layer.width(),
                    layer.height(),
                    layer.name(),
                )
            except Exception:
                # detection results are still returned even if a map layer cannot be created
                pass

    detections.sort(key=lambda d: d.get("area_m2", d.get("area", 0)), reverse=True)
    return detections


def run_stadium_detection(layer=None, min_area_pixels: int = 5000, create_output_layer: bool = True) -> List[Dict]:
    """Convenience entry point for QGIS 3.44.8 Python console.

    If `layer` is omitted, uses `iface.activeLayer()`.
    For raster layers this will also add a polygon output layer to the project.
    """
    try:
        from qgis.core import QgsProject
    except Exception as e:
        raise RuntimeError("Run this from the QGIS Python console.") from e

    if layer is None:
        from qgis.utils import iface

        layer = iface.activeLayer()
    if layer is None:
        raise ValueError("No active layer found.")

    results = detect_from_qgis_layer(layer, min_area=float(min_area_pixels), create_output_layer=create_output_layer)
    return results


if __name__ == "__main__":
    # simple CLI for OpenCV mode
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Detect stadiums in aerial image (heuristic)")
    parser.add_argument("--image", required=True, help="Path to aerial image")
    parser.add_argument("--min-area", type=int, default=5000, help="Minimum contour area in pixels")
    parser.add_argument("--output-png", help="Optional path to save a PNG overlay with detections")
    args = parser.parse_args()
    det = detect_stadium_opencv(args.image, min_area=args.min_area)

    if args.output_png:
        output_path = draw_detections(args.image, det, args.output_png)
        print(f"Saved overlay PNG to: {output_path}")

    print(json.dumps(det, indent=2))
