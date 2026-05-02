"""Stadium detection helpers for PyQGIS and OpenCV-based processing.

Two main entry points:
- `detect_from_qgis_layer(layer, ...)` — run inside QGIS (requires `qgis` Python modules).
- `detect_stadium_opencv(image_path, ...)` — standalone using OpenCV on an aerial image.

Output: list of detections with bounding boxes and simple scores.
"""
from typing import List, Dict, Tuple, Optional
import os

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None


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


def detect_from_qgis_layer(layer, min_area_m2: float = 1000.0) -> List[Dict]:
    """Detect stadium candidates from a QGIS vector or raster layer.

    This function is intended to be run from the QGIS Python console where `qgis` is available.
    For a raster layer it will run a simple thresholding + polygonize pipeline; for vector layers
    it filters by area and shape.
    """
    try:
        from qgis.core import QgsVectorLayer, QgsFeatureRequest, QgsGeometry
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
            if area < min_area_m2:
                continue
            bbox = geom.boundingBox()
            detections.append({
                "id": feat.id(),
                "area_m2": float(area),
                "bbox": (bbox.xMinimum(), bbox.yMinimum(), bbox.xMaximum(), bbox.yMaximum()),
            })
    else:
        # raster: suggest using external OpenCV or run a raster->vector conversion in QGIS
        raise NotImplementedError("Raster-based detection: use detect_stadium_opencv or implement raster polygonize pipeline in QGIS")

    detections.sort(key=lambda d: d.get("area_m2", d.get("area", 0)), reverse=True)
    return detections


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
