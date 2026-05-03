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
import pickle
from pathlib import Path

try:
    import pandas as pd
    from skimage import feature as _sk_feature
except Exception:
    pd = None
    _sk_feature = None

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


def _read_image_array(image_path: str):
    """Read an image into a BGR array, with TIFF-friendly fallbacks."""
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and numpy are required for this function. Install from requirements.txt")

    suffix = Path(image_path).suffix.lower()
    prefer_rasterio = suffix in {".tif", ".tiff"}

    if prefer_rasterio and rasterio is not None:
        try:
            with rasterio.open(image_path) as src:
                data = src.read()
            if data.ndim == 2:
                data = np.stack([data, data, data], axis=-1)
            else:
                data = np.transpose(data[:3], (1, 2, 0)) if data.shape[0] >= 3 else np.transpose(data, (1, 2, 0))
                if data.shape[2] == 1:
                    data = np.repeat(data, 3, axis=2)
                elif data.shape[2] > 3:
                    data = data[:, :, :3]

            if data.dtype != np.uint8:
                data = data.astype(np.float32)
                finite = data[np.isfinite(data)] if np.isfinite(data).any() else None
                if finite is not None and finite.size:
                    lo = float(np.percentile(finite, 2))
                    hi = float(np.percentile(finite, 98))
                else:
                    lo = float(np.min(data))
                    hi = float(np.max(data))
                if hi > lo:
                    data = (np.clip(data, lo, hi) - lo) * (255.0 / (hi - lo))
                else:
                    data = np.clip(data, 0, 255)
                data = data.astype(np.uint8)

            return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        except Exception:
            pass

    if Image is not None:
        try:
            with Image.open(image_path) as pil_img:
                data = np.array(pil_img)
            if data.ndim == 2:
                data = np.stack([data, data, data], axis=-1)
            elif data.shape[2] > 3:
                data = data[:, :, :3]
            if data.dtype != np.uint8:
                data = data.astype(np.float32)
                finite = data[np.isfinite(data)] if np.isfinite(data).any() else None
                if finite is not None and finite.size:
                    lo = float(np.percentile(finite, 2))
                    hi = float(np.percentile(finite, 98))
                else:
                    lo = float(np.min(data))
                    hi = float(np.max(data))
                if hi > lo:
                    data = (np.clip(data, lo, hi) - lo) * (255.0 / (hi - lo))
                else:
                    data = np.clip(data, 0, 255)
                data = data.astype(np.uint8)
            return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        except Exception:
            pass

    img = cv2.imread(image_path)
    if img is not None:
        return img

    if not prefer_rasterio and rasterio is not None:
        try:
            with rasterio.open(image_path) as src:
                data = src.read()
            if data.ndim == 2:
                data = np.stack([data, data, data], axis=-1)
            else:
                data = np.transpose(data[:3], (1, 2, 0)) if data.shape[0] >= 3 else np.transpose(data, (1, 2, 0))
                if data.shape[2] == 1:
                    data = np.repeat(data, 3, axis=2)
                elif data.shape[2] > 3:
                    data = data[:, :, :3]

            if data.dtype != np.uint8:
                data = data.astype(np.float32)
                finite = data[np.isfinite(data)] if np.isfinite(data).any() else None
                if finite is not None and finite.size:
                    lo = float(np.percentile(finite, 2))
                    hi = float(np.percentile(finite, 98))
                else:
                    lo = float(np.min(data))
                    hi = float(np.max(data))
                if hi > lo:
                    data = (np.clip(data, lo, hi) - lo) * (255.0 / (hi - lo))
                else:
                    data = np.clip(data, 0, 255)
                data = data.astype(np.uint8)

            return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        except Exception:
            pass

    return None


def detect_stadium_opencv(image_path: str, min_area: int = 5000, model_path: Optional[str] = None, scaler_path: Optional[str] = None, model_object: Optional[object] = None) -> List[Dict]:
    """Detect candidate stadium-like shapes in an aerial image using OpenCV.

    This is a heuristic starter implementation: it finds large contours and fits ellipses/rects.
    """
    from typing import Optional
    img = _read_image_array(image_path)
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

    # If a trained model is provided (path or object), run inference per-detection
    model = None
    scaler = None
    try:
        if model_object is not None:
            model = model_object
        elif model_path:
            model = load_model(model_path)
        if scaler_path:
            try:
                with open(scaler_path, 'rb') as sf:
                    scaler = pickle.load(sf)
            except Exception:
                scaler = None
    except Exception:
        model = None
        scaler = None

    if model is not None:
        for det in detections:
            try:
                features = _extract_features_from_crop(img, det['bbox'])
                label, score = predict_with_model(model, scaler, features)
                det['model_label'] = label
                det['model_score'] = score
            except Exception:
                det['model_label'] = None
                det['model_score'] = None

    return detections


def load_model(path: str):
    """Load a pickled sklearn model (or similar) from `path`."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(p, 'rb') as f:
        model = pickle.load(f)
    return model


def _extract_features_from_crop(img: 'np.ndarray', bbox: Tuple[int, int, int, int]) -> Dict[str, float]:
    """Compute a small set of features from an image crop suitable for model inference.

    Returns flat numeric features used by the training pipeline (means, CLAHE stats, LBP proxies).
    """
    x, y, w, h = bbox
    h_img, w_img = img.shape[:2]
    # clamp bbox
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w_img, x + w)
    y1 = min(h_img, y + h)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return {}

    features = {}
    # Per-channel means and stds
    chans = cv2.split(crop)
    for i, ch in enumerate(chans):
        features[f'chan{i}_mean'] = float(np.mean(ch))
        features[f'chan{i}_std'] = float(np.std(ch))
        # histogram mean (rough compact descriptor)
        hist, _ = np.histogram(ch.ravel(), bins=64, range=(0, 255))
        features[f'hist_c{i}_mean'] = float(np.mean(hist))

    # CLAHE (on grayscale)
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(gray)
        features['clahe_mean'] = float(np.mean(cl))
        features['clahe_std'] = float(np.std(cl))
    except Exception:
        features['clahe_mean'] = 0.0
        features['clahe_std'] = 0.0

    # LBP fallback: use skimage if available, otherwise edge density
    try:
        if _sk_feature is not None:
            grayf = (gray / 255.0).astype('float32')
            lbp = _sk_feature.local_binary_pattern((grayf * 255).astype(np.uint8), P=8, R=1, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
            features['lbp_mean'] = float(np.mean(lbp_hist))
            features['lbp_std'] = float(np.std(lbp_hist))
        else:
            edges = cv2.Canny(gray, 50, 150)
            features['lbp_mean'] = float(np.mean(edges))
            features['lbp_std'] = float(np.std(edges))
    except Exception:
        features['lbp_mean'] = 0.0
        features['lbp_std'] = 0.0

    return features


def predict_with_model(model, scaler, features: Dict[str, float]):
    """Run model inference on a single flattened `features` dict.

    - `model` is a scikit-learn style model (pickled) or None.
    - `scaler` is an optional scikit-learn scaler (fit on training features).

    Returns (label, score) where `score` is model-dependent (probability or anomaly score).
    """
    if model is None:
        return None, None
    if pd is None:
        # Not enough dependencies to prepare a DataFrame
        return None, None

    df = pd.DataFrame([features])
    # Align columns if model exposes feature names
    try:
        if hasattr(model, 'feature_names_in_'):
            cols = list(model.feature_names_in_)
            # add missing cols with zeros
            for c in cols:
                if c not in df.columns:
                    df[c] = 0.0
            df = df[cols]
    except Exception:
        pass

    X = df.values
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass

    # Classification with predict_proba
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            # choose positive class probability if available
            score = float(proba[0].max())
            label = int(model.predict(X)[0])
            return label, score
        elif hasattr(model, 'decision_function'):
            score = float(model.decision_function(X)[0])
            label = int(model.predict(X)[0]) if hasattr(model, 'predict') else None
            return label, score
        else:
            pred = model.predict(X)
            return int(pred[0]), None
    except Exception:
        return None, None


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
        QgsField("label", QVariant.String),
        QgsField("score", QVariant.Double),
        QgsField("model_label", QVariant.String),
        QgsField("model_score", QVariant.Double),
    ])
    layer.updateFields()
    QgsProject.instance().addMapLayer(layer)
    return layer


def _append_detection_features(layer, detections: List[Dict], extent, raster_width: int, raster_height: int, source_label: str, raster_path: Optional[str] = None):
    from qgis.core import QgsFeature

    provider = layer.dataProvider()
    features = []
    for detection in detections:
        # Attempt to use rasterio transform when possible for precise georeferencing
        geometry = None
        try:
            source_path = raster_path
            if not source_path:
                source = getattr(layer, 'dataProvider')().dataSourceUri()
                source_path = re.split(r"\|", source, maxsplit=1)[0]
            try:
                import rasterio
                if source_path:
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
        if 'label' in detection:
            feature['label'] = str(detection.get('label')) if detection.get('label') is not None else None
        if 'score' in detection:
            try:
                feature['score'] = float(detection.get('score')) if detection.get('score') is not None else None
            except Exception:
                feature['score'] = None
        # model fields (optional)
        if 'model_label' in detection:
            try:
                feature['model_label'] = str(detection.get('model_label')) if detection.get('model_label') is not None else None
            except Exception:
                feature['model_label'] = None
        if 'model_score' in detection:
            try:
                feature['model_score'] = float(detection.get('model_score')) if detection.get('model_score') is not None else None
            except Exception:
                feature['model_score'] = None
        features.append(feature)

    provider.addFeatures(features)
    layer.updateExtents()
    return layer


def draw_detections(image_path: str, detections: List[Dict], output_path: str) -> str:
    """Draw detection boxes and ellipses on the input image and save as PNG."""
    img = _read_image_array(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    overlay = img.copy()
    for detection in detections:
        x, y, w, h = detection["bbox"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        score = detection.get("score", detection.get("model_score"))
        text_bits = []
        if detection.get("label"):
            text_bits.append(str(detection["label"]))
        if score is not None:
            try:
                text_bits.append(f"{float(score):.2f}")
            except Exception:
                pass
        if not text_bits:
            text_bits.append(f"area={int(detection['area'])}")
        label = " | ".join(text_bits)
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


def detect_stadium_grounding_dino(
    image_path: str,
    min_area: int = 5000,
    text_prompt: str = "stadium. sports field. arena. athletics track. soccer stadium. football stadium.",
    tile_size: int = 800,
    tile_overlap: int = 100,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    model_id: str = "IDEA-Research/grounding-dino-tiny",
    device: Optional[str] = None,
) -> List[Dict]:
    """Run Grounding DINO zero-shot detection on a raster image."""
    from src.grounding_dino_zero_shot import detect_stadium_grounding_dino as _impl

    return _impl(
        raster_path=image_path,
        min_area=min_area,
        text_prompt=text_prompt,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        model_id=model_id,
        device=device,
    )


def draw_georeferenced_map(raster_path: str, detections: List[Dict], output_path: str, title: str = None, region: str = "World", raster_alpha: float = 0.12) -> str:
    """Render a georeferenced map-style PNG with lon/lat axes and detection boxes.
    
    Args:
        raster_path: Path to raster image
        detections: List of detection dicts with bbox
        output_path: Where to save PNG
        title: Optional title; if None, auto-generates from region
        region: Region name (used for title if title is None)
        raster_alpha: Transparency of raster overlay
    """
    if plt is None or Rectangle is None or rasterio is None:
        raise RuntimeError("matplotlib and rasterio are required for map export")

    if title is None:
        title = f"{region} Stadium Detection"

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


def detect_from_qgis_layer(layer, min_area: float = 1000.0, create_output_layer: bool = True, model_path: Optional[str] = None, scaler_path: Optional[str] = None, model_object: Optional[object] = None, region: str = "World", detection_backend: str = "heuristic", text_prompt: str = None, tile_size: int = 800, tile_overlap: int = 100, box_threshold: float = 0.25, text_threshold: float = 0.20, model_id: str = "IDEA-Research/grounding-dino-tiny", device: Optional[str] = None) -> List[Dict]:
    """Detect stadium candidates from a QGIS vector or raster layer.

    This function is intended to be run from the QGIS Python console where `qgis` is available.
    For a raster layer it will run a simple thresholding + polygonize pipeline; for vector layers
    it filters by area and shape.
    
    Args:
        layer: QGIS layer object
        min_area: Minimum area threshold
        create_output_layer: Whether to create output detection layer
        model_path: Optional path to pickled model
        scaler_path: Optional path to pickled scaler
        model_object: Optional model object
        region: Region name (used for output layer title)
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
        backend = (detection_backend or "heuristic").lower()
        if backend in {"grounding_dino", "grounding-dino", "zero-shot", "zeroshot"}:
            detections = detect_stadium_grounding_dino(
                raster_path,
                min_area=int(min_area),
                text_prompt=text_prompt or "stadium. sports field. arena. athletics track. soccer stadium. football stadium.",
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                model_id=model_id,
                device=device,
            )
        else:
            detections = detect_stadium_opencv(
                raster_path,
                min_area=int(min_area),
                model_path=model_path,
                scaler_path=scaler_path,
                model_object=model_object,
            )
        if create_output_layer:
            # attach georeferenced polygons when the layer is a raster
            try:
                crs_authid = layer.crs().authid() or "EPSG:4326"
                output_layer = _create_detection_layer(f"{region} stadium detections", crs_authid)
                _append_detection_features(
                    output_layer,
                    detections,
                    layer.extent(),
                    layer.width(),
                    layer.height(),
                    layer.name(),
                    raster_path,
                )
            except Exception:
                # detection results are still returned even if a map layer cannot be created
                pass

    detections.sort(key=lambda d: d.get("area_m2", d.get("area", 0)), reverse=True)
    return detections


def run_stadium_detection(
    layer=None,
    min_area_pixels: int = 5000,
    create_output_layer: bool = True,
    model_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
    model_object: Optional[object] = None,
    region: str = "World",
    detection_backend: str = "heuristic",
    text_prompt: str = None,
    tile_size: int = 800,
    tile_overlap: int = 100,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    model_id: str = "IDEA-Research/grounding-dino-tiny",
    device: Optional[str] = None,
) -> List[Dict]:
    """Convenience entry point for QGIS Python console.

    If `layer` is omitted, uses `iface.activeLayer()`.
    For raster layers this will also add a polygon output layer to the project.
    Accepts optional `model_path` / `scaler_path` or `model_object` to annotate detections
    with model predictions (fields `model_label` and `model_score`).
    
    Args:
        layer: QGIS layer (or None to use active layer)
        min_area_pixels: Minimum detection area in pixels
        create_output_layer: Whether to create output layer
        model_path: Optional path to pickled model
        scaler_path: Optional path to pickled scaler
        model_object: Optional model object
        region: Region name for output layer title (e.g., "California", "Madagascar")
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

    results = detect_from_qgis_layer(
        layer,
        min_area=float(min_area_pixels),
        create_output_layer=create_output_layer,
        model_path=model_path,
        scaler_path=scaler_path,
        model_object=model_object,
        region=region,
        detection_backend=detection_backend,
        text_prompt=text_prompt,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        model_id=model_id,
        device=device,
    )
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
