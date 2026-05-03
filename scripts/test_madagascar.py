"""End-to-end Madagascar test for GeoIA.

This offline test generates a synthetic Madagascar georeferenced raster,
runs the stadium detector, and saves a PNG overlay.

Example:
    python scripts/test_madagascar.py --output-dir data/madagascar_test
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_bounds


def build_synthetic_madagascar_raster(output_path: Path, width: int = 1400, height: int = 1000):
    """Create a Madagascar-area raster with a stadium-like oval feature."""
    # Antananarivo, Madagascar bounding box in lon/lat.
    bounds = (47.35, -18.98, 47.62, -18.82)

    rng = np.random.default_rng(42)
    base = rng.integers(35, 95, size=(height, width, 3), dtype=np.uint8)

    # Bright stadium-like oval and an inner track ring.
    center = (int(width * 0.62), int(height * 0.48))
    axes = (int(width * 0.16), int(height * 0.10))
    cv2.ellipse(base, center, axes, 18, 0, 360, (215, 215, 215), -1)
    cv2.ellipse(base, center, (int(axes[0] * 0.82), int(axes[1] * 0.82)), 18, 0, 360, (165, 165, 165), 8)
    cv2.ellipse(base, center, (int(axes[0] * 0.68), int(axes[1] * 0.50)), 18, 0, 360, (95, 150, 95), -1)

    # Add a few city-like bright rectangles nearby.
    cv2.rectangle(base, (120, 120), (210, 220), (130, 130, 140), -1)
    cv2.rectangle(base, (280, 680), (420, 780), (120, 125, 135), -1)
    cv2.rectangle(base, (920, 160), (1030, 250), (140, 140, 150), -1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_bounds(*bounds, width, height)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": base.dtype,
        "crs": "EPSG:4326",
        "transform": transform,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        for band in range(3):
            dst.write(base[:, :, band], band + 1)

    return output_path, bounds


def main():
    parser = argparse.ArgumentParser(description="GeoIA Madagascar synthetic test")
    parser.add_argument("--output-dir", default="data/madagascar_test")
    parser.add_argument("--min-area", type=int, default=5000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.append(str(repo_root))
    from src.detect_stadium import detect_stadium_opencv, draw_georeferenced_map

    raster_path, bounds = build_synthetic_madagascar_raster(output_dir / "madagascar_synthetic.tif")
    detections = detect_stadium_opencv(str(raster_path), min_area=args.min_area)
    png_path = output_dir / "madagascar_map.png"
    draw_georeferenced_map(str(raster_path), detections, str(png_path))

    print(json.dumps({
        "bounds": bounds,
        "raster_path": str(raster_path),
        "png_path": str(png_path),
        "detections": len(detections),
    }, indent=2))


if __name__ == "__main__":
    main()
