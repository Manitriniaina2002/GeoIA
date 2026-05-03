"""
Download NAIP tiles (via AWS Public Datasets) for given bounding boxes and compute simple image fingerprints.

Requirements (install in your venv):
    pip install rasterio requests numpy scikit-image tqdm

Usage:
    python scripts/naip_fingerprint.py --bbox MINX MINY MAXX MAXY --zoom 17 --outdir data/naip

The script will:
- Find NAIP tiles via AWS S3 URL pattern (assumes public NAIP tiles available)
- Download tiles overlapping the bbox
- For each tile compute fingerprints:
  - RGB histograms
  - Haralick texture features (via skimage)
  - CLAHE-based contrast stats
- Save fingerprints as JSON per-tile under `outdir/fingerprints/`

Note: NAIP access patterns vary; this script uses a heuristic URL pattern and will work if the tiles for your area exist in the public NAIP repo. For production, use a catalog (e.g., USDA NAIP index) to resolve exact tile URLs.
"""
import argparse
import os
import json
import math
from pathlib import Path
from urllib.parse import urljoin

import requests
import numpy as np
from tqdm import tqdm

try:
    import rasterio
    from skimage import exposure, color, feature
except Exception:
    rasterio = None


def bbox_to_tile_indices(minx, miny, maxx, maxy, zoom=17):
    # rough conversion: use slippy tile scheme to identify candidate tiles
    def deg2num(lon_deg, lat_deg, z):
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** z
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
        return xtile, ytile

    x0, y0 = deg2num(minx, maxy, zoom)
    x1, y1 = deg2num(maxx, miny, zoom)
    xs = range(min(x0, x1), max(x0, x1) + 1)
    ys = range(min(y0, y1), max(y0, y1) + 1)
    return xs, ys


def download_tile(url, out_path):
    resp = requests.get(url, stream=True, timeout=30)
    if resp.status_code != 200:
        return False
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(1024 * 64):
            if chunk:
                f.write(chunk)
    return True


def compute_fingerprint(raster_path):
    if rasterio is None:
        raise RuntimeError("rasterio and skimage required")

    with rasterio.open(raster_path) as src:
        array = src.read([1, 2, 3])  # read first three bands
        transform = src.transform

    # convert to HxWxC
    img = np.moveaxis(array, 0, -1)
    # Basic stats
    stats = {
        "mean": img.mean(axis=(0, 1)).tolist(),
        "std": img.std(axis=(0, 1)).tolist(),
        "min": img.min(axis=(0, 1)).tolist(),
        "max": img.max(axis=(0, 1)).tolist(),
    }

    # RGB histograms
    hists = []
    for c in range(3):
        hist, _ = np.histogram(img[:, :, c].ravel(), bins=64, range=(0, 255))
        hists.append(hist.tolist())

    # Convert to grayscale for texture
    gray = color.rgb2gray(img.astype(np.uint8))
    # CLAHE
    clahe = exposure.equalize_adapthist(gray, clip_limit=0.03)
    clahe_stats = {
        "mean": float(np.mean(clahe)),
        "std": float(np.std(clahe)),
    }

    # Haralick-like texture: use local binary pattern as a simple descriptor
    lbp = feature.local_binary_pattern((gray * 255).astype(np.uint8), P=8, R=1, method="uniform")
    (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
    lbp_hist = lbp_hist.tolist()

    return {
        "stats": stats,
        "histograms": hists,
        "clahe": clahe_stats,
        "lbp": lbp_hist,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", nargs=4, type=float, required=True, help="MINX MINY MAXX MAXY (lon/lat)")
    parser.add_argument("--zoom", type=int, default=17)
    parser.add_argument("--outdir", default="data/naip")
    args = parser.parse_args()

    minx, miny, maxx, maxy = args.bbox
    outdir = Path(args.outdir)
    tiles_dir = outdir / "tiles"
    fp_dir = outdir / "fingerprints"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    fp_dir.mkdir(parents=True, exist_ok=True)

    xs, ys = bbox_to_tile_indices(minx, miny, maxx, maxy, zoom=args.zoom)

    # NAIP public tiles vary by provider; try a best-effort URL pattern (this may need adjustment)
    # Example AWS NAIP read pattern: https://s3.amazonaws.com/elevation-tiles-prod/tiles/{z}/{x}/{y}.png
    # For NAIP, users often use AWS Open Data with different prefixes; here we attempt a placeholder
    base_url_template = "https://naip-public.s3.amazonaws.com/{z}/{x}/{y}.tif"

    results = []
    for x in tqdm(xs, desc="x tiles"):
        for y in ys:
            url = base_url_template.format(z=args.zoom, x=x, y=y)
            out_path = tiles_dir / f"naip_{args.zoom}_{x}_{y}.tif"
            ok = False
            if not out_path.exists():
                try:
                    ok = download_tile(url, out_path)
                except Exception:
                    ok = False
            else:
                ok = True

            if not ok:
                # skip if tile not available
                continue

            try:
                fp = compute_fingerprint(str(out_path))
                fp_path = fp_dir / (out_path.stem + ".json")
                with open(fp_path, "w") as f:
                    json.dump({"tile": str(out_path), "fingerprint": fp}, f, indent=2)
                results.append({"tile": str(out_path), "fingerprint": fp_path.as_posix()})
            except Exception as e:
                # log and continue
                print(f"Failed fingerprint for {out_path}: {e}")

    print(json.dumps({"tiles_processed": len(results), "results": results}, indent=2))


if __name__ == "__main__":
    main()
