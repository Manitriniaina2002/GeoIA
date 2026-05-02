"""Download sample NAIP imagery and building footprints using OpenGeoAI (`geoai-py`).

Requires: geoai-py and leafmap installed in the current Python environment.

Example:
    python scripts/download_opengeoai_data.py --out data --max-items 1
"""
import argparse
import os
import sys


def ensure_packages():
    try:
        import leafmap  # noqa: F401
        from geoai.download import download_naip, download_overture_buildings  # noqa: F401
    except Exception as e:
        print("This script requires 'geoai-py' and 'leafmap'. Install with:")
        print("  pip install geoai-py leafmap")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data", help="Output directory")
    parser.add_argument("--max-items", type=int, default=1, help="Max NAIP files to download")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("MINX", "MINY", "MAXX", "MAXY"),
                        help="Bounding box: minx miny maxx maxy (lon/lat)")
    args = parser.parse_args()

    ensure_packages()

    from geoai.download import download_naip, download_overture_buildings

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # default bbox (sample) if not provided: small ROI near Seattle
    if args.bbox:
        bbox = tuple(args.bbox)
    else:
        bbox = (-117.6029, 47.65, -117.5936, 47.6563)

    print(f"Downloading NAIP imagery for bbox={bbox} to {out_dir}")
    naip_files = download_naip(bbox=bbox, output_dir=os.path.join(out_dir, "naip_data"), max_items=args.max_items)
    print(f"Downloaded NAIP files: {naip_files}")

    print("Downloading building footprints (Overture) as GeoJSON")
    buildings_path = os.path.join(out_dir, "buildings.geojson")
    data_file = download_overture_buildings(bbox=bbox, output=buildings_path)
    print(f"Buildings saved to: {data_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
