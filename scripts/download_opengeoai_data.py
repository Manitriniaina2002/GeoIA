"""Download OpenGeoAI example data for a bounding box.

Downloads NAIP imagery and Overture building footprints using `geoai-py`.

Example:
    python scripts/download_opengeoai_data.py --bbox -117.6029 47.65 -117.5936 47.6563 --out data/opengeoai --max-items 1
"""
import argparse
import json
import os
import sys
from pathlib import Path


def ensure_packages():
    try:
        import leafmap  # noqa: F401
        from geoai.download import download_naip, download_overture_buildings, extract_building_stats  # noqa: F401
    except Exception as e:
        print("This script requires 'geoai-py' and 'leafmap'. Install with:")
        print("  pip install geoai-py leafmap")
        raise


def main():
    parser = argparse.ArgumentParser(description="Download NAIP imagery and building footprints using OpenGeoAI")
    parser.add_argument("--out", default="data", help="Output directory")
    parser.add_argument("--max-items", type=int, default=1, help="Max NAIP files to download")
    parser.add_argument("--year", type=int, default=None, help="Optional NAIP year filter")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("MINX", "MINY", "MAXX", "MAXY"),
                        help="Bounding box: minx miny maxx maxy (lon/lat)")
    parser.add_argument("--skip-naip", action="store_true", help="Skip downloading NAIP imagery")
    parser.add_argument("--skip-buildings", action="store_true", help="Skip downloading Overture buildings")
    parser.add_argument("--stats", action="store_true", help="Extract building statistics after download")
    args = parser.parse_args()

    ensure_packages()

    from geoai.download import download_naip, download_overture_buildings, extract_building_stats

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # default bbox (sample) if not provided: small ROI near Seattle
    if args.bbox:
        bbox = tuple(args.bbox)
    else:
        bbox = (-117.6029, 47.65, -117.5936, 47.6563)

    summary = {"bbox": bbox, "output_dir": out_dir}

    if not args.skip_naip:
        print(f"Downloading NAIP imagery for bbox={bbox} to {out_dir}")
        naip_kwargs = {"bbox": bbox, "output_dir": os.path.join(out_dir, "naip_data"), "max_items": args.max_items}
        if args.year is not None:
            naip_kwargs["year"] = args.year
        naip_files = download_naip(**naip_kwargs)
        summary["naip_files"] = naip_files
        print(f"Downloaded NAIP files: {naip_files}")
    else:
        summary["naip_files"] = []

    buildings_path = os.path.join(out_dir, "buildings.geojson")
    if not args.skip_buildings:
        print("Downloading building footprints (Overture) as GeoJSON")
        data_file = download_overture_buildings(bbox=bbox, output=buildings_path)
        summary["buildings_file"] = data_file
        print(f"Buildings saved to: {data_file}")

        if args.stats:
            try:
                stats = extract_building_stats(data_file)
                summary["building_stats"] = stats
                stats_path = Path(out_dir) / "building_stats.json"
                with open(stats_path, "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2)
                print(f"Building stats saved to: {stats_path}")
            except Exception as exc:
                summary["building_stats_error"] = str(exc)
                print(f"Warning: could not extract building stats: {exc}")
    else:
        summary["buildings_file"] = None

    summary_path = Path(out_dir) / "opengeoai_download_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
