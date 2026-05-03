#!/usr/bin/env python
"""CLI for GeoIA stadium detection tasks.

Commands:
  detect   Run detection on a raster in QGIS (via execute_code) or standalone OpenCV
  batch    Run batch detection on images in a folder (OpenCV mode)
  render   Render QGIS project map to PNG (via execute_code)
  regions  List available regions

This is a lightweight wrapper to speed common workflows.
"""
import argparse
import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detect_stadium import detect_stadium_opencv, detect_stadium_grounding_dino, draw_detections
from scripts.region_utils import list_regions, get_region, get_bounds


def detect_local(image_path, region="World", min_area=5000, output_png=None, backend="heuristic", prompt=None, tile_size=800, tile_overlap=100, box_threshold=0.25, text_threshold=0.20):
    """Run detection on local image."""
    if backend == "grounding_dino":
        det = detect_stadium_grounding_dino(
            image_path,
            min_area=min_area,
            text_prompt=prompt or "stadium. sports field. arena. athletics track. soccer stadium. football stadium.",
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
    else:
        det = detect_stadium_opencv(image_path, min_area=min_area)
    result = {"region": region, "detections": det, "count": len(det)}
    if output_png:
        draw_detections(image_path, det, output_png)
        result["overlay"] = output_png
    print(json.dumps(result, indent=2))


def batch_folder(folder, region="World", min_area=5000, out_dir="output/batch", backend="heuristic", prompt=None, tile_size=800, tile_overlap=100, box_threshold=0.25, text_threshold=0.20):
    """Run batch detection on folder."""
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.tif', '.tiff', '.png', '.jpg')):
            path = os.path.join(folder, fname)
            out_png = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_overlay.png")
            try:
                if backend == "grounding_dino":
                    det = detect_stadium_grounding_dino(
                        path,
                        min_area=min_area,
                        text_prompt=prompt or "stadium. sports field. arena. athletics track. soccer stadium. football stadium.",
                        tile_size=tile_size,
                        tile_overlap=tile_overlap,
                        box_threshold=box_threshold,
                        text_threshold=text_threshold,
                    )
                else:
                    det = detect_stadium_opencv(path, min_area=min_area)
                draw_detections(path, det, out_png)
                results.append({"file": path, "region": region, "count": len(det), "overlay": out_png})
            except Exception as e:
                results.append({"file": path, "error": str(e)})
    print(json.dumps({"region": region, "results": results}, indent=2))


def show_regions():
    """List all available regions."""
    regions = list_regions()
    print("Available regions:")
    for r in sorted(regions):
        region_info = get_region(r)
        bounds = region_info["bounds"]
        print(f"  {r:20s} - {region_info['title']:30s} bounds: {bounds}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd')

    d = sub.add_parser('detect')
    d.add_argument('--image', required=True)
    d.add_argument('--region', default='World', help='Region name (see: geoai_cli regions)')
    d.add_argument('--min-area', type=int, default=5000)
    d.add_argument('--output-png')
    d.add_argument('--backend', choices=['heuristic', 'grounding_dino'], default='grounding_dino')
    d.add_argument('--prompt', default='stadium. sports field. arena. athletics track. soccer stadium. football stadium.')
    d.add_argument('--tile-size', type=int, default=800)
    d.add_argument('--tile-overlap', type=int, default=100)
    d.add_argument('--box-threshold', type=float, default=0.25)
    d.add_argument('--text-threshold', type=float, default=0.20)

    b = sub.add_parser('batch')
    b.add_argument('--folder', required=True)
    b.add_argument('--region', default='World', help='Region name')
    b.add_argument('--min-area', type=int, default=5000)
    b.add_argument('--outdir', default='output/batch')
    b.add_argument('--backend', choices=['heuristic', 'grounding_dino'], default='heuristic')
    b.add_argument('--prompt', default='stadium. sports field. arena. athletics track. soccer stadium. football stadium.')
    b.add_argument('--tile-size', type=int, default=800)
    b.add_argument('--tile-overlap', type=int, default=100)
    b.add_argument('--box-threshold', type=float, default=0.25)
    b.add_argument('--text-threshold', type=float, default=0.20)

    r = sub.add_parser('regions')
    r.set_defaults(func=show_regions)

    args = p.parse_args()
    if args.cmd == 'detect':
        detect_local(args.image, region=args.region, min_area=args.min_area, output_png=args.output_png, backend=args.backend, prompt=args.prompt, tile_size=args.tile_size, tile_overlap=args.tile_overlap, box_threshold=args.box_threshold, text_threshold=args.text_threshold)
    elif args.cmd == 'batch':
        batch_folder(args.folder, region=args.region, min_area=args.min_area, out_dir=args.outdir, backend=args.backend, prompt=args.prompt, tile_size=args.tile_size, tile_overlap=args.tile_overlap, box_threshold=args.box_threshold, text_threshold=args.text_threshold)
    elif args.cmd == 'regions':
        show_regions()
    else:
        p.print_help()

if __name__ == '__main__':
    main()
