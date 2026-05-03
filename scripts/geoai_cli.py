#!/usr/bin/env python
"""CLI for GeoIA stadium detection tasks.

Commands:
  detect   Run detection on a raster in QGIS (via execute_code) or standalone OpenCV
  batch    Run batch detection on images in a folder (OpenCV mode)
  render   Render QGIS project map to PNG (via execute_code)

This is a lightweight wrapper to speed common workflows.
"""
import argparse
import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detect_stadium import detect_stadium_opencv, draw_detections


def detect_local(image_path, min_area=5000, output_png=None):
    det = detect_stadium_opencv(image_path, min_area=min_area)
    result = {"detections": det, "count": len(det)}
    if output_png:
        draw_detections(image_path, det, output_png)
        result["overlay"] = output_png
    print(json.dumps(result, indent=2))


def batch_folder(folder, min_area=5000, out_dir="output/batch"):
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.tif', '.tiff', '.png', '.jpg')):
            path = os.path.join(folder, fname)
            out_png = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_overlay.png")
            try:
                det = detect_stadium_opencv(path, min_area=min_area)
                draw_detections(path, det, out_png)
                results.append({"file": path, "count": len(det), "overlay": out_png})
            except Exception as e:
                results.append({"file": path, "error": str(e)})
    print(json.dumps({"results": results}, indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd')

    d = sub.add_parser('detect')
    d.add_argument('--image', required=True)
    d.add_argument('--min-area', type=int, default=5000)
    d.add_argument('--output-png')

    b = sub.add_parser('batch')
    b.add_argument('--folder', required=True)
    b.add_argument('--min-area', type=int, default=5000)
    b.add_argument('--outdir', default='output/batch')

    args = p.parse_args()
    if args.cmd == 'detect':
        detect_local(args.image, min_area=args.min_area, output_png=args.output_png)
    elif args.cmd == 'batch':
        batch_folder(args.folder, min_area=args.min_area, out_dir=args.outdir)
    else:
        p.print_help()

if __name__ == '__main__':
    main()
