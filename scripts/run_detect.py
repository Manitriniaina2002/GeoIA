"""Runner script to call the detection code (standalone OpenCV mode).

Example:
    python scripts/run_detect.py --image data/aerial.tif
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.detect_stadium import detect_stadium_opencv, draw_detections
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to aerial image")
    parser.add_argument("--min-area", type=int, default=5000)
    parser.add_argument("--output-png", help="Optional PNG overlay output path")
    args = parser.parse_args()
    det = detect_stadium_opencv(args.image, min_area=args.min_area)
    if args.output_png:
        output_path = draw_detections(args.image, det, args.output_png)
        print(f"Saved overlay PNG to: {output_path}")
    print(json.dumps(det, indent=2))


if __name__ == "__main__":
    main()
