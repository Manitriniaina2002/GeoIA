#!/usr/bin/env python
"""Example: using GeoIA with different regions.

This demonstrates how to run stadium detection on any region.
"""
import sys
import json
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.region_utils import list_regions, get_region, add_custom_region


def example_list_regions():
    """Show available regions."""
    print("=" * 70)
    print("Available Regions in GeoIA")
    print("=" * 70)
    
    regions = list_regions()
    print(f"\nTotal regions: {len(regions)}\n")
    
    for i, region_name in enumerate(sorted(regions), 1):
        region = get_region(region_name)
        bounds = region['bounds']
        title = region['title']
        print(f"{i:2d}. {region_name:20s} → {title:30s}")
        print(f"    Bounds: {bounds}\n")


def example_custom_region():
    """Add and use a custom region."""
    print("=" * 70)
    print("Custom Region Example")
    print("=" * 70)
    
    # Define a custom region (e.g., São Paulo, Brazil)
    print("\nAdding custom region: 'sao_paulo'")
    add_custom_region(
        name="sao_paulo",
        bounds=(-46.6, -24.0, -46.3, -23.5),
        title="São Paulo, Brazil",
        center=(-46.45, -23.75)
    )
    
    region = get_region("sao_paulo")
    print(f"Created region: {region['title']}")
    print(f"  Bounds: {region['bounds']}")
    print(f"  Center: {region['center']}")
    
    print("\nNow you can run detection with:")
    print("  python scripts/geoai_cli.py detect --image raster.tif --region sao_paulo")


def example_region_in_qgis():
    """Show how to use region in QGIS."""
    print("=" * 70)
    print("QGIS Detection with Region")
    print("=" * 70)
    
    code = """
# In QGIS Python console
from scripts.claude_detect_wrapper import run_detection_on_layer
from scripts.region_utils import get_region

# Example 1: Detect in California
result = run_detection_on_layer(
    "my_raster_layer",
    region="california",
    min_area=5000,
    create_layer=True
)
print(f"Found {result['detection_count']} stadiums in {result['region']}")

# Example 2: Detect in Madagascar
result = run_detection_on_layer(
    "my_raster_layer",
    region="madagascar",
    min_area=5000,
    create_layer=True
)
print(f"Created layer: {result['created_layer']}")

# Example 3: With model scoring
from src.detect_stadium import load_model
model = load_model(r"C:\\path\\to\\stadium_detector_model.pkl")

result = run_detection_on_layer(
    "my_raster_layer",
    region="texas",
    min_area=5000,
    create_layer=True,
    model_object=model
)
print(f"Detections with ML scores: {result['detection_count']}")
"""
    
    print("\n" + code)


def main():
    """Run all examples."""
    print("\n")
    example_list_regions()
    print("\n" * 2)
    example_custom_region()
    print("\n" * 2)
    example_region_in_qgis()
    print("\n")


if __name__ == "__main__":
    main()
