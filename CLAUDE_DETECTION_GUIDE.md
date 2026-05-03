# Stadium Detection Integration with Claude + QGIS

This guide shows how to use the integrated detection pipeline through Claude.

## Setup

1. **Ensure QGIS is running** with the QGIS MCP plugin enabled and server started
2. **Have a raster layer loaded** in QGIS (e.g., "Madagascar Raster")
3. **Use this exact code in Claude** via the `execute_code` tool

---

## Complete Detection Pipeline

Copy and paste this code into Claude's execute_code tool:

```python
import sys
sys.path.append(r"C:\Users\MaZik\GeoIA")

from scripts.claude_detect_wrapper import (
    run_detection_on_layer, 
    export_detections_as_geojson,
    list_available_layers
)
import json

# Step 1: List available layers
print("=" * 60)
print("Step 1: Available Layers in QGIS")
print("=" * 60)
layers_result = list_available_layers()
print(json.dumps(layers_result, indent=2))

# Step 2: Run detection on the Madagascar raster
print("\n" + "=" * 60)
print("Step 2: Running OpenCV Stadium Detection")
print("=" * 60)
detection_result = run_detection_on_layer(
    "Madagascar Raster",
    min_area=5000,
    create_layer=True,
    draw_overlay=True,
    overlay_path=r"C:\Users\MaZik\GeoIA\output\detections_overlay.png"
)
print(json.dumps(detection_result, indent=2))

# Step 3: Export results as GeoJSON
if detection_result["status"] == "success" and detection_result["detections"]:
    print("\n" + "=" * 60)
    print("Step 3: Exporting Detections as GeoJSON")
    print("=" * 60)
    
    export_result = export_detections_as_geojson(
        "Madagascar Raster",
        detection_result["detections"],
        r"C:\Users\MaZik\GeoIA\output\detected_stadiums_real.geojson"
    )
    print(json.dumps(export_result, indent=2))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"✓ Detected: {detection_result['detection_count']} stadiums")
    print(f"✓ Created QGIS layer: {detection_result['created_layer']}")
    print(f"✓ PNG overlay: {detection_result['overlay_path']}")
    print(f"✓ GeoJSON exported: {export_result['output_path']}")
else:
    print("Detection failed or no results returned.")
    if detection_result.get("error_type"):
        print(f"Error: {detection_result['message']}")
```

---

## What This Does

1. **Lists all layers** currently loaded in QGIS
2. **Runs your actual OpenCV detection** on the Madagascar raster (finds stadium-like shapes)
3. **Creates a QGIS polygon layer** with georeferenced detection results
4. **Generates a PNG overlay** with bounding boxes for visual inspection
5. **Exports GeoJSON** with proper geographic coordinates

---

## Expected Output

```
Step 1: Available Layers in QGIS
- Layer: Madagascar Raster (6144×6144 px)

Step 2: Running OpenCV Stadium Detection
- Found: N stadiums
- Created layer: Madagascar Raster stadium detections
- PNG overlay: C:\Users\MaZik\GeoIA\output\detections_overlay.png

Step 3: Exporting Detections as GeoJSON
- Exported 5 features to: C:\Users\MaZik\GeoIA\output\detected_stadiums_real.geojson

Summary
✓ Detected: N stadiums
✓ Created QGIS layer: Madagascar Raster stadium detections
✓ PNG overlay: C:\Users\MaZik\GeoIA\output\detections_overlay.png
✓ GeoJSON exported: C:\Users\MaZik\GeoIA\output\detected_stadiums_real.geojson
```

---

## After Detection

You can then ask Claude to:

- **Render the map with results:**
  ```
  Use render_map to save: C:\Users\MaZik\GeoIA\output\stadium_detections_map.png
  ```

- **Load the GeoJSON back into QGIS:**
  ```
  Use add_vector_layer to load: C:\Users\MaZik\GeoIA\output\detected_stadiums_real.geojson
  ```

- **Get statistics on the detections:**
  ```
  List the detected stadiums and their sizes
  ```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "QGIS modules not available" | Make sure code is executed via `execute_code` tool from QGIS context |
| "Layer not found" | Check layer name in QGIS exactly (case-sensitive) |
| "No detections" | Try lowering `min_area` parameter (e.g., 1000) or check image contrast |
| "Rasterio error" | Ensure rasterio is installed: `pip install rasterio` |
| "No PNG created" | Check that OpenCV is installed: `pip install opencv-python` |

---

## Key Functions

### `run_detection_on_layer(layer_name, min_area=5000, create_layer=True, draw_overlay=False, overlay_path=None)`

Runs the actual OpenCV detection on a QGIS raster layer.

**Returns:**
- `status`: "success" or "error"
- `detection_count`: Number of stadiums found
- `detections`: List of detections with pixel bounding boxes
- `created_layer`: Name of the QGIS polygon layer created

### `export_detections_as_geojson(layer_name, detections, output_path)`

Exports detections to GeoJSON with geographic coordinates (lat/lon).

**Returns:**
- `output_path`: Path to saved GeoJSON file
- `feature_count`: Number of features exported

### `list_available_layers()`

Lists all raster and vector layers currently in the QGIS project.

**Returns:**
- `layers`: List of layer info dicts with name, type, dimensions

---

## Integration With Your Existing Detection

This wrapper uses your existing:
- `detect_stadium_opencv()` — OpenCV contour detection
- `draw_detections()` — PNG overlay visualization
- `_bbox_to_polygon()` — Pixel-to-geographic coordinate conversion

No changes needed to your core detection logic!
