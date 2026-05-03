"""
Claude-compatible wrapper for stadium detection in QGIS.

This module is called by Claude via the execute_code MCP tool.
It provides high-level functions that integrate detection with QGIS project state.

Usage in Claude:
    execute_code('''
    import sys
    sys.path.append(r"C:\\Users\\MaZik\\GeoIA")
    from scripts.claude_detect_wrapper import run_detection_on_layer
    
    result = run_detection_on_layer("layer_name", min_area=5000)
    print(result)
    ''')
"""

import sys
import json
import os
from typing import Dict, List, Any

# Ensure GeoIA is on path
if r"C:\Users\MaZik\GeoIA" not in sys.path:
    sys.path.insert(0, r"C:\Users\MaZik\GeoIA")

from src.detect_stadium import detect_stadium_opencv


def run_detection_on_layer(
    layer_name_or_id: str, 
    min_area: int = 5000,
    create_layer: bool = True,
    draw_overlay: bool = False,
    overlay_path: str = None,
    region: str = "World",
    detection_mode: str = "grounding_dino",
    text_prompt: str = None,
    tile_size: int = 800,
    tile_overlap: int = 100,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    model_id: str = "IDEA-Research/grounding-dino-tiny",
    device: str = None,
    model_path: str = None,
    scaler_path: str = None,
    model_object: Any = None,
) -> Dict[str, Any]:
    """
    Run real stadium detection on an active QGIS raster layer using OpenCV.
    
    Uses the actual detect_stadium.py detection pipeline to find stadium-like shapes.
    Automatically creates a georeferenced polygon layer with results in QGIS.
    
    Args:
        layer_name_or_id: Name or ID of the raster layer to process
        min_area: Minimum pixel area to consider a detection
        create_layer: If True, create a polygon layer with results in QGIS
        draw_overlay: If True, create a PNG overlay with bounding boxes
        overlay_path: Path to save PNG overlay (ignored if draw_overlay=False)
        region: Region name for output layer (e.g., "California", "Madagascar")
    
    Returns:
        Dict with detection results, layer info, and created layer name
    
    Example:
        result = run_detection_on_layer("Madagascar Raster", min_area=5000, region="Madagascar")
        print(f"Found {result['detection_count']} stadiums")
        print(f"Created layer: {result['created_layer']}")
    """
    try:
        from qgis.core import QgsProject, QgsMapLayer
        from src.detect_stadium import detect_from_qgis_layer, draw_detections
        
        project = QgsProject.instance()
        
        # Find the layer by name or ID
        layer = None
        for lid, lyr in project.mapLayers().items():
            if lyr.name() == layer_name_or_id or lid == layer_name_or_id:
                layer = lyr
                break
        
        if not layer:
            return {
                "status": "error",
                "message": f"Layer '{layer_name_or_id}' not found in project",
                "detections": [],
                "detection_count": 0
            }
        
        if layer.type() != QgsMapLayer.RasterLayer:
            return {
                "status": "error", 
                "message": f"Layer '{layer_name_or_id}' is not a raster layer (type: {layer.type()})",
                "detections": [],
                "detection_count": 0
            }
        
        # Get source path for potential overlay
        source_path = layer.source()
        if "|" in source_path:
            source_path = source_path.split("|")[0]
        
        # Run the full QGIS-aware detection pipeline.
        # Grounding DINO zero-shot is the default. A trained model remains available as a separate mode.
        model_args = {}
        backend = detection_mode.lower().strip()
        if backend in {"model", "trained_model", "trained-model"}:
            model_args = {
                "model_path": model_path,
                "scaler_path": scaler_path,
                "model_object": model_object,
                "detection_backend": "heuristic",
            }
            algorithm_label = "OpenCV contour detection with ellipse fitting"
        elif backend in {"heuristic", "opencv", "open_cv"}:
            model_args = {
                "detection_backend": "heuristic",
                "model_path": model_path,
                "scaler_path": scaler_path,
                "model_object": model_object,
            }
            algorithm_label = "OpenCV contour detection with ellipse fitting"
        else:
            model_args = {
                "detection_backend": "grounding_dino",
                "text_prompt": text_prompt,
                "tile_size": tile_size,
                "tile_overlap": tile_overlap,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "model_id": model_id,
                "device": device,
            }
            algorithm_label = "Grounding DINO zero-shot"

        detections = detect_from_qgis_layer(
            layer, 
            min_area=float(min_area), 
            create_output_layer=create_layer,
            region=region,
            **model_args,
        )
        
        # Find the created output layer
        created_layer_name = None
        if create_layer and detections:
            for lid, lyr in project.mapLayers().items():
                if "stadium detections" in lyr.name().lower():
                    created_layer_name = lyr.name()
                    break
        
        # Optionally create PNG overlay
        overlay_result = None
        if draw_overlay and overlay_path and os.path.exists(source_path):
            try:
                draw_detections(source_path, detections, overlay_path)
                overlay_result = overlay_path
            except Exception as e:
                print(f"Warning: Could not create overlay PNG: {e}")
        
        return {
            "status": "success",
            "message": f"Detection complete: found {len(detections)} stadium candidates",
            "layer_name": layer.name(),
            "layer_id": layer.id(),
            "source": source_path,
            "detections": detections,
            "detection_count": len(detections),
            "created_layer": created_layer_name,
            "overlay_path": overlay_result,
            "region": region,
            "parameters": {
                "min_area": min_area,
                "region": region,
                "detection_mode": detection_mode,
                "text_prompt": text_prompt,
                "tile_size": tile_size,
                "tile_overlap": tile_overlap,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "algorithm": algorithm_label
            }
        }
    
    except ImportError as e:
        return {
            "status": "error",
            "message": f"QGIS modules not available. Execute this via QGIS's execute_code tool. Error: {e}",
            "detections": [],
            "detection_count": 0
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": f"Detection failed: {str(e)}",
            "detections": [],
            "detection_count": 0,
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        }



def list_available_layers() -> Dict[str, Any]:
    """
    List all layers currently loaded in the QGIS project.
    
    Returns:
        Dict with layer list and metadata
    """
    try:
        from qgis.core import QgsProject, QgsMapLayer
        
        project = QgsProject.instance()
        layers = []
        
        for layer_id, layer in project.mapLayers().items():
            layer_info = {
                "id": layer_id,
                "name": layer.name(),
                "type": "raster" if layer.type() == QgsMapLayer.RasterLayer else "vector",
                "visible": project.layerTreeRoot().findLayer(layer_id).isVisible()
            }
            
            if layer.type() == QgsMapLayer.RasterLayer:
                layer_info.update({
                    "width": layer.width(),
                    "height": layer.height()
                })
            
            layers.append(layer_info)
        
        return {
            "status": "success",
            "layer_count": len(layers),
            "layers": layers
        }
    
    except ImportError:
        return {
            "status": "error",
            "message": "QGIS modules not available",
            "layers": []
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "layers": []
        }


def export_detections(
    detections: List[Dict],
    output_path: str,
    format: str = "json",
    layer_name_or_id: str = None
) -> Dict[str, Any]:
    """
    Export detection results to JSON or CSV format.
    
    For GeoJSON with georeferencing, use export_detections_as_geojson() instead.
    
    Args:
        detections: List of detection dicts
        output_path: Where to save the file
        format: Output format (json, csv, or geojson)
        layer_name_or_id: If format='geojson', use this layer for georeferencing
    
    Returns:
        Status dict
    """
    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        if format == "json":
            import json
            with open(output_path, "w") as f:
                json.dump(detections, f, indent=2)
        elif format == "csv":
            import csv
            if not detections:
                return {"status": "error", "message": "No detections to export"}
            
            with open(output_path, "w", newline="") as f:
                fieldnames = ["id", "area", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for i, det in enumerate(detections):
                    bbox = det.get("bbox", (0, 0, 0, 0))
                    row = {
                        "id": i,
                        "area": det.get("area", 0),
                        "bbox_x": bbox[0],
                        "bbox_y": bbox[1],
                        "bbox_w": bbox[2],
                        "bbox_h": bbox[3]
                    }
                    writer.writerow(row)
        elif format == "geojson":
            if layer_name_or_id:
                return export_detections_as_geojson(layer_name_or_id, detections, output_path)
            else:
                # Fallback: simple geojson without georeferencing
                import json
                features = []
                for i, det in enumerate(detections):
                    bbox = det.get("bbox", (0, 0, 100, 100))
                    x, y, w, h = bbox
                    feature = {
                        "type": "Feature",
                        "properties": {
                            "id": i,
                            "area_pixels": det.get("area", 0),
                            "coordinate_system": "pixel"
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [x, y], [x+w, y], [x+w, y+h], [x, y+h], [x, y]
                            ]]
                        }
                    }
                    features.append(feature)
                
                geojson_obj = {"type": "FeatureCollection", "features": features}
                with open(output_path, "w") as f:
                    json.dump(geojson_obj, f, indent=2)
        else:
            return {"status": "error", "message": f"Unsupported format: {format}"}
        
        return {
            "status": "success",
            "message": f"Exported {len(detections)} detections to {format}",
            "output_path": output_path,
            "format": format,
            "count": len(detections)
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__
        }


def export_detections_as_geojson(
    layer_name_or_id: str,
    detections: List[Dict],
    output_path: str
) -> Dict[str, Any]:
    """
    Export detections as GeoJSON, converting pixel coordinates to geographic coordinates.
    
    This function finds the raster layer, gets its georeferencing info, and creates
    proper GeoJSON features with geographic coordinates.
    
    Args:
        layer_name_or_id: Name or ID of the raster layer (for georeferencing)
        detections: List of detection dicts with pixel coordinates
        output_path: Where to save the GeoJSON file
    
    Returns:
        Status dict with output path and feature count
    
    Example:
        result = export_detections_as_geojson("Madagascar Raster", detections, "output.geojson")
    """
    try:
        from qgis.core import QgsProject, QgsMapLayer
        import json
        
        project = QgsProject.instance()
        
        # Find the raster layer for georeferencing
        layer = None
        for lid, lyr in project.mapLayers().items():
            if (lyr.name() == layer_name_or_id or lid == layer_name_or_id) and \
               lyr.type() == QgsMapLayer.RasterLayer:
                layer = lyr
                break
        
        if not layer:
            # Fallback: use pixel coordinates as-is
            print(f"Warning: Layer '{layer_name_or_id}' not found. Using pixel coordinates.")
            features = []
            for i, det in enumerate(detections):
                bbox = det.get("bbox", (0, 0, 100, 100))
                x, y, w, h = bbox
                feature = {
                    "type": "Feature",
                    "properties": {
                        "id": i,
                        "area_pixels": det.get("area", 0),
                        "coordinate_system": "pixel"
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [x, y], [x+w, y], [x+w, y+h], [x, y+h], [x, y]
                        ]]
                    }
                }
                features.append(feature)
        else:
            # Use layer's georeferencing to convert pixel -> geographic
            try:
                import rasterio
                source_path = layer.source().split("|")[0]
                
                with rasterio.open(source_path) as src:
                    transform = src.transform
                    crs_authid = layer.crs().authid() or "EPSG:4326"
                
                features = []
                for i, det in enumerate(detections):
                    bbox = det.get("bbox", (0, 0, 100, 100))
                    x, y, w, h = bbox
                    
                    # Convert pixel coords to geographic using rasterio transform
                    lon1, lat1 = transform * (x, y + h)
                    lon2, lat2 = transform * (x + w, y)
                    
                    feature = {
                        "type": "Feature",
                        "properties": {
                            "id": i,
                            "area_pixels": det.get("area", 0),
                            "bbox_pixel": [x, y, w, h],
                            "coordinate_system": crs_authid
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [lon1, lat1], [lon2, lat1], [lon2, lat2], [lon1, lat2], [lon1, lat1]
                            ]]
                        }
                    }
                    features.append(feature)
            except Exception as e:
                print(f"Warning: Could not georeferencer detections: {e}. Using pixel coords.")
                features = []
                for i, det in enumerate(detections):
                    bbox = det.get("bbox", (0, 0, 100, 100))
                    x, y, w, h = bbox
                    feature = {
                        "type": "Feature",
                        "properties": {
                            "id": i,
                            "area_pixels": det.get("area", 0),
                            "coordinate_system": "pixel"
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [x, y], [x+w, y], [x+w, y+h], [x, y+h], [x, y]
                            ]]
                        }
                    }
                    features.append(feature)
        
        geojson_obj = {
            "type": "FeatureCollection",
            "features": features
        }
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(geojson_obj, f, indent=2)
        
        return {
            "status": "success",
            "message": f"Exported {len(features)} detections to GeoJSON",
            "output_path": output_path,
            "format": "geojson",
            "feature_count": len(features)
        }
    
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
