"""
Render the current QGIS project (or selected layers) to a PNG image.
Run this inside QGIS Python console or via the MCP `execute_code` tool.
"""
from qgis.core import QgsProject, QgsMapSettings, QgsMapLayer
from qgis.PyQt.QtCore import QSize
from qgis.PyQt.QtGui import QColor


def render_project_map(output_path: str, width: int = 1200, height: int = 800, layer_names: list = None, dpi: int = 96) -> dict:
    """
    Render the project or specific layers to an image file.

    Args:
        output_path: Full path to write PNG
        width: image width in pixels
        height: image height in pixels
        layer_names: optional list of layer names to include (if None use all project layers)
        dpi: output DPI

    Returns:
        dict with status and path or error
    """
    try:
        project = QgsProject.instance()

        # Collect layers and attempt to ensure rasters render in RGB when possible
        layers = []
        layer_diagnostics = []
        for lid, lyr in project.mapLayers().items():
            if layer_names is None or lyr.name() in layer_names:
                # Try to coerce raster symbology to multiband color if band count >= 3
                try:
                    if lyr.type() == QgsMapLayer.RasterLayer:
                        provider = lyr.dataProvider()
                        band_count = provider.bandCount()
                        layer_diagnostics.append({"id": lid, "name": lyr.name(), "type": "raster", "band_count": band_count})
                        if band_count and band_count >= 3:
                            try:
                                from qgis.core import QgsMultiBandColorRenderer
                                # use bands 1,2,3 for RGB rendering
                                renderer = QgsMultiBandColorRenderer(provider, 1, 2, 3)
                                lyr.setRenderer(renderer)
                                lyr.triggerRepaint()
                            except Exception:
                                # If renderer class or operation not available, ignore and continue
                                pass
                    else:
                        layer_diagnostics.append({"id": lid, "name": lyr.name(), "type": "vector"})
                except Exception:
                    # Non-fatal: collect layer and continue
                    layer_diagnostics.append({"id": lid, "name": lyr.name(), "type": "unknown"})

                layers.append(lyr)

        if not layers:
            return {"status": "error", "message": "No matching layers found in project", "layers": layer_diagnostics}

        # Create map settings
        ms = QgsMapSettings()
        ms.setLayers(layers)

        # Compute combined extent from selected layers
        extent = None
        for lyr in layers:
            if extent is None:
                extent = lyr.extent()
            else:
                extent.combineExtentWith(lyr.extent())

        if extent is None:
            return {"status": "error", "message": "Could not compute map extent"}

        ms.setExtent(extent)
        ms.setOutputSize(QSize(width, height))
        ms.setBackgroundColor(QColor(255, 255, 255))
        ms.setOutputDpi(dpi)

        # Render with QgsMapRendererParallelJob for thread-safe rendering
        from qgis._core import QgsMapRendererParallelJob
        render = QgsMapRendererParallelJob(ms)
        render.start()
        render.waitForFinished()

        img = render.renderedImage()
        if img.save(output_path):
            return {"status": "success", "output_path": output_path, "layers": layer_diagnostics}
        else:
            return {"status": "error", "message": f"Failed to save image to {output_path}", "layers": layer_diagnostics}

    except Exception as e:
        import traceback
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    # Quick local test when run in a QGIS Python environment
    out = r"C:\Users\MaZik\GeoIA\output\qgis_project_render.png"
    print(render_project_map(out))
