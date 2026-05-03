"""Smoke tests for GeoIA detection and CLI."""
import os
import sys
import tempfile
import pytest
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detect_stadium import detect_stadium_opencv, draw_detections
from scripts.claude_detect_wrapper import list_available_layers, export_detections


class TestDetectionBasics:
    """Test core detection functionality."""

    def test_detect_with_synthetic_image(self):
        """Test detection on a simple synthetic image."""
        # Create a simple test image: 1000x1000 with a bright rectangle (simulated stadium)
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        # Draw a bright rectangle (stadium-like)
        img[200:400, 300:500] = [200, 200, 200]
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            import cv2
            cv2.imwrite(tmp.name, img)
            tmp_path = tmp.name
        
        try:
            detections = detect_stadium_opencv(tmp_path, min_area=1000)
            assert isinstance(detections, list)
            # May or may not detect depending on algorithm; just ensure no crash
        finally:
            os.unlink(tmp_path)

    def test_detect_with_real_data(self):
        """Test detection on real Madagascar data if available."""
        test_raster = PROJECT_ROOT / 'data' / 'madagascar_test' / 'madagascar_antananarivo.tif'
        if test_raster.exists():
            detections = detect_stadium_opencv(str(test_raster), min_area=5000)
            assert isinstance(detections, list)
            # Should find some detections in the real data
            if len(detections) > 0:
                det = detections[0]
                assert 'bbox' in det or 'polygon' in det
                assert 'confidence' in det


class TestCLI:
    """Test CLI commands."""

    def test_cli_detect_help(self):
        """Test that CLI detect command shows help."""
        import subprocess
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / 'scripts' / 'geoai_cli.py'), '--help'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert 'detect' in result.stdout

    def test_cli_detect_with_image(self):
        """Test CLI detect command on a synthetic image."""
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        img[100:200, 100:200] = [150, 150, 150]
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            import cv2
            cv2.imwrite(tmp.name, img)
            tmp_path = tmp.name
        
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / 'scripts' / 'geoai_cli.py'), 
                 'detect', '--image', tmp_path, '--min-area', '100'],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0
            assert 'detections' in result.stdout or 'count' in result.stdout
        finally:
            os.unlink(tmp_path)


class TestWrapperFunctions:
    """Test wrapper functions (require QGIS)."""

    def test_wrapper_functions_importable(self):
        """Test that wrapper functions can be imported (basic import test)."""
        try:
            from scripts.claude_detect_wrapper import (
                run_detection_on_layer,
                list_available_layers,
                export_detections,
                export_detections_as_geojson
            )
            # If we get here, imports succeeded
            assert callable(run_detection_on_layer)
            assert callable(list_available_layers)
            assert callable(export_detections)
        except ImportError as e:
            # QGIS might not be available; that's OK for this test
            pytest.skip(f"QGIS not available: {e}")


class TestNAIPFingerprint:
    """Test NAIP fingerprint pipeline."""

    def test_fingerprint_import(self):
        """Test that fingerprint script can be imported."""
        try:
            from scripts import naip_fingerprint
            assert hasattr(naip_fingerprint, 'compute_fingerprint')
        except ImportError as e:
            pytest.skip(f"NAIP fingerprint deps not available: {e}")

    def test_fingerprint_on_synthetic(self):
        """Test fingerprint computation on synthetic image."""
        try:
            from scripts.naip_fingerprint import compute_fingerprint
            img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            fp = compute_fingerprint(img)
            assert isinstance(fp, dict)
            assert 'histogram_rgb' in fp or 'clahe' in fp
        except ImportError:
            pytest.skip("NAIP fingerprint deps not available")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
