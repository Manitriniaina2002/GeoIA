# GeoIA — Stadium Detection (refactor)

This repository is refactored to be a minimal GeoAI project for stadium detection.

Quick commands

- Run a local detection on an image:

```bash
python scripts/geoai_cli.py detect --image data/sample.tif --min-area 5000 --output-png output/overlay.png
```

- Batch process a folder of images:

```bash
python scripts/batch_detect.py --input-dir data/images --out-dir output/batch
```

- Run detection from QGIS via Claude/execute_code using `scripts/claude_detect_wrapper.py` (see guide CLAUDE_DETECTION_GUIDE.md).

Dependencies

Install in your venv:

```bash
pip install -r requirements.txt
```

Next steps

- Add training data generation from `scripts/naip_fingerprint.py` (already included)
- Add unit tests and CI
- Add model training scripts (not included yet)
