# Training Data Generation Guide

This guide explains how to generate and prepare training data for the stadium detection model.

## Pipeline Overview

```
NAIP Tiles (GeoTIFF)
    ↓ [naip_fingerprint.py]
Fingerprints (JSON: histogram, CLAHE, LBP)
    ↓ [generate_training_data.py]
Training Dataset (CSV: normalized features)
    ↓ [train_model.py]
Trained Model (pickle)
```

## Step 1: Download NAIP Tiles and Compute Fingerprints

```bash
# Download NAIP tiles for a region and compute fingerprints
python scripts/naip_fingerprint.py \
  --region madagascar \
  --output-dir data/naip/fingerprints \
  --tile-size 512
```

Output: `data/naip/fingerprints/{tile_id}.json` files

Each JSON contains:
- `histogram_rgb`: RGB channel histograms
- `clahe`: Contrast-Limited Adaptive Histogram Equalization stats
- `lbp_histogram`: Local Binary Pattern histogram
- `stats`: image statistics (mean, std, etc.)

## Step 2: Generate Training Dataset

```bash
# Aggregate all fingerprints into train/val splits
python scripts/generate_training_data.py \
  --input-dir data/naip/fingerprints \
  --output-dir output/training_data \
  --format csv \
  --train-split 0.8 \
  --normalize
```

Outputs:
- `output/training_data/train.csv`: Training features (80%)
- `output/training_data/val.csv`: Validation features (20%)
- `output/training_data/scaler.pkl`: Normalization scaler (if --normalize used)
- `output/training_data/metadata.json`: Dataset metadata

## Step 3: Train a Model

### Option A: Anomaly Detection (Unsupervised)

```bash
# Train an Isolation Forest to detect stadiums as anomalies
python scripts/train_model.py \
  --train-csv output/training_data/train.csv \
  --val-csv output/training_data/val.csv \
  --model-type anomaly \
  --output-model output/stadium_detector_model.pkl
```

### Option B: Supervised Classification (with labels)

First, create a labels CSV:
```csv
filename,label
tile_001,stadium
tile_002,non_stadium
tile_003,stadium
...
```

Then:
```bash
python scripts/train_model.py \
  --train-csv output/training_data/train.csv \
  --val-csv output/training_data/val.csv \
  --labels-csv data/labels.csv \
  --model-type classifier \
  --output-model output/stadium_detector_model.pkl
```

## Using the Trained Model

```python
import pickle
import pandas as pd

# Load model and scaler
with open('output/stadium_detector_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load scaler (if used)
with open('output/training_data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load fingerprint and normalize
fingerprint = load_fingerprint('tile.json')
features = flatten_fingerprint(fingerprint)
X = scaler.transform([features])

# Predict
prediction = model.predict(X)
```

## Example: Full Pipeline

```bash
# 1. Fingerprint NAIP tiles
python scripts/naip_fingerprint.py --region madagascar --output-dir data/naip/fingerprints

# 2. Generate training data
python scripts/generate_training_data.py \
  --input-dir data/naip/fingerprints \
  --output-dir output/training_data \
  --normalize

# 3. Train model
python scripts/train_model.py \
  --train-csv output/training_data/train.csv \
  --val-csv output/training_data/val.csv \
  --output-model output/stadium_detector_model.pkl

# 4. Inspect results
cat output/training_data/metadata.json
```

## Troubleshooting

**No fingerprints found:**
- Check that NAIP tiles are in `data/naip/fingerprints/` as `.json` files
- Run `naip_fingerprint.py` first to generate them

**Training data has few samples:**
- Download more NAIP tiles (requires USDA index)
- Current placeholder URL needs to be replaced with real NAIP index

**Model not improving:**
- Try supervised classification with manually labeled data
- Increase `contamination` parameter in Isolation Forest if too many false positives

## Next Steps

- [ ] Integrate trained model into `src/detect_stadium.py` for faster inference
- [ ] Use model predictions as features in ensemble with OpenCV detection
- [ ] Add cross-validation and hyperparameter tuning
- [ ] Export model to ONNX for deployment
