"""Quick example: end-to-end fingerprint → training data → model.

This demonstrates the full pipeline with synthetic data.
"""
import tempfile
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_training_data import load_fingerprints, create_dataset, flatten_fingerprint


def create_synthetic_fingerprints(n_samples=20, output_dir='data/naip/fingerprints'):
    """Create synthetic fingerprints for demonstration."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i in range(n_samples):
        # Synthetic fingerprint with random features
        fp = {
            'histogram_rgb': [
                np.random.randint(0, 256, 256).tolist(),
                np.random.randint(0, 256, 256).tolist(),
                np.random.randint(0, 256, 256).tolist(),
            ],
            'clahe': {
                'contrast': float(np.random.uniform(0.5, 3.0)),
                'brightness': float(np.random.uniform(50, 200)),
            },
            'lbp_histogram': np.random.randint(0, 100, 59).tolist(),
            'stats': {
                'mean': float(np.random.uniform(100, 150)),
                'std': float(np.random.uniform(20, 60)),
            }
        }
        
        output_file = output_path / f'synthetic_{i:03d}.json'
        with open(output_file, 'w') as f:
            json.dump(fp, f)
    
    print(f"Created {n_samples} synthetic fingerprints in {output_dir}")
    return output_dir


def example_pipeline():
    """Run example end-to-end pipeline."""
    print("=" * 60)
    print("Example: Fingerprints → Training Data → Model")
    print("=" * 60)
    
    # 1. Create synthetic fingerprints
    print("\n1. Creating synthetic fingerprints...")
    fp_dir = create_synthetic_fingerprints(n_samples=30)
    
    # 2. Load and convert to dataset
    print("\n2. Loading fingerprints and flattening...")
    fingerprints = load_fingerprints(fp_dir)
    print(f"   Loaded {len(fingerprints)} fingerprints")
    
    # Example: flatten one
    if fingerprints:
        _, first_fp = fingerprints[0]
        flat = flatten_fingerprint(first_fp)
        print(f"   Example flattened features: {list(flat.keys())[:5]}")
    
    # 3. Create dataset
    print("\n3. Creating dataset...")
    df, scaler = create_dataset(fingerprints, normalize=True)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Features: {list(df.columns)[:5]}... ({len(df.columns)} total)")
    
    # 4. Quick ML: fit a simple model
    print("\n4. Training simple anomaly detector...")
    from sklearn.ensemble import IsolationForest
    model = IsolationForest(contamination=0.2, random_state=42)
    predictions = model.fit_predict(df.values)
    n_anomalies = (predictions == -1).sum()
    print(f"   Detected {n_anomalies} anomalies out of {len(df)}")
    
    # 5. Feature importance (if available)
    print("\n5. Model summary:")
    print(f"   Contamination: {model.contamination}")
    print(f"   Outliers: {n_anomalies}/{len(df)}")
    
    print("\n" + "=" * 60)
    print("Example complete! You can now:")
    print("  - Scale this to real NAIP tiles")
    print("  - Train supervised models with labeled data")
    print("  - Export model for inference")
    print("=" * 60)


if __name__ == '__main__':
    example_pipeline()
