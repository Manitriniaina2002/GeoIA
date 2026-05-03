"""Generate training data from NAIP fingerprints.

Reads computed fingerprints (JSON) and aggregates into a training dataset (CSV/pickle).
Handles feature normalization, label encoding, and train/val split.

Usage:
  python scripts/generate_training_data.py \
    --input-dir data/naip/fingerprints \
    --output-dir output/training_data \
    --format csv \
    --train-split 0.8 \
    --normalize
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_fingerprints(input_dir):
    """Load all fingerprints from directory.
    
    Args:
        input_dir: path to directory containing *.json fingerprint files
    
    Returns:
        list of (filename, fingerprint_dict) tuples
    """
    fingerprints = []
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Warning: input dir {input_dir} does not exist; returning empty list")
        return fingerprints
    
    for json_file in input_path.glob('*.json'):
        try:
            with open(json_file) as f:
                fp = json.load(f)
                fingerprints.append((json_file.stem, fp))
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    print(f"Loaded {len(fingerprints)} fingerprints")
    return fingerprints


def flatten_fingerprint(fp):
    """Flatten fingerprint dict into feature vector.
    
    Extracts:
    - Histogram bins (R, G, B channels)
    - CLAHE stats (contrast, brightness)
    - LBP histogram
    - Misc stats
    
    Args:
        fp: fingerprint dict
    
    Returns:
        dict of flat features
    """
    features = {}
    
    # Histogram (if present)
    if 'histogram_rgb' in fp:
        hist = fp['histogram_rgb']
        if isinstance(hist, list) and len(hist) > 0:
            features.update({
                f'hist_r_mean': np.mean(hist[0]) if len(hist) > 0 else 0,
                f'hist_g_mean': np.mean(hist[1]) if len(hist) > 1 else 0,
                f'hist_b_mean': np.mean(hist[2]) if len(hist) > 2 else 0,
            })
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if 'clahe' in fp:
        clahe = fp['clahe']
        if isinstance(clahe, dict):
            for k, v in clahe.items():
                if isinstance(v, (int, float)):
                    features[f'clahe_{k}'] = v
    
    # LBP (Local Binary Patterns)
    if 'lbp_histogram' in fp:
        lbp = fp['lbp_histogram']
        if isinstance(lbp, list) and len(lbp) > 0:
            features.update({
                'lbp_mean': np.mean(lbp),
                'lbp_std': np.std(lbp) if len(lbp) > 1 else 0,
                'lbp_max': np.max(lbp) if len(lbp) > 0 else 0,
            })
    
    # Other stats
    if 'stats' in fp and isinstance(fp['stats'], dict):
        for k, v in fp['stats'].items():
            if isinstance(v, (int, float)):
                features[f'stat_{k}'] = v
    
    # Ensure all features are numeric
    for k in list(features.keys()):
        if not isinstance(features[k], (int, float)):
            features.pop(k)
    
    return features


def create_dataset(fingerprints, normalize=False):
    """Create dataset DataFrame from fingerprints.
    
    Args:
        fingerprints: list of (filename, fp_dict) tuples
        normalize: whether to apply StandardScaler
    
    Returns:
        DataFrame with features, index is filename
    """
    data = []
    for filename, fp in fingerprints:
        features = flatten_fingerprint(fp)
        features['filename'] = filename
        data.append(features)
    
    if not data:
        print("Warning: no fingerprints to convert to dataset")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    df.set_index('filename', inplace=True)
    
    # Handle missing values
    df.fillna(0, inplace=True)
    
    # Normalize if requested
    if normalize and len(df) > 0:
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df),
            columns=df.columns,
            index=df.index
        )
        return df_scaled, scaler
    
    return df, None


def save_training_data(df, output_dir, format='csv', train_split=0.8, scaler=None):
    """Save training dataset in specified format.
    
    Args:
        df: DataFrame with features
        output_dir: output directory
        format: 'csv' or 'pickle'
        train_split: fraction for training set
        scaler: StandardScaler (if used) to save
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Split into train/val if enough data
    if len(df) >= 2:
        train, val = train_test_split(df, train_size=train_split, random_state=42)
    else:
        train, val = df, df.iloc[:0]
    
    # Save
    if format == 'csv':
        train_path = output_path / 'train.csv'
        val_path = output_path / 'val.csv'
        train.to_csv(train_path)
        val.to_csv(val_path)
        print(f"Saved train: {train_path} ({len(train)} samples)")
        print(f"Saved val: {val_path} ({len(val)} samples)")
    elif format == 'pickle':
        train_path = output_path / 'train.pkl'
        val_path = output_path / 'val.pkl'
        train.to_pickle(train_path)
        val.to_pickle(val_path)
        print(f"Saved train: {train_path}")
        print(f"Saved val: {val_path}")
    
    # Save scaler if present
    if scaler is not None:
        import pickle
        scaler_path = output_path / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Saved scaler: {scaler_path}")
    
    # Save metadata
    meta_path = output_path / 'metadata.json'
    meta = {
        'n_samples': len(df),
        'n_train': len(train),
        'n_val': len(val),
        'n_features': len(df.columns),
        'features': list(df.columns),
        'normalized': scaler is not None,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', default='data/naip/fingerprints')
    p.add_argument('--output-dir', default='output/training_data')
    p.add_argument('--format', choices=['csv', 'pickle'], default='csv')
    p.add_argument('--train-split', type=float, default=0.8)
    p.add_argument('--normalize', action='store_true', default=False)
    args = p.parse_args()
    
    print(f"Loading fingerprints from {args.input_dir}...")
    fingerprints = load_fingerprints(args.input_dir)
    
    if not fingerprints:
        print("No fingerprints found; skipping dataset creation")
        return
    
    print(f"Creating dataset with normalization={args.normalize}...")
    if args.normalize:
        df, scaler = create_dataset(fingerprints, normalize=True)
    else:
        df, scaler = create_dataset(fingerprints, normalize=False)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns)[:5]}... ({len(df.columns)} total)")
    
    print(f"Saving to {args.output_dir} (format={args.format})...")
    save_training_data(df, args.output_dir, format=args.format, 
                       train_split=args.train_split, scaler=scaler)
    
    print("Done!")


if __name__ == '__main__':
    main()
