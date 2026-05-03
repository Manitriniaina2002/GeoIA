"""Example model training on generated fingerprint data.

This script demonstrates how to train a simple model (RandomForest) on the
generated training data from NAIP fingerprints.

Usage:
  python scripts/train_model.py \
    --train-csv output/training_data/train.csv \
    --val-csv output/training_data/val.csv \
    --output-model output/stadium_detector_model.pkl
"""
import argparse
import json
import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, silhouette_score


def train_anomaly_detector(train_csv, output_model):
    """Train an anomaly detector (Isolation Forest) for stadium detection.
    
    Assumes: fingerprints from stadiums vs non-stadiums.
    For unsupervised learning, we can use Isolation Forest to detect
    anomalies (stadiums are structurally different from typical terrain).
    
    Args:
        train_csv: path to training CSV
        output_model: path to save model
    """
    print(f"Loading training data from {train_csv}...")
    df_train = pd.read_csv(train_csv, index_col=0)
    
    print(f"Training data shape: {df_train.shape}")
    print(f"Training Isolation Forest (anomaly detection)...")
    
    model = IsolationForest(contamination=0.1, random_state=42)
    predictions = model.fit_predict(df_train)
    
    n_anomalies = (predictions == -1).sum()
    print(f"Detected {n_anomalies} anomalies out of {len(df_train)} samples")
    
    # Save model
    Path(output_model).parent.mkdir(parents=True, exist_ok=True)
    with open(output_model, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved model to {output_model}")
    
    return model


def train_classifier(train_csv, val_csv, output_model, labels_csv=None):
    """Train a supervised classifier (RandomForest).
    
    Note: requires labels. If labels_csv is provided, use it;
    otherwise fall back to anomaly detection.
    
    Args:
        train_csv: path to training CSV
        val_csv: path to validation CSV
        output_model: path to save model
        labels_csv: optional CSV with labels (index=filename, column=label)
    """
    print(f"Loading training data from {train_csv}...")
    df_train = pd.read_csv(train_csv, index_col=0)
    
    if val_csv and Path(val_csv).exists():
        print(f"Loading validation data from {val_csv}...")
        df_val = pd.read_csv(val_csv, index_col=0)
    else:
        df_val = None
    
    # If labels provided, do supervised learning
    if labels_csv and Path(labels_csv).exists():
        print(f"Loading labels from {labels_csv}...")
        df_labels = pd.read_csv(labels_csv, index_col=0)
        
        # Merge labels with training data
        y_train = df_labels.loc[df_train.index, 'label'].values
        X_train = df_train.values
        
        print(f"Training RandomForest ({len(np.unique(y_train))} classes)...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate on validation
        if df_val is not None:
            y_val = df_labels.loc[df_val.index, 'label'].values
            X_val = df_val.values
            score = model.score(X_val, y_val)
            print(f"Validation accuracy: {score:.3f}")
        
        print(f"Saved model to {output_model}")
        with open(output_model, 'wb') as f:
            pickle.dump(model, f)
    else:
        print("No labels provided; falling back to anomaly detection...")
        return train_anomaly_detector(train_csv, output_model)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-csv', required=True)
    p.add_argument('--val-csv')
    p.add_argument('--labels-csv', help='Optional labels file (index=filename, label=class)')
    p.add_argument('--output-model', default='output/stadium_detector_model.pkl')
    p.add_argument('--model-type', choices=['anomaly', 'classifier'], default='anomaly')
    args = p.parse_args()
    
    if args.model_type == 'anomaly':
        train_anomaly_detector(args.train_csv, args.output_model)
    else:
        train_classifier(args.train_csv, args.val_csv, args.output_model, args.labels_csv)
    
    print("Training complete!")


if __name__ == '__main__':
    import numpy as np
    main()
