#!/usr/bin/env python3
"""
train_model.py
==============
Trains an SVM on the QML-selected features and saves it as model.pkl.
Run this ONCE locally before deploying to Hugging Face Spaces.

QML-Selected Features (Quantum Wrapper, VQC, best accuracy 72%):
  - Std_ExG        (Variability in Excess Green Index)
  - Mean_RBR       (Red-Blue Ratio)
  - Mean_B         (Mean Blue Channel)
  - Correlation    (GLCM Texture Correlation)
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Paths
HERE     = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, '..', 'QuantumFeatureSelection', 'icml_features_FULL.csv')
OUT_PKL  = os.path.join(HERE, 'model.pkl')

# QML-selected features (best subset from VQC wrapper)
QML_FEATURES = ['Std_ExG', 'Mean_RBR', 'Mean_B', 'Correlation']

def main():
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna()

    # Label encoding
    df['y'] = (df['Label'] == 'Pre_Defoliation').astype(int)

    X = df[QML_FEATURES].values
    y = df['y'].values

    print(f"  Samples: {len(X)} | Pre: {y.sum()} | Post: {(1-y).sum()}")

    # Build pipeline: scaler + SVM
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=10, gamma='scale',
                    probability=True, random_state=42))
    ])

    # Cross-validate
    print("\nRunning 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    print(f"  CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    # Train on full dataset
    print("\nFitting on full dataset...")
    pipeline.fit(X, y)

    # Quick sanity check
    preds = pipeline.predict(X)
    print(classification_report(y, preds, target_names=['Post_Defoliation', 'Pre_Defoliation']))

    # Save
    joblib.dump(pipeline, OUT_PKL)
    print(f"\n✅ Model saved to: {OUT_PKL}")

if __name__ == '__main__':
    main()
