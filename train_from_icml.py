#!/usr/bin/env python3
"""
train_from_icml.py
==================
Extracts features DIRECTLY from raw ICML folder images and retrains the SVM.

Folder labeling (by name):
  Contains 'post' (any case)  → Post_Defoliation
  Contains 'pre'  (any case)  → Pre_Defoliation

Features used (overflow-safe):
  Std_ExG       – std of Excess Green Index   (high = leafy canopy)
  Log_RBR       – log(1 + mean(R/B))           (safe, was overflowing to 500 000+)
  Mean_B        – mean Blue channel            (POST fields have higher blue)
  ExG_pos_frac  – fraction of pixels where ExG > 0  (strong leaf indicator)
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import joblib
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")

ICML_ROOT    = '/Volumes/T9/ICML'
HERE         = os.path.dirname(os.path.abspath(__file__))
OUT_PKL      = os.path.join(HERE, 'model.pkl')
QML_FEATURES = ['Std_ExG', 'Log_RBR', 'Mean_B', 'ExG_pos_frac']
VALID_EXT    = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG'}
LIMIT_PER_FOLDER = None   # set to e.g. 200 for quick test


def label_from_folder(folder_name: str) -> str | None:
    fn = folder_name.lower()
    if 'post' in fn:
        return 'Post_Defoliation'
    if 'pre' in fn:
        return 'Pre_Defoliation'
    return None


def extract_features(img_bgr: np.ndarray) -> dict | None:
    """Safe feature extraction matching app.py's extract_features()."""
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(float) / 255.0
        R, G, B = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]

        ExG = 2*G - R - B
        RBR = R / (B + 1e-6)

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = img_gray.shape
        if h > 512 or w > 512:
            img_gray = cv2.resize(img_gray, (min(w, 512), min(h, 512)))

        glcm = graycomatrix(img_gray, distances=[1], angles=[0],
                            levels=256, symmetric=True, normed=True)

        return {
            'Std_ExG':      float(np.std(ExG)),
            'Log_RBR':      float(np.log1p(np.mean(RBR))),
            'Mean_B':       float(np.mean(B)),
            'ExG_pos_frac': float(np.mean(ExG > 0)),
            # extras stored for analysis but not used in SVM
            'Mean_ExG':     float(np.mean(ExG)),
            'Mean_G':       float(np.mean(G)),
            'Mean_R':       float(np.mean(R)),
            'Correlation':  float(graycoprops(glcm, 'correlation')[0, 0]),
        }
    except Exception as e:
        return None


def load_dataset():
    records = []
    for folder in sorted(os.listdir(ICML_ROOT)):
        folder_path = os.path.join(ICML_ROOT, folder)
        if not os.path.isdir(folder_path):
            continue
        label = label_from_folder(folder)
        if label is None:
            print(f"  [SKIP] {folder} — no pre/post in name")
            continue

        files = [f for f in os.listdir(folder_path)
                 if os.path.splitext(f)[1] in VALID_EXT]
        if LIMIT_PER_FOLDER:
            files = files[:LIMIT_PER_FOLDER]

        print(f"  [{label[:4].upper()}] {folder}: {len(files)} images")

        for fname in files:
            img = cv2.imread(os.path.join(folder_path, fname))
            if img is None:
                continue
            feats = extract_features(img)
            if feats is None:
                continue
            feats['Label'] = label
            feats['Folder'] = folder
            records.append(feats)

    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("  Training from real ICML UAV images")
    print("=" * 60)

    print("\nScanning ICML folders …")
    df = load_dataset()

    if df.empty:
        print("ERROR: No data loaded. Check ICML_ROOT path.")
        sys.exit(1)

    print(f"\nLoaded {len(df)} images")
    print(df.groupby('Label').size().to_string())

    # Feature distributions (quick sanity check)
    print("\n=== Feature Means by Class ===")
    for feat in QML_FEATURES:
        for label in ['Pre_Defoliation', 'Post_Defoliation']:
            sub = df[df['Label'] == label][feat]
            print(f"  {label[:4]}  {feat}: {sub.mean():.5f} ± {sub.std():.5f}")

    # Encode labels
    X = df[QML_FEATURES].values
    y = (df['Label'] == 'Pre_Defoliation').astype(int).values   # 1=Pre, 0=Post

    # Build pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    SVC(kernel='rbf', C=10, gamma='scale',
                       probability=True, random_state=42))
    ])

    # 5-fold CV
    print("\nRunning 5-fold stratified CV …")
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs  = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    print(f"  CV Accuracy: {accs.mean():.4f} ± {accs.std():.4f}")

    # Train on full data
    print("\nFitting on full dataset …")
    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    print("\n=== Classification Report (train set) ===")
    print(classification_report(y, preds,
                                target_names=['Post_Defoliation', 'Pre_Defoliation']))

    print("=== Confusion Matrix ===")
    cm = confusion_matrix(y, preds)
    print(f"  Predicted Post  Pre")
    print(f"  True Post: {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"  True Pre:  {cm[1,0]:4d}  {cm[1,1]:4d}")

    # Quick test on known images
    print("\n=== Quick generalisation check (unseen images) ===")
    tests = [
        ('/Volumes/T9/ICML/Part_one_pre_def_rgb/DJI_20250929095743_0311_D.JPG', 'Pre_Defoliation'),
        ('/Volumes/T9/ICML/Post_def_rgb_part1/DJI_20250929124149_0029_D.JPG',  'Post_Defoliation'),
        ('/Volumes/T9/ICML/205_Post_Def_rgb/DJI_20250929124505_0127_D.JPG',     'Post_Defoliation'),
        ('/Volumes/T9/ICML/part 2_pre_def_rgb/DJI_20250929093936_0722_D.JPG',   'Pre_Defoliation'),
    ]
    for path, true_label in tests:
        img = cv2.imread(path)
        if img is None:
            print(f"  [SKIP] {os.path.basename(path)}")
            continue
        f = extract_features(img)
        x  = np.array([[f[k] for k in QML_FEATURES]])
        probs = pipeline.predict_proba(x)[0]
        pred  = 'Pre_Defoliation' if np.argmax(probs) == 1 else 'Post_Defoliation'
        ok    = '✅' if pred == true_label else '❌'
        print(f"  {ok} {os.path.basename(path)}")
        print(f"       True={true_label[:4]}  Pred={pred[:4]}  "
              f"Post={probs[0]*100:.1f}%  Pre={probs[1]*100:.1f}%")

    # Save
    joblib.dump(pipeline, OUT_PKL)
    print(f"\n✅ Model saved → {OUT_PKL}")


if __name__ == '__main__':
    main()
