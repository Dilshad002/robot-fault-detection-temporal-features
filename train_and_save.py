# train_and_save.py
import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             balanced_accuracy_score, f1_score)
from sklearn.linear_model import LinearRegression


def extract_l2_features_row(row_vals):
    """
    row_vals: iterable of length 90 representing:
      Fx1,Fy1,Fz1,Tx1,Ty1,Tz1, Fx2,...,Tz15
    returns: dict of features
    """
    arr = np.array(row_vals, dtype=float).reshape(15, 6)  # (timesteps, channels)
    l2_ts = np.linalg.norm(arr, axis=1)  # shape (15,)
    features = {}
    features['l2_mean'] = l2_ts.mean()
    features['l2_std'] = l2_ts.std()
    features['l2_max'] = l2_ts.max()
    features['l2_min'] = l2_ts.min()
    features['l2_median'] = np.median(l2_ts)
    x = np.arange(len(l2_ts)).reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(x, l2_ts)
    features['l2_slope'] = float(lr.coef_[0])
    dif = np.diff(l2_ts)
    features['l2_diff_mean'] = dif.mean() if dif.size else 0.0
    features['l2_diff_max'] = dif.max() if dif.size else 0.0
    features['l2_energy'] = float(np.sum(l2_ts ** 2))
    features['raw_sum_abs'] = float(np.sum(np.abs(arr)))
    return features


def build_feature_dataframe(df_90):
    rows = []
    for _, row in df_90.iterrows():
        rows.append(extract_l2_features_row(row.values[:90]))
    return pd.DataFrame(rows)


def make_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])


def cross_validate(X, y, name, repeats=5):
    """Repeated stratified 5-fold CV; returns mean/std of accuracy, balanced accuracy, macro-F1."""
    scores = []
    for seed in range(repeats):
        pred = np.zeros(len(y), dtype=int)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            pred[te] = make_pipeline().fit(X[tr], y[tr]).predict(X[te])
        scores.append((accuracy_score(y, pred), balanced_accuracy_score(y, pred),
                       f1_score(y, pred, average='macro')))
    a = np.array(scores)
    print(f"  {name:24s} acc {a[:,0].mean():.3f}+-{a[:,0].std():.3f} | "
          f"bal.acc {a[:,1].mean():.3f} | macro-F1 {a[:,2].mean():.3f}")


def main(args):
    os.makedirs("model", exist_ok=True)
    print("Loading merged dataset:", args.path)
    df = pd.read_csv(args.path)
    if 'label' not in df.columns:
        raise RuntimeError("Merged CSV must contain 'label' column")

    meas_cols = list(df.columns[:90])
    n_before = len(df)
    df = df.drop_duplicates(subset=meas_cols).reset_index(drop=True)   # safety net: no copies across folds
    if len(df) != n_before:
        print(f"Dropped {n_before - len(df)} duplicate samples ({n_before} -> {len(df)})")
    print("Class counts (0=normal, 1=fault):", np.bincount(df['label'].astype(int).values).tolist())

    X_raw = df[meas_cols]
    y = df['label'].astype(int).values

    print("Extracting features (this may take a little while)...")
    X_feats = build_feature_dataframe(X_raw)
    print("Feature shape:", X_feats.shape)

    # --- single hold-out split (same style as before, for the console figure) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_feats, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = make_pipeline()
    print("Training RandomForest classifier...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("\nHold-out test set (20%, seed 42) - one split only, treat with caution:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # --- more reliable estimate: repeated 5-fold CV, with a raw-value baseline ---
    print("\nRepeated stratified 5-fold CV (5 repeats), mean +- std over repeats:")
    cross_validate(X_feats.values, y, "10 engineered features")
    cross_validate(X_raw.values.astype(float), y, "90 raw values (baseline)")
    print(f"  majority-class guess     acc {max(np.bincount(y)) / len(y):.3f} | bal.acc 0.500")

    # --- final model for the dashboard: fit on ALL unique samples ---
    final = make_pipeline().fit(X_feats, y)
    model_path = os.path.join("model", "fault_detection_model.pkl")
    print("\nSaving final pipeline (trained on all unique samples) to", model_path)
    joblib.dump(final, model_path)
    print("Saved. You can now run the dashboard (streamlit) or load the model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/robotexec_merged.csv", help="merged dataset path")
    args = parser.parse_args()
    main(args)
