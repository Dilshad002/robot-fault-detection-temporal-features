import os, glob
import pandas as pd
import numpy as np

DATA_DIR = "data"
OUT_FILE = os.path.join(DATA_DIR, "robotexec_merged.csv")
BINARY = True   # set False to preserve multiclass (distinct integers per text class)

# Class names that mean "no fault". LP3 calls its normal class "ok"; the other files use "normal".
NORMAL_NAMES = {"normal", "ok"}


def detect_label_column(df):
    for name in ('label', 'target', 'y', 'class'):
        if name in df.columns:
            return name
    return None


def sanitize_row_block(df):
    # Return only numeric measurement columns (prefer first 90 numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 90:
        return df[numeric_cols[:90]]
    if df.shape[1] >= 90:
        return df.iloc[:, :90]
    return None


def map_text_label_series(s, binary=True):
    s_lower = s.astype(str).str.strip().str.lower()
    if binary:
        return s_lower.apply(lambda x: 0 if x in NORMAL_NAMES else 1).astype(int)
    uniques = sorted(s_lower.unique())
    mapping = {u: i for i, u in enumerate(uniques)}
    return s_lower.map(mapping).astype(int), mapping


def process_file(path, binary=True):
    basename = os.path.basename(path)
    print(f"Processing {basename} ...")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  ERROR reading {basename}: {e}")
        return None, None

    label_col = detect_label_column(df)
    measurements = sanitize_row_block(df)

    if measurements is None:
        if df.shape[0] == 1 and df.shape[1] >= 90:
            measurements = pd.DataFrame([df.iloc[0, :90].values])
        else:
            print(f"  WARNING: Could not find 90 measurement columns in {basename} (found {df.shape[1]} cols). Skipping.")
            return None, None

    if label_col is None:
        print(f"  No label/target/class column in {basename}. Attempting filename heuristic...")
        lower = basename.lower()
        if 'fault' in lower or 'fail' in lower or 'collision' in lower or 'obstruction' in lower:
            label_series = pd.Series([1] * len(measurements), name='label')
        else:
            label_series = pd.Series([0] * len(measurements), name='label')
        return measurements, label_series

    if label_col == 'class':
        txt_series = df['class']
        # Show exactly how each text class is mapped, so mislabelling is visible.
        print("  class counts:", txt_series.value_counts().to_dict())
        if binary:
            label_series = map_text_label_series(txt_series, binary=True)
            label_series.name = 'label'
            print("  -> label 0 (normal):", int((label_series == 0).sum()),
                  "| label 1 (fault):", int((label_series == 1).sum()))
            return measurements, label_series
        label_series, mapping = map_text_label_series(txt_series, binary=False)
        label_series.name = 'label'
        print(f"  Multiclass mapping for {basename}: {mapping}")
        return measurements, label_series

    lab = df[label_col]
    try:
        lab_num = pd.to_numeric(lab).astype(int)
        lab_num.name = 'label'
        return measurements, lab_num
    except Exception:
        lab_series = lab.astype(str)
        if binary:
            lab_num = map_text_label_series(lab_series, binary=True)
            lab_num.name = 'label'
            return measurements, lab_num
        lab_num, mapping = map_text_label_series(lab_series, binary=False)
        lab_num.name = 'label'
        print(f"  Multiclass mapping: {mapping}")
        return measurements, lab_num


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    # Only the five original LP files. A bare "*.csv" would also pick up the merged
    # output and the train/test splits from earlier runs and silently duplicate the data.
    files = sorted(glob.glob(os.path.join(DATA_DIR, "lp*.csv")))
    if not files:
        print(f"No lp*.csv files found in {DATA_DIR}. Place lp1.csv ... lp5.csv there, then run again.")
        return

    rows, skipped = [], []
    for f in files:
        meas, labels = process_file(f, binary=BINARY)
        if meas is None or labels is None:
            skipped.append(os.path.basename(f))
            continue
        if len(meas) != len(labels):
            print(f"  WARNING: row mismatch in {os.path.basename(f)}: truncating to min.")
            nmin = min(len(meas), len(labels))
            meas, labels = meas.iloc[:nmin], labels.iloc[:nmin]

        meas = meas.iloc[:, :90].copy()
        meas.columns = [f'col{i+1}' for i in range(90)]
        tmp = meas.reset_index(drop=True)
        tmp['label'] = labels.reset_index(drop=True)
        tmp['source_file'] = os.path.basename(f)
        rows.append(tmp)

    if not rows:
        print("No files successfully processed. Check logs above.")
        return

    merged = pd.concat(rows, ignore_index=True)
    print(f"\nBefore de-duplication: {merged.shape[0]} rows "
          f"(normal={int((merged.label == 0).sum())}, fault={int((merged.label == 1).sum())})")

    # The LP files overlap (e.g. LP4 is contained in LP5), so identical 90-value samples
    # appear several times. Keep one copy; otherwise copies land in both train and test.
    meas_cols = [f'col{i+1}' for i in range(90)]
    conflict = merged.groupby(meas_cols).label.nunique()
    if (conflict > 1).any():
        raise RuntimeError(f"{int((conflict > 1).sum())} identical samples carry conflicting labels - check label mapping.")
    merged = merged.drop_duplicates(subset=meas_cols, keep='first').reset_index(drop=True)
    print(f"After de-duplication:  {merged.shape[0]} rows "
          f"(normal={int((merged.label == 0).sum())}, fault={int((merged.label == 1).sum())})")

    merged = merged[meas_cols + ['label']]   # same layout as before: col1..col90, label
    merged.to_csv(OUT_FILE, index=False)
    print(f"Saved {OUT_FILE}. Skipped files: {skipped}")


if __name__ == "__main__":
    main()
