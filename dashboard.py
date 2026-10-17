# dashboard.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

MODEL_PATH = "model/fault_detection_model.pkl"

def extract_l2_features_array(arr90):
    arr = np.array(arr90, dtype=float).reshape(15, 6)
    l2_ts = np.linalg.norm(arr, axis=1)
    feats = {}
    feats['l2_mean'] = l2_ts.mean()
    feats['l2_std'] = l2_ts.std()
    feats['l2_max'] = l2_ts.max()
    feats['l2_min'] = l2_ts.min()
    feats['l2_median'] = np.median(l2_ts)
    x = np.arange(len(l2_ts)).reshape(-1, 1)
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(x, l2_ts)
    feats['l2_slope'] = float(lr.coef_[0])
    dif = np.diff(l2_ts)
    feats['l2_diff_mean'] = dif.mean() if dif.size else 0.0
    feats['l2_diff_max'] = dif.max() if dif.size else 0.0
    feats['l2_energy'] = float(np.sum(l2_ts ** 2))
    feats['raw_sum_abs'] = float(np.sum(np.abs(arr)))
    return feats, l2_ts

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}. Run training script first.")
        return None
    obj = joblib.load(path)
    # If old code saved a dict like {"model": model}
    if isinstance(obj, dict):
        # heuristics: pick first value that has predict()
        for v in obj.values():
            if hasattr(v, "predict"):
                return v
        st.warning("Loaded a dict but no value exposes .predict(); returning whole dict.")
        return obj
    return obj

st.set_page_config(layout="wide", page_title="Fault Detection - Dashboard")
st.title("Robotic System Fault Detection")

model = load_model(MODEL_PATH)
if model is None:
    st.stop()

st.sidebar.header("Input")
mode = st.sidebar.selectbox("Mode", ["Upload CSV sample", "Manual input (paste 90 values)"])
if mode == "Upload CSV sample":
    uploaded = st.sidebar.file_uploader("Upload a single-row CSV (90 cols) or merged file", type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded)
        # If merged file with many rows, let user pick row
        if df.shape[0] > 1:
            idx = st.sidebar.number_input("Row index", min_value=0, max_value=df.shape[0]-1, value=0)
            row = df.iloc[idx]
        else:
            row = df.iloc[0]
        # take first 90 numeric columns
        arr90 = row.iloc[:90].values
        feats_dict, l2_ts = extract_l2_features_array(arr90)
        feats_df = pd.DataFrame([feats_dict])
        prediction = model.predict(feats_df)[0]
        if hasattr(model, "predict_proba"):
            conf = model.predict_proba(feats_df).max()
        else:
            conf = None
    else:
        st.info("Upload a CSV to get predictions.")
        st.stop()
else:
    text = st.sidebar.text_area("Paste 90 numeric values separated by commas or spaces")
    if not text:
        st.info("Paste 90 numbers to analyze.")
        st.stop()
    vals = [float(x) for x in text.replace("\n", " ").split() if x.strip()]
    if len(vals) < 90:
        st.error(f"Need 90 numbers, got {len(vals)}")
        st.stop()
    arr90 = vals[:90]
    feats_dict, l2_ts = extract_l2_features_array(arr90)
    feats_df = pd.DataFrame([feats_dict])
    prediction = model.predict(feats_df)[0]
    if hasattr(model, "predict_proba"):
        conf = model.predict_proba(feats_df).max()
    else:
        conf = None

label_map = {0: "NORMAL", 1: "FAULT"}
st.header("Result")
if prediction == 0:
    st.success(f"Prediction: {label_map.get(prediction, prediction)}" + (f" — confidence {conf:.2f}" if conf is not None else ""))
else:
    # cinematic highlight for faults
    st.markdown("<div style='background:#be1616;padding:12px;border-radius:6px;color:white;font-weight:bold;font-size:22px;'>!!! FAULT DETECTED — ATTENTION REQUIRED !!!</div>", unsafe_allow_html=True)
    st.error(f"Prediction: {label_map.get(prediction, prediction)}" + (f" — confidence {conf:.2f}" if conf is not None else ""))

st.subheader("Temporal L2 (per timestep)")
fig, ax = plt.subplots(figsize=(8,2.5))
ax.plot(np.arange(1, len(l2_ts)+1), l2_ts, marker='o')
ax.set_xlabel("Timestep")
ax.set_ylabel("L2 magnitude")
ax.grid(True)
st.pyplot(fig)

st.subheader("Feature values used for prediction")
st.table(feats_df.T.rename(columns={0:"value"}))

st.write("Model file used:", MODEL_PATH)
