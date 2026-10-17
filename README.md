# Sensor-Based Fault Detection in Robotic Systems Using Temporal Features

B.E. final-year project, Artificial Intelligence & Machine Learning (VTU), SEA College of Engineering & Technology, Bengaluru, 2025–26.

The project detects robot execution failures from force–torque sensor data. Each six-channel sample is reduced to one L2-magnitude time series, ten temporal features are extracted from it, and a Random Forest classifies the sample as normal or fault. A Streamlit dashboard shows the prediction, confidence, the L2 curve and the feature values.

## How it works

1. **Data preparation** (`prepare_robotexec_files.py`): merges `lp1.csv` to `lp5.csv`, maps labels to binary (normal/ok = 0, all failure types = 1), removes duplicate samples, and stops with an error if identical samples have different labels.
2. **Feature extraction:** each sample has 15 time steps × 6 channels (Fx, Fy, Fz, Tx, Ty, Tz). The L2 norm per time step gives a 15-value series. Ten features are computed: mean, std, max, min, median, slope, mean and max step change, energy, and the sum of absolute raw values.
3. **Training** (`train_and_save.py`): StandardScaler + Random Forest (100 trees, seed 42). Prints a stratified hold-out result, repeated stratified 5-fold cross-validation and two baselines. The saved model is fitted on all unique samples.
4. **Dashboard** (`dashboard.py`): loads the saved model and predicts from either an uploaded CSV (a single row of 90 values, or a merged file where you pick the row index) or 90 values pasted manually. It shows the prediction, the confidence (highest class probability), the L2 curve and the ten feature values. For CSV input it reads the first 90 columns of the chosen row, so the 90 sensor values must come first.

## Data

[UCI Robot Execution Failures](https://archive.ics.uci.edu/dataset/138/robot+execution+failures) (Seabra Lopes and Camarinha-Matos, 1999). The five files overlap heavily: 463 rows contain only **251 unique samples (65 normal, 186 fault)**. Duplicates are removed before evaluation so that identical samples never appear in both training and test data.

## Results

Repeated stratified 5-fold cross-validation (5 repeats) on the 251 unique samples:

| Method | Accuracy | Balanced acc. | Macro-F1 |
|---|---|---|---|
| Majority-class guess | 0.741 | 0.500 | – |
| 10 engineered L2 features (proposed) | 0.946 | 0.930 | 0.930 |
| 90 raw values (baseline) | 0.963 | 0.956 | 0.952 |

On the stratified hold-out split (51 test samples) the model scores 94.1%. With only 13 normal samples in that split, one sample changes the result by about two percentage points, so the cross-validation figures are the more reliable ones.

The engineered features do **not** beat the raw signal. Their advantage is a compact, interpretable representation: 10 values instead of 90.

## Limitations

- One small public dataset (251 unique samples); samples from the same experimental session may still be correlated.
- All fault types are merged into one class, so the model says a fault occurred but not which kind.
- No experiments on physical hardware.
- The dashboard model is trained on all 251 samples, so rows from the dataset used in the dashboard demo are not held-out data.

## Run

Requires Python (see `.python-version`) and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Build the cleaned dataset from data/lp1.csv ... lp5.csv
uv run prepare_robotexec_files.py

# 2. Train, evaluate and save the model to model/fault_detection_model.pkl
uv run train_and_save.py

# 3. Launch the dashboard
uv run streamlit run dashboard.py
```

Without uv: `pip install -r requirements.txt`, then run the same scripts with `python`.

## Repository layout

```
data/        lp1.csv ... lp5.csv (UCI source files) and the merged dataset
model/       saved model (fault_detection_model.pkl)
prepare_robotexec_files.py   data cleaning and label mapping
train_and_save.py            feature extraction, evaluation, model saving
dashboard.py                 Streamlit app
```

## Tech stack

Python, NumPy, pandas, scikit-learn, joblib, Matplotlib, Streamlit.

## Author

Dilshad D