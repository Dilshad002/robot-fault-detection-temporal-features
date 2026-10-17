# SENSOR-BASED FAULT DETECTION IN ROBOTIC SYSTEM USING TEMPORAL FEATURES

## Project Overview
This project presents a sensor-based fault detection system for robotic manipulators
using temporal feature modeling. The proposed approach detects abnormal robot behavior
by analyzing time-dependent characteristics of sensor signals such as force and torque.

Instead of relying on instantaneous sensor readings, the system extracts temporal
features—including L2-norm–based magnitude features—from fixed-length time windows.
A Random Forest classifier is used to distinguish between normal and faulty operating
conditions with high accuracy and low computational overhead.

The project also includes an interactive dashboard for real-time fault visualization
and analysis, making the system suitable for both academic research and industrial
robot monitoring applications.

---

## Dataset Description

- lp1.csv – lp5.csv  
  Individual robot execution datasets representing different experimental runs.
  Each file contains raw force–torque sensor readings recorded over time.

- robotexec_merged.csv  
  Combined dataset created from all individual execution files.
  Used for feature extraction and model training.

- robotexec_train_60.csv  
  60% split of the merged dataset used for training the model.

- robotexec_test_40.csv  
  40% split of the merged dataset used for testing and validation.

Each data sample consists of 90 values:
15 time steps × 6 sensor channels (Fx, Fy, Fz, Tx, Ty, Tz).

---

## Methodology

1. Sensor data is collected from robotic subsystems.
2. Data is segmented into fixed-length temporal windows.
3. L2 norm is computed at each timestep to capture signal magnitude.
4. Temporal features are extracted, including:
   - Mean, standard deviation, min, max, median
   - Trend (slope)
   - First-difference statistics
   - Signal energy
5. A Random Forest classifier is trained on labeled feature data.
6. The trained model is deployed for real-time fault detection via a dashboard.

---

## How to Run the Project

### Install dependencies
pip install -r requirements.txt

### Train the model
python train_and_save.py --path data/robotexec_train_60.csv

The trained model will be saved to:
model/fault_detection_model.pkl

### Run the dashboard
streamlit run dashboard.py

---

## Dashboard Features

- CSV upload or row-wise selection from merged dataset
- Temporal L2 magnitude visualization
- Fault prediction with confidence score
- Feature-level interpretability
- Visual fault alerts for intuitive monitoring

---

## Model Details

- Feature Type: Temporal features using L2 norm
- Classifier: Random Forest
- Learning Mode: Supervised learning
- Output: Normal / Faulty classification
- Designed for scalability and real-time deployment

---

## Applications

- Industrial robotic arm monitoring
- Predictive maintenance systems
- Fault diagnosis in automation
- Academic research and demonstrations

---
