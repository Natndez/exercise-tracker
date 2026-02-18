# 🏋️ Exercise Classification from Wearable IMU Data

## Overview

This project explores whether wearable inertial measurement unit (IMU) data can be used to:

- Classify barbell exercises  
- Distinguish movement intensity (e.g., heavy vs. medium sets)  
- Detect repetition patterns  

Using multi-axis accelerometer and gyroscope data collected from multiple participants, I built a structured preprocessing pipeline and exploratory analysis workflow to prepare the data for machine learning modeling.

---

## Motivation

As someone deeply interested in both fitness and machine learning, I wanted to understand:

- How raw IMU signals represent barbell movement  
- Whether exercise patterns are visually separable  
- Whether movement intensity produces detectable signal differences  

The goal is to move from raw wearable sensor data to a trained classification and repetition-counting model.

---

## Dataset

The dataset consists of:

- ~350 raw CSV files  
- Multi-participant recordings  
- Exercises including squat, bench press, overhead press, deadlift, and row  
- Intensity labels (e.g., heavy, medium)  
- Accelerometer sampled at **12.5 Hz**  
- Gyroscope sampled at **25 Hz**

Raw files are parsed directly to extract:

- Participant ID  
- Exercise label  
- Intensity category  
- Set identifier  

---

## Project Structure
```
exercise-tracker/
├── data/
│   ├── raw/              # Original IMU CSV files
│   ├── interim/          # Resampled and cleaned dataset
│   ├── processed/
│   └── external/
│
├── docs/
├── models/
├── notebooks/
├── references/
├── reports/
│   └── figures/
│
├── src/
│   ├── data/
│   │   └── make_dataset.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train_model.py
│   │   └── predict_model.py
│   └── visualization/
│       └── visualize.py
│
├── environment.yml
├── requirements.txt
├── .gitignore
└── README.md
```


The structure follows a modular ML workflow:

- Data ingestion  
- Feature engineering  
- Model training  
- Visualization  
- Reproducible environments  

---

## Data Engineering Pipeline

The preprocessing workflow includes:

### 1. File Parsing

- Extract metadata directly from file naming conventions  
- Assign participant, label, category, and set  

### 2. Sensor Separation

- Load accelerometer and gyroscope files independently  
- Maintain set tracking  

### 3. Timestamp Handling

- Convert epoch (ms) to datetime index  
- Remove redundant time columns  

### 4. Multi-Frequency Alignment

- Merge accelerometer (12.5 Hz) and gyroscope (25 Hz) streams  
- Resample to **200ms intervals**  
- Aggregate using mean for numeric features  
- Preserve metadata using last observation  

### 5. Clean & Export

- Remove unnecessary columns  
- Normalize set type  
- Persist processed dataset for downstream modeling  

This pipeline transforms raw sensor logs into a structured time-series dataset suitable for feature extraction and machine learning.

---

## Exploratory Analysis

Initial visualization shows meaningful signal differences even before modeling:

- Vertical acceleration patterns vary between exercise types  
- Heavy and medium squat sets produce distinguishable signal amplitude patterns  
- Participant-specific movement signatures are observable  

Multi-axis acceleration and angular velocity plots reveal consistent cyclical structure corresponding to repetitions.

These observations support the feasibility of supervised classification.

---

## Next Steps

- Outlier removal
- Feature engineering (magnitude, rolling statistics, frequency-domain features)  
- Repetition detection  
- Supervised classification (exercise + intensity)  
- Cross-participant validation  
- Generalization analysis  

---

## Tech Stack

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- SciPy
- Scikit-learn  
- Conda environment management  

---

## Status

Data preprocessing and exploratory visualization complete.  
Feature engineering and model training in progress.