# ❤️ Heart Disease Prediction

A machine learning project that predicts the likelihood of heart disease based on clinical parameters. Trained on 918 patient records using multiple classifiers, with the best model deployed via a Streamlit web app.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [Models & Results](#models--results)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)

---

## Overview

This project builds an end-to-end binary classification pipeline to predict whether a patient has heart disease (`1`) or not (`0`). It includes exploratory data analysis, data cleaning, feature engineering, model training, evaluation, and deployment as an interactive web app.

---

## Dataset

**File:** `heart.csv`  
**Source:** [UCI Heart Disease Dataset (Kaggle variant)](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)  
**Rows:** 918 | **Columns:** 12

| Feature | Description |
|---|---|
| `Age` | Patient age in years |
| `Sex` | M / F |
| `ChestPainType` | ATA, NAP, TA, ASY |
| `RestingBP` | Resting blood pressure (mm Hg) |
| `Cholesterol` | Serum cholesterol (mg/dL) |
| `FastingBS` | Fasting blood sugar > 120 mg/dL (1 = Yes) |
| `RestingECG` | Normal / ST / LVH |
| `MaxHR` | Maximum heart rate achieved |
| `ExerciseAngina` | Exercise-induced angina (Y / N) |
| `Oldpeak` | ST depression induced by exercise |
| `ST_Slope` | Slope of peak exercise ST segment (Up / Flat / Down) |
| `HeartDisease` | Target — 0: No Disease, 1: Disease |

**Data Issues Fixed:**
- `Cholesterol` and `RestingBP` contained biologically impossible zero values → replaced with the mean of valid (non-zero) entries.

---

## Project Structure

```
heart-disease-prediction/
│
├── heart.csv               # Raw dataset
├── heart.ipynb             # Full ML pipeline notebook
├── app.py                  # Streamlit web app
│
├── Logistic_heart.pkl      # Saved Logistic Regression model
├── scalar.pkl              # Saved StandardScaler
├── columns.pkl             # Saved feature column names
│
└── README.md
```

---

## ML Pipeline

1. **Data Loading** — Read CSV, preview shape and types
2. **EDA** — Class balance check, distribution plots, categorical analysis, correlation heatmap
3. **Data Cleaning** — Replace zero values in `Cholesterol` and `RestingBP` with column means
4. **Preprocessing** — One-hot encoding of categorical features (`pd.get_dummies`, `drop_first=True`)
5. **Train/Test Split** — 80% train / 20% test (`random_state=42`)
6. **Feature Scaling** — `StandardScaler` fit on training data only; `.transform()` applied to test data (no data leakage)
7. **Model Training** — Five classifiers trained and compared
8. **Evaluation** — Accuracy, F1 Score, and full classification report
9. **Export** — Model, scaler, and feature columns saved with `joblib`

---

## Models & Results

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | Best | — |
| Support Vector Machine (RBF) | — | — |
| K-Nearest Neighbors | — | — |
| Naive Bayes | — | — |
| Decision Tree | — | — |

> Run `heart.ipynb` to see the exact scores. Logistic Regression was selected as the final model.

---

## Installation

**Prerequisites:** Python 3.8+

```bash
# Clone the repository
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

---

## Usage

### Run the Notebook

Open `heart.ipynb` in Jupyter or Google Colab to reproduce the full analysis and retrain the model.

### Run the Web App

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser. Enter patient details using the sliders and dropdowns, then click **Predict** to get a risk assessment.

![Streamlit App](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)

> **Note:** The `.pkl` files (`Logistic_heart.pkl`, `scalar.pkl`, `columns.pkl`) must be present in the same directory as `app.py`.

---

## Technologies Used

- **Python 3.10**
- **Pandas / NumPy** — Data manipulation
- **Matplotlib / Seaborn** — Visualization
- **Scikit-learn** — ML models and preprocessing
- **Joblib** — Model serialization
- **Streamlit** — Web app deployment

---

## Author

Made with ❤️ — feel free to fork and build on it!
