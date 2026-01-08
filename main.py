"""
HAB Prediction Model for Northern Bay of Bengal
Author: Md Abu Yousuf Prodhan
Date: 2024
Description: This script loads oceanographic time-series data, merges it, 
and trains a Machine Learning model (Random Forest) to predict Harmful Algal Bloom (HAB) risks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import os

# --- Configuration ---
DATA_DIR = 'data/'
MODEL_OUTPUT = 'hab_model.pkl'
PREDICTION_OUTPUT = 'hab_risk_predictions.csv'

def load_data():
    """Loads and preprocesses the CSV data files."""
    try:
        sst = pd.read_csv(os.path.join(DATA_DIR, 'sst.csv'))
        salinity = pd.read_csv(os.path.join(DATA_DIR, 'salinity.csv'))
        nitrate = pd.read_csv(os.path.join(DATA_DIR, 'nitrate.csv'))
        turbidity = pd.read_csv(os.path.join(DATA_DIR, 'turbidity.csv'))
        historical = pd.read_csv(os.path.join(DATA_DIR, 'hab_events.csv'))
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error loading data: {e}. Check your 'data/' folder.")

    # Convert dates
    for df in [sst, salinity, nitrate, turbidity, historical]:
        df.columns = df.columns.str.strip() # Clean column names
        df['date'] = pd.to_datetime(df['date'])

    # Merge on date (Time-Series merge)
    df = sst.merge(salinity, on='date', how='inner') \
            .merge(nitrate, on='date', how='inner') \
            .merge(turbidity, on='date', how='inner') \
            .merge(historical, on='date', how='inner')
    
    # Cleaning
    df = df.dropna().drop_duplicates(subset=['date'])
    return df

def train_model(X, y):
    """Trains a Random Forest classifier and performs Grid Search."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # Base Model
    clf = RandomForestClassifier(class_weight='balanced', random_state=42)
    
    # Hyperparameter Tuning
    print("Tuning hyperparameters...")
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    grid = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print(f"Best Parameters: {grid.best_params_}")
    
    # Evaluation
    best_model = grid.best_estimator_
    print("\n--- Model Evaluation (Test Set) ---")
    print(classification_report(y_test, best_model.predict(X_test)))
    
    return best_model, X.columns

def save_predictions(model, df, features):
    """Generates risk probabilities and saves to CSV."""
    df['HAB_Prob'] = model.predict_proba(df[features])[:, 1]
    
    def classify_risk(prob):
        if prob < 0.4: return 'Low'
        elif prob < 0.7: return 'Medium'
        else: return 'High'

    df['Risk_Level'] = df['HAB_Prob'].apply(classify_risk)
    df.to_csv(PREDICTION_OUTPUT, index=False)
    print(f"Predictions saved to {PREDICTION_OUTPUT}")

if __name__ == "__main__":
    # 1. Load Data
    data = load_data()
    
    # 2. Define Features
    feature_cols = ['sst', 'salinity', 'nitrate', 'turbidity']
    X = data[feature_cols]
    y = data['HAB_Risk']
    
    # 3. Train
    model, feature_names = train_model(X, y)
    
    # 4. Save Model
    joblib.dump(model, MODEL_OUTPUT)
    
    # 5. Run Predictions
    save_predictions(model, data, feature_cols)