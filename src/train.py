"""
End-to-end Model Training and Export Pipeline
Reads cleaned data, trains candidate models, evaluates metrics, and exports the champion model.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint


def train_and_export_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed", "house_prices_cleaned.csv")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Feature Engineering & Filtering
    df["bhk"] = df["size"].str.extract(r"(\d+)").astype(float)
    df["sqft_per_bhk"] = df["total_sqft_num"] / df["bhk"]

    df_model = df[df["sqft_per_bhk"] >= 300].copy()
    df_model["area_type"] = df_model["area_type"].str.strip()
    df_model = df_model[~df_model["area_type"].isin(["Plot  Area", "Plot Area"])].copy()
    df_model = df_model[df_model["total_sqft_num"] <= 20000].copy()

    features = ["area_type", "location", "bhk", "bath", "total_sqft_num", "balcony"]
    categorical_features = ["area_type", "location"]
    numerical_features = ["bhk", "bath", "total_sqft_num", "balcony"]

    X = df_model[features]
    y = df_model["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numerical", "passthrough", numerical_features)
        ]
    )

    print("Training candidate models...")
    # Gradient Boosting
    gb_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.08, max_depth=5, subsample=0.85, random_state=42
        ))
    ])
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_r2 = r2_score(y_test, gb_pred)
    gb_mae = mean_absolute_error(y_test, gb_pred)

    print(f"Gradient Boosting -> R2: {gb_r2:.4f}, MAE: {gb_mae:.2f}")

    # Random Forest
    rf_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1))
    ])
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_mae = mean_absolute_error(y_test, rf_pred)

    print(f"Random Forest -> R2: {rf_r2:.4f}, MAE: {rf_mae:.2f}")

    # Champion Selection
    if gb_r2 >= rf_r2:
        champ_name = "Gradient Boosting Regressor"
        champ_model = gb_model
        champ_r2 = gb_r2
        champ_mae = gb_mae
    else:
        champ_name = "Random Forest Regressor"
        champ_model = rf_model
        champ_r2 = rf_r2
        champ_mae = rf_mae

    model_path = os.path.join(models_dir, "bangalore_house_price_model.joblib")
    joblib.dump(champ_model, model_path)
    print(f"Exported champion model ({champ_name}) to {model_path}")

    # Metadata
    unique_locations = sorted([str(loc) for loc in df_model["location"].dropna().unique()])
    unique_areas = sorted([str(area) for area in df_model["area_type"].dropna().unique()])

    metadata = {
        "champion_model": champ_name,
        "features": features,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "locations": unique_locations,
        "area_types": unique_areas,
        "metrics": {"r2": round(champ_r2, 4), "mae": round(champ_mae, 2)}
    }
    meta_path = os.path.join(models_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Exported metadata to {meta_path}")


if __name__ == "__main__":
    train_and_export_model()
