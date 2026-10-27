"""
Real Estate Price Prediction Service
Loads trained model pipeline and metadata to produce predictions.
"""

import os
import json
import joblib
import pandas as pd
from typing import Dict, Any, Union

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "bangalore_house_price_model.joblib")
META_PATH = os.path.join(BASE_DIR, "models", "metadata.json")

_model = None
_metadata = None


def load_artifacts():
    """Load model and metadata with lazy initialization and caching."""
    global _model, _metadata
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train model first.")
        _model = joblib.load(MODEL_PATH)
    
    if _metadata is None:
        if not os.path.exists(META_PATH):
            raise FileNotFoundError(f"Metadata file not found at {META_PATH}.")
        with open(META_PATH, "r", encoding="utf-8") as f:
            _metadata = json.load(f)
            
    return _model, _metadata


def get_available_locations() -> list:
    """Return sorted list of all known Bangalore locations."""
    _, meta = load_artifacts()
    return meta.get("locations", [])


def get_available_area_types() -> list:
    """Return sorted list of valid property area types."""
    _, meta = load_artifacts()
    return meta.get("area_types", [])


def get_model_metrics() -> dict:
    """Return champion model performance metrics and comparison table."""
    _, meta = load_artifacts()
    return {
        "champion_model": meta.get("champion_model", "Unknown"),
        "metrics": meta.get("metrics", {}),
        "comparison_table": meta.get("comparison_table", [])
    }


def predict_price(
    location: str,
    total_sqft: float,
    bhk: int,
    bath: int,
    balcony: int = 1,
    area_type: str = "Super built-up  Area"
) -> Dict[str, Any]:
    """
    Predict real estate price for given property specifications.

    Parameters:
    -----------
    location : str
        Name of Bangalore locality (e.g., 'Whitefield', 'Indira Nagar')
    total_sqft : float
        Total built-up/carpet area in square feet
    bhk : int
        Number of bedrooms
    bath : int
        Number of bathrooms
    balcony : int
        Number of balconies (default: 1)
    area_type : str
        Property area category (default: 'Super built-up  Area')

    Returns:
    --------
    dict containing predicted price in Lakhs, formatted INR string, price/sqft, and EMI estimation.
    """
    model, meta = load_artifacts()

    # Input validation & sanitization
    total_sqft = max(100.0, float(total_sqft))
    bhk = max(1, int(bhk))
    bath = max(1, int(bath))
    balcony = max(0, int(balcony))
    
    input_df = pd.DataFrame([{
        "area_type": str(area_type).strip(),
        "location": str(location).strip(),
        "bhk": float(bhk),
        "bath": float(bath),
        "total_sqft_num": float(total_sqft),
        "balcony": float(balcony)
    }])

    # Predict price in Lakhs
    pred_lakhs = float(model.predict(input_df)[0])
    # Guard against theoretical negative values in linear edge cases
    pred_lakhs = max(5.0, pred_lakhs)

    # Price in INR
    pred_inr = pred_lakhs * 100_000
    price_per_sqft = pred_inr / total_sqft

    # Human readable formatting
    if pred_lakhs >= 100.0:
        crores = pred_lakhs / 100.0
        formatted_price = f"₹ {crores:.2f} Crore"
    else:
        formatted_price = f"₹ {pred_lakhs:.2f} Lakh"

    # Estimated Monthly EMI (20-year home loan at 8.5% interest, 80% LTV)
    loan_amount = pred_inr * 0.80
    annual_rate = 0.085
    monthly_rate = annual_rate / 12
    tenure_months = 20 * 12
    monthly_emi = (loan_amount * monthly_rate * ((1 + monthly_rate) ** tenure_months)) / (((1 + monthly_rate) ** tenure_months) - 1)

    return {
        "predicted_price_lakhs": round(pred_lakhs, 2),
        "predicted_price_inr": round(pred_inr, 2),
        "formatted_price": formatted_price,
        "price_per_sqft": round(price_per_sqft, 0),
        "estimated_monthly_emi": round(monthly_emi, 0),
        "inputs": {
            "location": location,
            "total_sqft": total_sqft,
            "bhk": bhk,
            "bath": bath,
            "balcony": balcony,
            "area_type": area_type
        }
    }


if __name__ == "__main__":
    # Test example prediction
    sample_res = predict_price("Whitefield", 1250, 2, 2, 1)
    print("Sample Inference Result:")
    print(json.dumps(sample_res, indent=2))
