# 🏙️ Bengaluru Real Estate Price Prediction & Valuation Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)]()

An end-to-end Machine Learning system that predicts housing prices across **1,000+ localities in Bengaluru, India**. The solution encompasses complete exploratory data analysis, data cleaning, outlier elimination, multi-model benchmarking (Linear Regression, Decision Tree, Random Forest, Gradient Boosting), and an interactive **Streamlit Web Application** for real-time valuations and financial analytics.

---

## 📌 Project Architecture

```
Real-Estate-Price-Prediction/
│
├── app/
│   └── app.py                      # Interactive Streamlit Web Application
│
├── data/
│   ├── raw/
│   │   └── Bengaluru_House_Data.csv      # Raw Kaggle dataset (13,000+ rows)
│   └── processed/
│       ├── house_prices_cleaned.csv      # Cleaned and parsed dataset
│       └── house_prices_model_ready.csv  # Outlier-filtered model-ready dataset
│
├── models/
│   ├── bangalore_house_price_model.joblib # Serialized champion model pipeline
│   └── metadata.json                     # Feature lists, localities & metrics
│
├── notebooks/
│   ├── 01_real_estate_eda.ipynb          # Data cleaning, null imputation, sqft parser
│   ├── 02_exploratory_data_analysis.ipynb# Visual EDA, distribution & domain rules
│   └── 03_model_training.ipynb           # Model training, hyperparameter tuning & evaluation
│
├── src/
│   ├── __init__.py
│   ├── predict.py                        # Reusable inference engine & EMI calculator
│   └── train.py                          # Headless model retraining pipeline script
│
├── requirements.txt                      # Project dependencies
└── README.md                             # Documentation
```

---

## 🏆 Model Performance Benchmark

Models were trained and evaluated on an 80/20 train-test split:

| Model | Train $R^2$ | Test $R^2$ | Test MAE (₹ Lakhs) | Test RMSE (₹ Lakhs) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting Regressor** | **0.865** | **0.739** | **₹ 24.09 L** | **₹ 66.12 L** | 🥇 **Champion** |
| **Tuned Random Forest** | 0.884 | 0.691 | ₹ 24.52 L | ₹ 71.95 L | 🥈 Runner-up |
| **Random Forest (Default)** | 0.898 | 0.689 | ₹ 24.60 L | ₹ 72.22 L | Benchmark |
| **Decision Tree ($depth=10$)** | 0.771 | 0.626 | ₹ 27.04 L | ₹ 79.20 L | Baseline |
| **Linear Regression** | 0.648 | 0.586 | ₹ 28.39 L | ₹ 83.35 L | Baseline |
| **Ridge Regression** | 0.645 | 0.585 | ₹ 28.41 L | ₹ 83.42 L | Regularized |

> **Champion Model:** `Gradient Boosting Regressor` achieved the highest generalization score ($R^2 = 0.739$) and lowest prediction error ($MAE = 24.09$ Lakhs) without severe overfitting.

---

## 🚀 Key Features of the Streamlit App

1. **Instant Property Valuation:**
   - Input **Locality** (searchable among 1,000+ Bangalore neighborhoods), **BHK**, **Square Feet**, **Bathrooms**, **Balconies**, and **Area Type**.
   - Outputs predicted valuation formatted in Indian Rupees (**₹ Lakhs** or **₹ Crores**).
   - Generates confidence price range and price-per-square-foot metric.

2. **Financial Affordability Snapshot:**
   - **20-Year Home Loan EMI calculation** (at standard 8.5% p.a., 80% LTV).
   - Down payment breakdown (20%).
   - Karnataka Stamp Duty & Registration estimation (~5.6%).
   - Estimated monthly society maintenance.

3. **Locality Benchmarking:**
   - Live comparative table showing 1 BHK, 2 BHK, 3 BHK, and 4 BHK estimated market values in the selected neighborhood.

---

## 🛠️ Quickstart & Usage

### 1. Setup Virtual Environment
```bash
# Clone or navigate to repository
cd Real-Estate-Price-Prediction

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web App
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

### 3. Run Inference from Python Script
```python
from src.predict import predict_price

valuation = predict_price(
    location="Whitefield",
    total_sqft=1250,
    bhk=2,
    bath=2,
    balcony=1,
    area_type="Super built-up  Area"
)

print(valuation["formatted_price"])   # e.g. "₹ 63.36 Lakh"
print(valuation["price_per_sqft"])    # e.g. 5069.0
print(valuation["estimated_monthly_emi"]) # e.g. 43987.0
```

### 4. Retrain Model Headless
```bash
python src/train.py
```

---

## 📈 Data Cleaning & Engineering Highlights

- **Irregular Sqft Conversions:** Parsed ranged values (e.g. `2100 - 2850` converted to average `2475.0`) and converted uncommon units (`Sq. Meter`, `Acres`, `Guntha`) to standard square feet.
- **BHK Feature Extraction:** Regular expression extraction from diverse text strings (`2 BHK`, `4 Bedroom`).
- **Domain-Specific Outlier Removal:**
  - Removed properties with $< 300$ square feet per BHK (physically unrealistic luxury/apartment ratio).
  - Filtered extreme outliers ($> 20,000$ sqft) and standalone Plot Areas.
  - Grouped categorical variables and encoded categorical location data using scikit-learn `OneHotEncoder(handle_unknown='ignore')`.

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
