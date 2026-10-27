import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np

# Add project root and src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.predict import predict_price, get_available_locations, get_available_area_types, get_model_metrics

# Streamlit Page Config
st.set_page_config(
    page_title="Bangalore Real Estate Price Predictor",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern clean aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.8rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 16px;
        padding: 24px;
        color: white;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }
    .metric-title {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 6px;
    }
    .metric-price {
        font-size: 2.6rem;
        font-weight: 800;
        color: #38BDF8;
        line-height: 1.1;
        margin-bottom: 12px;
    }
    .stat-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.1);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.88rem;
        margin-right: 8px;
        margin-bottom: 6px;
    }
    .info-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1582407947304-fd86f028f716?w=600&auto=format&fit=crop&q=80", use_container_width=True)
    st.markdown("### 📊 Project Overview")
    st.markdown("""
    This intelligent prediction system uses Machine Learning trained on **10,000+ verified Bangalore housing listings**.
    
    **Features Considered:**
    - 📍 **1,000+ Localities** across Bengaluru
    - 📐 **Total Square Feet Area**
    - 🛏️ **BHK Configuration**
    - 🚿 **Number of Bathrooms**
    - 🌅 **Balconies**
    - 🏗️ **Area Type** (Super built-up, Built-up, Carpet)
    """)
    st.divider()
    st.info("💡 **Tip:** Real estate valuations in Bangalore depend heavily on locality infrastructure, tech corridor proximity, and carpet-to-sqft ratio.")

# Main title
st.markdown('<div class="main-header">🏙️ Bangalore Real Estate Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Accurate, machine-learning-powered property valuation engine</div>', unsafe_allow_html=True)

# Fetch available locations & area types
try:
    locations = get_available_locations()
    area_types = get_available_area_types()
    metrics_info = get_model_metrics()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    locations = ["Whitefield", "Indira Nagar", "Electronic City", "HSR Layout", "Rajaji Nagar"]
    area_types = ["Super built-up  Area", "Built-up  Area", "Carpet  Area"]
    metrics_info = {}

# Set smart defaults
default_loc_idx = 0
if "Whitefield" in locations:
    default_loc_idx = locations.index("Whitefield")

# User Input Form
col1, col2 = st.columns([1.1, 0.9], gap="large")

with col1:
    st.markdown("#### 1. Property Specifications")
    
    col_loc, col_area = st.columns([1.2, 1.0])
    with col_loc:
        selected_location = st.selectbox(
            "📍 Locality / Neighborhood",
            options=locations,
            index=default_loc_idx,
            help="Select or type the Bangalore neighborhood"
        )
    with col_area:
        selected_area_type = st.selectbox(
            "🏗️ Property Type",
            options=area_types,
            index=0 if "Super built-up  Area" in area_types else 0,
            help="Super built-up includes common areas; Built-up includes walls."
        )

    col_sqft, col_bhk = st.columns(2)
    with col_sqft:
        total_sqft = st.number_input(
            "📐 Total Area (Square Feet)",
            min_value=300,
            max_value=15000,
            value=1250,
            step=50,
            help="Enter property size in sqft"
        )
    with col_bhk:
        bhk = st.slider(
            "🛏️ Bedrooms (BHK)",
            min_value=1,
            max_value=8,
            value=2,
            step=1
        )

    col_bath, col_balc = st.columns(2)
    with col_bath:
        bath = st.slider(
            "🚿 Bathrooms",
            min_value=1,
            max_value=8,
            value=min(bhk, 4),
            step=1
        )
    with col_balc:
        balcony = st.slider(
            "🌅 Balconies",
            min_value=0,
            max_value=4,
            value=1,
            step=1
        )

    predict_btn = st.button("🔮 Calculate Valuation", type="primary", use_container_width=True)

# Prediction Calculation
try:
    result = predict_price(
        location=selected_location,
        total_sqft=total_sqft,
        bhk=bhk,
        bath=bath,
        balcony=balcony,
        area_type=selected_area_type
    )
except Exception as err:
    st.error(f"Prediction failed: {err}")
    result = None

with col2:
    st.markdown("#### 2. Valuation & Financial Summary")
    
    if result:
        price_lakhs = result["predicted_price_lakhs"]
        formatted_price = result["formatted_price"]
        price_sqft = result["price_per_sqft"]
        monthly_emi = result["estimated_monthly_emi"]
        
        # Upper and lower range (+- 7% confidence window)
        lower_bound = price_lakhs * 0.93
        upper_bound = price_lakhs * 1.07
        
        if lower_bound >= 100:
            range_str = f"₹ {lower_bound/100:.2f} Cr - ₹ {upper_bound/100:.2f} Cr"
        else:
            range_str = f"₹ {lower_bound:.2f} L - ₹ {upper_bound:.2f} L"

        # Valuation Card
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Estimated Market Value</div>
            <div class="metric-price">{formatted_price}</div>
            <div style="margin-bottom: 14px;">
                <span class="stat-pill">📍 {selected_location}</span>
                <span class="stat-pill">📐 ₹ {price_sqft:,.0f} / sqft</span>
                <span class="stat-pill">🛏️ {bhk} BHK ({total_sqft} sqft)</span>
            </div>
            <div style="border-top: 1px solid rgba(255,255,255,0.15); padding-top: 12px; font-size: 0.9rem; color: #CBD5E1;">
                <strong>Estimated Value Range:</strong> {range_str}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Financial Snapshot
        m1, m2 = st.columns(2)
        with m1:
            st.metric(label="Estimated EMI (20 yrs @ 8.5%)", value=f"₹ {monthly_emi:,.0f} / mo")
        with m2:
            st.metric(label="Down Payment (20%)", value=f"₹ {(result['predicted_price_inr'] * 0.2) / 100000:.2f} Lakhs")

# Detail Tabs
st.divider()
tab1, tab2, tab3 = st.tabs(["📊 Financial Breakdown", "📈 Locality & BHK Benchmarks", "🤖 Model Performance"])

with tab1:
    if result:
        price_inr = result["predicted_price_inr"]
        stamp_duty = price_inr * 0.056  # 5.6% stamp duty + registration in Karnataka
        down_payment = price_inr * 0.20
        loan_principal = price_inr * 0.80
        est_maintenance = total_sqft * 3.5  # Rs 3.5 / sqft monthly maintenance

        st.markdown("##### Complete Cost of Acquisition")
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        f_col1.metric("Base Property Price", formatted_price)
        f_col2.metric("Stamp Duty & Reg. (~5.6%)", f"₹ {stamp_duty/100000:.2f} Lakhs")
        f_col3.metric("Initial Down Payment (20%)", f"₹ {down_payment/100000:.2f} Lakhs")
        f_col4.metric("Est. Monthly Maintenance", f"₹ {est_maintenance:,.0f} / mo")

        # Loan breakdown chart
        chart_data = pd.DataFrame({
            "Component": ["Home Loan (80%)", "Down Payment (20%)", "Govt Duties (5.6%)"],
            "Amount (INR)": [loan_principal, down_payment, stamp_duty]
        })
        st.bar_chart(chart_data.set_index("Component"), use_container_width=True)

with tab2:
    st.markdown(f"##### Comparative BHK Valuations in {selected_location}")
    # Simulate variations for 1 to 4 BHK in this location
    sim_data = []
    base_sqft_map = {1: 650, 2: 1100, 3: 1600, 4: 2400}
    for b in [1, 2, 3, 4]:
        sq = base_sqft_map[b]
        b_res = predict_price(
            location=selected_location,
            total_sqft=sq,
            bhk=b,
            bath=min(b, 4),
            balcony=1,
            area_type=selected_area_type
        )
        sim_data.append({
            "Configuration": f"{b} BHK (~{sq} sqft)",
            "Predicted Price (Lakhs)": b_res["predicted_price_lakhs"],
            "Price / Sqft (₹)": b_res["price_per_sqft"]
        })
    sim_df = pd.DataFrame(sim_data)
    st.dataframe(sim_df, use_container_width=True, hide_index=True)

with tab3:
    st.markdown("##### Machine Learning Model Evaluation & Leaderboard")
    champ = metrics_info.get("champion_model", "Gradient Boosting Regressor")
    st.success(f"🏆 Current Champion Production Model: **{champ}**")
    
    comp_table = metrics_info.get("comparison_table", [])
    if comp_table:
        comp_df = pd.DataFrame(comp_table)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
    else:
        st.info("Train the model via `python src/train.py` or run `notebooks/03_model_training.ipynb` to view the latest evaluation metrics.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>"
    "Bangalore Real Estate Price Predictor • Built with Streamlit & Scikit-Learn • Data source: Bengaluru House Data"
    "</div>",
    unsafe_allow_html=True
)
