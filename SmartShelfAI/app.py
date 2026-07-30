import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="SmartShelf AI | Out-of-Stock Prediction",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS (Professional look)
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.6rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5A6A7A;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODELS (cached)
# ============================================================
@st.cache_resource
def load_models():
    base_path = Path(r"C:\Users\AiK\SmartShelf AI\Models")
    
    models = {
        "Logistic Regression": joblib.load(base_path / "logistic_regression_model.pkl"),
        "XGBoost": joblib.load(base_path / "xgboost_model.pkl"),
        "Random Forest": joblib.load(base_path / "random_forest.pkl")
    }
    return models

try:
    models = load_models()
except Exception as e:
    st.error(f"❌ Could not load models. Please check the path.\n\nError: {e}")
    st.stop()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.image("https://img.icons8.com/color/96/box.png", width=80)
st.sidebar.title("SmartShelf AI")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📌 Business Problem", "🔮 Predictions", "📊 Model Metrics", "⚖️ Model Comparison"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.info("**SmartShelf AI**\nOut-of-Stock Risk Prediction System")

# ============================================================
# 1. HOME PAGE
# ============================================================
if page == "🏠 Home":
    st.markdown('<p class="main-header">📦 SmartShelf AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent Out-of-Stock Prediction System for Retail</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Model", "Random Forest", "F1 = 0.7471")
    with col2:
        st.metric("Highest ROC-AUC", "Random Forest", "0.9389")
    with col3:
        st.metric("Models Available", "3", "LR | XGB | RF")

    st.markdown("---")
    st.subheader("What does this system do?")
    st.write("""
    SmartShelf AI predicts the **probability of a product going out of stock** using machine learning.
    It helps store managers and supply chain teams take proactive actions before stockouts occur.
    """)

    st.subheader("Key Features")
    st.markdown("""
    - 🎯 **Real-time Predictions** – Instant risk assessment for any product
    - 📈 **Model Comparison** – Compare Logistic Regression, XGBoost & Random Forest
    - 📊 **Performance Metrics** – Accuracy, Precision, Recall, F1-Score, ROC-AUC
    - 🛡️ **Production Ready** – Models trained with proper preprocessing & SMOTE
    """)

# ============================================================
# 2. BUSINESS PROBLEM
# ============================================================
elif page == "📌 Business Problem":
    st.markdown('<p class="main-header">📌 Business Problem</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("The Challenge")
    st.write("""
    Retail stores frequently face **out-of-stock (OOS)** situations which lead to:
    - Lost sales and revenue
    - Poor customer experience
    - Brand damage
    - Inefficient inventory management
    """)

    st.subheader("Our Solution")
    st.write("""
    SmartShelf AI uses historical inventory, sales, supplier, and external data to predict 
    the **likelihood of a stockout** for any product at a given store snapshot.
    
    Store managers can then:
    1. Prioritize restocking for high-risk items
    2. Adjust reorder levels dynamically
    3. Reduce both stockouts and overstocking
    """)

    st.subheader("Impact")
    col1, col2, col3 = st.columns(3)
    col1.metric("Potential Sales Recovery", "8-15%")
    col2.metric("Inventory Efficiency", "↑ 20%")
    col3.metric("Customer Satisfaction", "↑ High")

# ============================================================
# 3. PREDICTIONS
# ============================================================
elif page == "🔮 Predictions":
    st.markdown('<p class="main-header">🔮 Out-of-Stock Prediction</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.info("Fill in the product & store details below to get a stockout risk prediction.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Product & Store Info**")
            category = st.selectbox("Category", [
                "Bakery", "Beverages", "Canned Goods", "Dairy", "Frozen",
                "Household", "Meat", "Personal Care", "Produce", "Snacks"
            ])
            supplier = st.selectbox("Supplier", [
                "Supplier_A1", "Supplier_A2", "Supplier_B1", "Supplier_B2",
                "Supplier_C1", "Supplier_C2", "Supplier_D1", "Supplier_D2",
                "Supplier_E1", "Supplier_E2", "Supplier_F1", "Supplier_F2",
                "Supplier_G1", "Supplier_G2", "Supplier_H1", "Supplier_H2",
                "Supplier_I1", "Supplier_I2", "Supplier_J1", "Supplier_J2",
                "Supplier_K1", "Supplier_K2", "Supplier_L1", "Supplier_L2",
                "Supplier_M1", "Supplier_M2", "Supplier_N1", "Supplier_N2",
                "Supplier_O1", "Supplier_O2", "Supplier_P1", "Supplier_P2",
                "Supplier_Q1", "Supplier_Q2", "Supplier_R1", "Supplier_R2",
                "Supplier_S1", "Supplier_S2", "Supplier_T1", "Supplier_U1",
                "Supplier_V1", "Supplier_W1", "Supplier_X1", "Supplier_Y1", "Supplier_Z1"
            ])
            region = st.selectbox("Region", ["Central", "East", "North", "South", "West"])
            store_size = st.selectbox("Store Size", ["Small", "Medium", "Large"])
            season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])
            weather = st.selectbox("Weather Impact", ["Low", "Medium", "High"])

        with col2:
            st.markdown("**Inventory & Sales**")
            current_stock = st.number_input("Current Stock", min_value=0, value=40)
            daily_sales = st.number_input("Daily Sales", min_value=0.1, value=3.5, step=0.1)
            reorder_level = st.number_input("Reorder Level", min_value=1, value=15)
            lead_time = st.number_input("Lead Time (Days)", min_value=1.0, value=3.0, step=0.5)
            shelf_capacity = st.number_input("Shelf Capacity", min_value=10.0, value=50.0)
            days_since_restock = st.number_input("Days Since Last Restock", min_value=0.0, value=3.0)

        with col3:
            st.markdown("**Pricing & Other**")
            unit_price = st.number_input("Unit Price ($)", min_value=0.5, value=4.5, step=0.1)
            discount = st.number_input("Discount %", min_value=0, max_value=50, value=5)
            promotion = st.selectbox("Promotion", ["No", "Yes"])
            holiday = st.selectbox("Holiday Week", ["No", "Yes"])
            is_perishable = st.selectbox("Is Perishable", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            demand_index = st.number_input("Customer Demand Index", min_value=20.0, value=65.0)
            stock_coverage = st.number_input("Stock Coverage Days", min_value=0.0, value=8.0)
            price_per_demand = st.number_input("Price per Demand", min_value=0.1, value=1.5)
            supplier_reliability = st.number_input("Supplier Lead Reliability", min_value=0.7, max_value=1.3, value=0.98)
            store_age = st.number_input("Store Age (Years)", min_value=1, value=15)
            audit_score = st.number_input("Last Audit Score", min_value=60.0, max_value=100.0, value=80.0)
            competitor_price = st.number_input("Competitor Price Index", min_value=0.5, value=3.5)
            random_noise = st.number_input("Random Noise A", value=50.0)

        model_choice = st.selectbox("Select Model for Prediction", list(models.keys()), index=2)
        submitted = st.form_submit_button("🚀 Predict Stockout Risk", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame([{
            "Category": category,
            "Supplier": supplier,
            "Region": region,
            "Store_Size": store_size,
            "Snapshot_Date": "7/22/2025",          # dummy date (model handles unknown)
            "Season": season,
            "Current_Stock": current_stock,
            "Daily_Sales": daily_sales,
            "Reorder_Level": reorder_level,
            "Lead_Time_Days": lead_time,
            "Unit_Price": unit_price,
            "Discount_Percent": discount,
            "Shelf_Capacity": shelf_capacity,
            "Promotion": promotion,
            "Holiday_Week": holiday,
            "Days_Since_Last_Restock": days_since_restock,
            "Customer_Demand_Index": demand_index,
            "Weather_Impact": weather,
            "Is_Perishable": is_perishable,
            "Stock_Coverage_Days": stock_coverage,
            "Price_per_Demand": price_per_demand,
            "Supplier_Lead_Reliability": supplier_reliability,
            "Store_Age_Years": store_age,
            "Last_Audit_Score": audit_score,
            "Competitor_Price_Index": competitor_price,
            "Random_Noise_A": random_noise
        }])

        model = models[model_choice]
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        st.markdown("---")
        st.subheader("Prediction Result")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if prediction == 1:
                st.error("🚨 **HIGH RISK** – Likely Out of Stock")
            else:
                st.success("✅ **LOW RISK** – Stock Available")
        with col_b:
            st.metric("Probability of Stockout", f"{probability[1]:.1%}")
        with col_c:
            st.metric("Probability of No Stockout", f"{probability[0]:.1%}")

        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability[1] * 100,
            title={'text': "Stockout Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#ef4444" if probability[1] > 0.5 else "#22c55e"},
                'steps': [
                    {'range': [0, 30], 'color': "#dcfce7"},
                    {'range': [30, 70], 'color': "#fef9c3"},
                    {'range': [70, 100], 'color': "#fee2e2"}
                ]
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 4. MODEL METRICS
# ============================================================
elif page == "📊 Model Metrics":
    st.markdown('<p class="main-header">📊 Model Performance Metrics</p>', unsafe_allow_html=True)
    st.markdown("---")

    metrics_df = pd.DataFrame({
        "Model": ["Logistic Regression", "XGBoost", "Random Forest"],
        "Accuracy": [0.7325, 0.9130, 0.9137],
        "Precision": [0.3825, 0.8552, 0.7952],
        "Recall": [0.7781, 0.6252, 0.7044],
        "F1-Score": [0.5129, 0.7223, 0.7471],
        "ROC-AUC": [0.8137, 0.9386, 0.9389]
    })

    st.dataframe(
        metrics_df.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1-Score": "{:.2%}",
            "ROC-AUC": "{:.4f}"
        }).highlight_max(axis=0, color="#bbf7d0"),
        use_container_width=True
    )

    st.markdown("### Detailed Insights")
    st.write("""
    - **Random Forest** achieves the best overall balance (highest F1 & ROC-AUC).
    - **XGBoost** has the highest Precision (fewer false alarms).
    - **Logistic Regression** has the highest Recall (catches more actual stockouts) but lower precision.
    """)

# ============================================================
# 5. MODEL COMPARISON
# ============================================================
elif page == "⚖️ Model Comparison":
    st.markdown('<p class="main-header">⚖️ Model Comparison</p>', unsafe_allow_html=True)
    st.markdown("---")

    metrics_df = pd.DataFrame({
        "Model": ["Logistic Regression", "XGBoost", "Random Forest"],
        "Accuracy": [0.7325, 0.9130, 0.9137],
        "Precision": [0.3825, 0.8552, 0.7952],
        "Recall": [0.7781, 0.6252, 0.7044],
        "F1-Score": [0.5129, 0.7223, 0.7471],
        "ROC-AUC": [0.8137, 0.9386, 0.9389]
    })

    # Bar chart comparison
    fig_bar = px.bar(
        metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
        x="Metric", y="Score", color="Model",
        barmode="group",
        title="Performance Metrics Comparison",
        color_discrete_sequence=["#3b82f6", "#f97316", "#22c55e"]
    )
    fig_bar.update_layout(yaxis_tickformat=".0%", height=450)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Radar chart
    categories = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    fig_radar = go.Figure()

    for _, row in metrics_df.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row[c] for c in categories],
            theta=categories,
            fill='toself',
            name=row["Model"]
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Radar Chart – Model Strengths",
        height=500
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.success("**Winner:** Random Forest is recommended for production use due to the best F1-Score and ROC-AUC.")