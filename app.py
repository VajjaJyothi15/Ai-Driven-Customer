import streamlit as st

from database.db_operations import get_data
from utils.preprocessing import preprocess
from utils.feature_engineering import add_all_features
from utils.ui import (
    load_css,
    page_header,
    section_title,
    metric_row
)

st.set_page_config(
    page_title="AI Customer Analytics",
    page_icon="📊",
    layout="wide"
)

load_css()

# ==========================
# Load Data
# ==========================

df = get_data()
df = preprocess(df)
df = add_all_features(df)

# ==========================
# Header
# ==========================

page_header(
    "📊 AI-Driven Customer Analytics Platform",
    "Customer Intelligence • Segmentation • Churn Prediction • Revenue Analytics"
)

st.divider()

# ==========================
# KPI Preview
# ==========================

section_title("Business Overview")

total_customers = len(df)

total_revenue = round(
    df["MonthlyCharges"].sum(),
    0
)

avg_clv = round(
    df["CLV"].mean(),
    0
)

churn_rate = round(
    (
        df["Churn"].sum()
        /
        len(df)
    ) * 100,
    2
)

metric_row(
    [
        (
            "Customers",
            f"{total_customers:,}"
        ),
        (
            "Revenue",
            f"₹{total_revenue:,.0f}"
        ),
        (
            "Avg CLV",
            f"₹{avg_clv:,.0f}"
        ),
        (
            "Churn %",
            churn_rate
        )
    ]
)

st.divider()

# ==========================
# Platform Features
# ==========================

section_title("Platform Modules")

c1, c2, c3 = st.columns(3)

with c1:

    st.info("""
    ### 📊 Executive Dashboard

    Business KPIs

    Revenue Insights

    Customer Analytics
    """)

    st.info("""
    ### 👥 Customer 360°

    Customer Profile

    Loyalty Analysis

    Risk Analysis
    """)

with c2:

    st.success("""
    ### 🎯 Segmentation

    K-Means Clustering

    Customer Personas

    Segment Discovery
    """)

    st.success("""
    ### ⚠ Churn Intelligence

    Risk Detection

    Retention Strategy

    Churn Prediction
    """)

with c3:

    st.warning("""
    ### 💰 Revenue Intelligence

    CLV Analytics

    Revenue Forecast

    Top Customers
    """)

    st.warning("""
    ### 🤖 AI Recommendation Center

    Executive Insights

    Growth Opportunities

    Action Plans
    """)

st.divider()

# ==========================
# Quick Stats
# ==========================

section_title("Customer Insights")

avg_income = round(
    df["AnnualIncome"].mean(),
    0
)

avg_spending = round(
    df["SpendingScore"].mean(),
    2
)

avg_satisfaction = round(
    df["SatisfactionScore"].mean(),
    2
)

metric_row(
    [
        (
            "Avg Income",
            f"₹{avg_income:,.0f}"
        ),
        (
            "Avg Spending",
            avg_spending
        ),
        (
            "Avg Satisfaction",
            avg_satisfaction
        )
    ]
)

st.divider()

st.markdown("""
### 🚀 Navigation

Use the pages menu on the left to explore:

- Executive Dashboard
- Customer 360°
- Customer Segmentation
- Churn Intelligence
- Revenue Intelligence
- AI Recommendation Center
- Prediction Center
""")

st.markdown(
    """
    <div class='footer'>
    AI-Driven Customer Analytics Platform
    </div>
    """,
    unsafe_allow_html=True
)