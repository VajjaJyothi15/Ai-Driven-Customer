import streamlit as st
import pandas as pd
import joblib

from database.db_operations import get_data
from utils.preprocessing import preprocess
from utils.feature_engineering import add_all_features
from utils.ui import (
    load_css,
    page_header,
    section_title,
    metric_row
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Customer 360",
    page_icon="👥",
    layout="wide"
)

load_css()

page_header(
    "👥 Customer 360°",
    "Complete Customer Profile, Value & Risk Intelligence"
)

# =====================================================
# LOAD DATA
# =====================================================

df = get_data()
df = preprocess(df)
df = add_all_features(df)

# =====================================================
# LOAD MODELS
# =====================================================

try:
    churn_model = joblib.load(
        "saved_models/churn_model.pkl"
    )

    scaler = joblib.load(
        "saved_models/scaler.pkl"
    )

    segment_model = joblib.load(
        "saved_models/segment_model.pkl"
    )

except Exception as e:

    st.error(
        f"Model loading error: {e}"
    )

    st.stop()

# =====================================================
# CUSTOMER SELECTOR
# =====================================================

section_title("Select Customer")

customer_id = st.selectbox(
    "Choose Customer ID",
    sorted(df["CustomerID"].unique())
)

customer = df[
    df["CustomerID"] == customer_id
].iloc[0]

# =====================================================
# SEGMENT PREDICTION
# =====================================================

segment_input = pd.DataFrame({
    "AnnualIncome": [customer["AnnualIncome"]],
    "SpendingScore": [customer["SpendingScore"]],
    "PurchaseFrequency": [customer["PurchaseFrequency"]]
})

scaled = scaler.transform(
    segment_input
)

segment = segment_model.predict(
    scaled
)[0]

segment_names = {
    0: "💎 Premium Customer",
    1: "⭐ Loyal Customer",
    2: "⚠ At-Risk Customer",
    3: "🆕 New Customer"
}

customer_segment = segment_names.get(
    segment,
    "Unknown"
)

# =====================================================
# CHURN PROBABILITY
# =====================================================

feature_columns = [
    "Age",
    "Gender",
    "AnnualIncome",
    "SpendingScore",
    "Tenure",
    "MonthlyCharges",
    "PurchaseFrequency",
    "SatisfactionScore",
    "CLV",
    "LoyaltyScore",
    "EngagementScore",
    "CustomerValue",
    "RiskScore",
    "CustomerHealthScore"
]

prediction_df = pd.DataFrame(
    [customer[feature_columns]]
)

churn_probability = (
    churn_model
    .predict_proba(prediction_df)[0][1]
)

# =====================================================
# CUSTOMER PROFILE
# =====================================================

section_title("Customer Profile")

col1, col2 = st.columns(2)

with col1:

    st.info(f"""
    Customer ID: {customer['CustomerID']}

    Age: {customer['Age']}

    Gender: {'Male' if customer['Gender']==1 else 'Female'}

    Annual Income:
    ₹{customer['AnnualIncome']:,.0f}
    """)

with col2:

    st.info(f"""
    Tenure: {customer['Tenure']} Years

    Monthly Charges:
    ₹{customer['MonthlyCharges']:,.0f}

    Spending Score:
    {customer['SpendingScore']}

    Purchase Frequency:
    {customer['PurchaseFrequency']}
    """)

# =====================================================
# CUSTOMER KPI
# =====================================================

section_title("Customer Metrics")

metric_row(
    [
        (
            "Customer Value",
            f"₹{customer['CustomerValue']:,.0f}"
        ),
        (
            "CLV",
            f"₹{customer['CLV']:,.0f}"
        ),
        (
            "Loyalty",
            round(
                customer["LoyaltyScore"],
                2
            )
        ),
        (
            "Engagement",
            round(
                customer["EngagementScore"],
                2
            )
        )
    ]
)

metric_row(
    [
        (
            "Risk Score",
            round(
                customer["RiskScore"],
                2
            )
        ),
        (
            "Health Score",
            round(
                customer["CustomerHealthScore"],
                2
            )
        ),
        (
            "Satisfaction",
            round(
                customer["SatisfactionScore"],
                2
            )
        ),
        (
            "Churn %",
            f"{churn_probability*100:.2f}"
        )
    ]
)

# =====================================================
# CUSTOMER CLASSIFICATION
# =====================================================

section_title("Customer Intelligence")

c1, c2, c3 = st.columns(3)

with c1:

    st.success(f"""
    Segment

    {customer_segment}
    """)

with c2:

    if churn_probability > 0.7:

        st.error("""
        Risk Level

        High
        """)

    elif churn_probability > 0.4:

        st.warning("""
        Risk Level

        Medium
        """)

    else:

        st.success("""
        Risk Level

        Low
        """)

with c3:

    if customer["CustomerValue"] > df["CustomerValue"].median():

        st.success("""
        Customer Category

        High Value
        """)

    else:

        st.info("""
        Customer Category

        Standard
        """)

# =====================================================
# AI RECOMMENDATIONS
# =====================================================

section_title("🤖 AI Recommendations")

if churn_probability > 0.7:

    recommendation = """
    • Immediate retention campaign

    • Personalized discount offers

    • Dedicated customer support

    • Loyalty incentives
    """

elif churn_probability > 0.4:

    recommendation = """
    • Improve engagement

    • Offer reward points

    • Product recommendations

    • Satisfaction surveys
    """

else:

    recommendation = """
    • Upsell premium products

    • Referral programs

    • VIP membership offers

    • Cross-sell opportunities
    """

st.success(recommendation)

# =====================================================
# CUSTOMER SUMMARY
# =====================================================

section_title("Executive Summary")

st.info(f"""
Customer ID: {customer['CustomerID']}

Segment:
{customer_segment}

Customer Value:
₹{customer['CustomerValue']:,.0f}

Lifetime Value:
₹{customer['CLV']:,.0f}

Churn Probability:
{churn_probability*100:.2f}%

Risk Score:
{customer['RiskScore']:.2f}

Recommendation:

{recommendation}
""")

# =====================================================
# EXPORT CUSTOMER REPORT
# =====================================================

section_title("Download Report")

report_df = pd.DataFrame({
    "Metric": [
        "Customer ID",
        "Segment",
        "Customer Value",
        "CLV",
        "Churn Probability",
        "Risk Score",
        "Health Score"
    ],
    "Value": [
        customer["CustomerID"],
        customer_segment,
        customer["CustomerValue"],
        customer["CLV"],
        round(churn_probability*100, 2),
        customer["RiskScore"],
        customer["CustomerHealthScore"]
    ]
})

csv = report_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "⬇ Download Customer Report",
    csv,
    f"customer_{customer_id}_report.csv",
    "text/csv"
)

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "AI-Driven Customer Analytics Platform | Customer 360°"
)