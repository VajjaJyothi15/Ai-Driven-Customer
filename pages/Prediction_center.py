import streamlit as st
import pandas as pd
import joblib

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
    page_title="Prediction Center",
    page_icon="🔮",
    layout="wide"
)

load_css()

page_header(
    "🔮 Prediction Center",
    "AI Customer Intelligence Simulator"
)

# =====================================================
# LOAD MODELS
# =====================================================

try:

    churn_model = joblib.load(
        "saved_models/churn_model.pkl"
    )

    segment_model = joblib.load(
        "saved_models/segment_model.pkl"
    )

    scaler = joblib.load(
        "saved_models/scaler.pkl"
    )

except Exception as e:

    st.error(
        f"Model loading error: {e}"
    )

    st.stop()

# =====================================================
# INPUT SECTION
# =====================================================

section_title(
    "Customer Information"
)

col1, col2 = st.columns(2)

with col1:

    age = st.selectbox(
        "Age",
        list(range(18, 81))
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )

    income = st.number_input(
        "Annual Income (₹)",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    spending_score = st.selectbox(
        "Spending Score",
        list(range(1, 101))
    )

with col2:

    tenure = st.selectbox(
        "Tenure (Years)",
        list(range(1, 31))
    )

    monthly_charges = st.number_input(
        "Monthly Charges (₹)",
        min_value=0.0,
        value=3000.0,
        step=100.0
    )

    purchase_frequency = st.selectbox(
        "Purchase Frequency",
        list(range(1, 51))
    )

    satisfaction_score = st.selectbox(
        "Satisfaction Score",
        list(range(1, 11))
    )

# =====================================================
# PREDICT BUTTON
# =====================================================

if st.button(
    "🚀 Generate Customer Intelligence Report",
    use_container_width=True
):

    gender_encoded = (
        1 if gender == "Male"
        else 0
    )

    # =================================================
    # FEATURE ENGINEERING
    # =================================================

    clv = (
        monthly_charges
        * tenure
    )

    loyalty_score = (
        tenure * 0.5
        +
        purchase_frequency * 0.3
        +
        satisfaction_score * 0.2
    )

    engagement_score = (
        purchase_frequency * 0.6
        +
        spending_score * 0.4
    )

    customer_value = (
        income * 0.3
        +
        spending_score * 0.3
        +
        clv * 0.4
    )

    risk_score = (
        (100 - satisfaction_score * 10)
        +
        (100 - spending_score)
    )

    health_score = (
        200 - risk_score
    )

    # =================================================
    # SEGMENT PREDICTION
    # =================================================

    segment_input = pd.DataFrame({
        "AnnualIncome": [income],
        "SpendingScore": [spending_score],
        "PurchaseFrequency": [purchase_frequency]
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

    persona = segment_names.get(
        segment,
        "Unknown"
    )

    # =================================================
    # CHURN PREDICTION
    # =================================================

    churn_input = pd.DataFrame({
        "Age": [age],
        "Gender": [gender_encoded],
        "AnnualIncome": [income],
        "SpendingScore": [spending_score],
        "Tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "PurchaseFrequency": [purchase_frequency],
        "SatisfactionScore": [satisfaction_score],
        "CLV": [clv],
        "LoyaltyScore": [loyalty_score],
        "EngagementScore": [engagement_score],
        "CustomerValue": [customer_value],
        "RiskScore": [risk_score],
        "CustomerHealthScore": [health_score]
    })

    churn_probability = (
        churn_model
        .predict_proba(churn_input)[0][1]
    )

    # =================================================
    # RISK CLASSIFICATION
    # =================================================

    if churn_probability >= 0.70:

        risk_level = "🔴 High Risk"

    elif churn_probability >= 0.40:

        risk_level = "🟡 Medium Risk"

    else:

        risk_level = "🟢 Low Risk"

    # =================================================
    # CUSTOMER CATEGORY
    # =================================================

    if customer_value > 100000:

        category = "💎 High Value Customer"

    elif customer_value > 50000:

        category = "🚀 Growth Customer"

    else:

        category = "👤 Standard Customer"

    # =================================================
    # REVENUE POTENTIAL
    # =================================================

    if monthly_charges > 5000:

        revenue_potential = "🔥 Very High"

    elif monthly_charges > 3000:

        revenue_potential = "📈 High"

    else:

        revenue_potential = "📊 Medium"

    # =================================================
    # KPI SECTION
    # =================================================

    st.divider()

    section_title(
        "Customer Intelligence Report"
    )

    metric_row(
        [
            (
                "Segment",
                persona
            ),
            (
                "Risk",
                risk_level
            ),
            (
                "CLV",
                f"₹{clv:,.0f}"
            ),
            (
                "Value",
                f"₹{customer_value:,.0f}"
            )
        ]
    )

    metric_row(
        [
            (
                "Loyalty",
                round(
                    loyalty_score,
                    2
                )
            ),
            (
                "Engagement",
                round(
                    engagement_score,
                    2
                )
            ),
            (
                "Health",
                round(
                    health_score,
                    2
                )
            ),
            (
                "Revenue",
                revenue_potential
            )
        ]
    )

    # =================================================
    # RECOMMENDATIONS
    # =================================================

    if churn_probability >= 0.70:

        recommendations = """
        • Immediate retention campaign

        • Personalized discount offers

        • Dedicated customer support

        • Loyalty bonus program

        • Customer satisfaction follow-up
        """

    elif churn_probability >= 0.40:

        recommendations = """
        • Improve engagement

        • Product recommendations

        • Reward points campaign

        • Upsell opportunities

        • Customer survey program
        """

    else:

        recommendations = """
        • Premium membership offer

        • Referral incentives

        • Cross-selling opportunities

        • Upselling premium services

        • VIP loyalty program
        """

    # =================================================
    # AI CONSULTANT
    # =================================================

    section_title(
        "🤖 AI Customer Consultant"
    )

    st.success(f"""
### Customer Assessment

Segment:
{persona}

Customer Category:
{category}

Risk Level:
{risk_level}

Revenue Potential:
{revenue_potential}

Recommended Actions:

{recommendations}
""")

    # =================================================
    # EXECUTIVE SUMMARY
    # =================================================

    section_title(
        "Executive Summary"
    )

    st.info(f"""
Customer Type:
{category}

Predicted Segment:
{persona}

Customer Lifetime Value:
₹{clv:,.0f}

Customer Value:
₹{customer_value:,.0f}

Revenue Potential:
{revenue_potential}

Retention Risk:
{risk_level}

Strategic Recommendation:

Focus on increasing engagement,
improving loyalty,
and maximizing customer lifetime value.
""")

    # =================================================
    # DOWNLOAD REPORT
    # =================================================

    report_df = pd.DataFrame({
        "Metric": [
            "Segment",
            "Customer Category",
            "Churn Probability",
            "Risk Level",
            "CLV",
            "Customer Value",
            "Loyalty Score",
            "Engagement Score",
            "Health Score",
            "Revenue Potential"
        ],
        "Value": [
            persona,
            category,
            round(
                churn_probability * 100,
                2
            ),
            risk_level,
            round(clv, 2),
            round(customer_value, 2),
            round(loyalty_score, 2),
            round(engagement_score, 2),
            round(health_score, 2),
            revenue_potential
        ]
    })

    csv = (
        report_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "⬇ Download Prediction Report",
        csv,
        "prediction_report.csv",
        "text/csv",
        use_container_width=True
    )

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "AI-Driven Customer Analytics Platform | Prediction Center"
)