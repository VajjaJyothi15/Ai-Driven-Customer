import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

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
    page_title="Churn Intelligence",
    page_icon="⚠️",
    layout="wide"
)

load_css()

page_header(
    "⚠️ Churn Intelligence",
    "AI-Powered Customer Retention & Risk Analytics"
)

# =====================================================
# LOAD DATA
# =====================================================

df = get_data()
df = preprocess(df)
df = add_all_features(df)

# =====================================================
# LOAD MODEL
# =====================================================

try:

    churn_model = joblib.load(
        "saved_models/churn_model.pkl"
    )

except Exception as e:

    st.error(
        f"Model loading error: {e}"
    )

    st.stop()

# =====================================================
# PREDICT CHURN
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

df["ChurnProbability"] = (
    churn_model
    .predict_proba(
        df[feature_columns]
    )[:, 1]
)

# =====================================================
# RISK CLASSIFICATION
# =====================================================

def classify_risk(prob):

    if prob >= 0.70:
        return "High Risk"

    elif prob >= 0.40:
        return "Medium Risk"

    return "Low Risk"


df["RiskCategory"] = (
    df["ChurnProbability"]
    .apply(classify_risk)
)

# =====================================================
# KPI CALCULATIONS
# =====================================================

total_customers = len(df)

high_risk = len(
    df[
        df["RiskCategory"]
        == "High Risk"
    ]
)

medium_risk = len(
    df[
        df["RiskCategory"]
        == "Medium Risk"
    ]
)

low_risk = len(
    df[
        df["RiskCategory"]
        == "Low Risk"
    ]
)

churn_rate = round(
    (
        df["Churn"]
        .mean()
    ) * 100,
    2
)

retention_rate = round(
    100 - churn_rate,
    2
)

# =====================================================
# KPI SECTION
# =====================================================

section_title(
    "Risk Overview"
)

metric_row(
    [
        (
            "Customers",
            total_customers
        ),
        (
            "High Risk",
            high_risk
        ),
        (
            "Medium Risk",
            medium_risk
        ),
        (
            "Low Risk",
            low_risk
        )
    ]
)

metric_row(
    [
        (
            "Churn %",
            f"{churn_rate}%"
        ),
        (
            "Retention %",
            f"{retention_rate}%"
        ),
        (
            "Avg Risk",
            round(
                df["RiskScore"].mean(),
                2
            )
        ),
        (
            "Avg Satisfaction",
            round(
                df["SatisfactionScore"].mean(),
                2
            )
        )
    ]
)

st.divider()

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

section_title(
    "Feature Importance"
)

if hasattr(
    churn_model,
    "feature_importances_"
):

    importance_df = pd.DataFrame(
        {
            "Feature":
            feature_columns,

            "Importance":
            churn_model.feature_importances_
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
    )

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

# =====================================================
# HIGH RISK CUSTOMERS
# =====================================================

section_title(
    "🚨 High-Risk Customers"
)

high_risk_df = (
    df[
        df["RiskCategory"]
        == "High Risk"
    ]
    .sort_values(
        "ChurnProbability",
        ascending=False
    )
)

st.dataframe(
    high_risk_df[
        [
            "CustomerID",
            "AnnualIncome",
            "MonthlyCharges",
            "CustomerValue",
            "RiskScore",
            "ChurnProbability"
        ]
    ],
    use_container_width=True
)

# =====================================================
# CUSTOMER RISK EXPLORER
# =====================================================

section_title(
    "Customer Risk Explorer"
)

customer_id = st.selectbox(
    "Select Customer",
    sorted(
        df["CustomerID"]
        .unique()
    )
)

customer = df[
    df["CustomerID"]
    == customer_id
].iloc[0]

st.info(f"""
Customer ID:
{customer_id}

Risk Category:
{customer['RiskCategory']}

Churn Probability:
{customer['ChurnProbability']:.2%}

Customer Value:
₹{customer['CustomerValue']:,.0f}

Satisfaction Score:
{customer['SatisfactionScore']}

Loyalty Score:
{customer['LoyaltyScore']:.2f}
""")

# =====================================================
# RETENTION STRATEGY
# =====================================================

section_title(
    "🤖 Retention Recommendations"
)

st.success("""
### High-Risk Customers

• Immediate support outreach

• Retention campaigns

• Discount incentives

• Loyalty bonuses

---

### Medium-Risk Customers

• Personalized offers

• Engagement programs

• Product recommendations

---

### Low-Risk Customers

• Upsell premium services

• Referral campaigns

• Membership upgrades
""")

# =====================================================
# EXECUTIVE SUMMARY
# =====================================================

section_title(
    "Executive Summary"
)

avg_probability = round(
    df["ChurnProbability"]
    .mean() * 100,
    2
)

st.warning(f"""
Average Churn Probability:
{avg_probability}%

High-Risk Customers:
{high_risk}

Current Churn Rate:
{churn_rate}%

Retention Rate:
{retention_rate}%

Key Recommendations:

• Prioritize high-risk customers

• Improve customer satisfaction

• Increase loyalty rewards

• Strengthen customer engagement
""")

# =====================================================
# DOWNLOAD REPORT
# =====================================================

section_title(
    "Download Risk Report"
)

csv = (
    high_risk_df
    .to_csv(index=False)
    .encode("utf-8")
)

st.download_button(
    "⬇ Download High-Risk Customer Report",
    csv,
    "high_risk_customers.csv",
    "text/csv"
)

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "AI-Driven Customer Analytics Platform | Churn Intelligence"
)