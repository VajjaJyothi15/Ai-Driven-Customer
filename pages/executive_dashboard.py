import streamlit as st
import pandas as pd
import plotly.express as px
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
    page_title="Executive Dashboard",
    page_icon="📊",
    layout="wide"
)

load_css()

page_header(
    "📊 Executive Dashboard",
    "Business Performance & Customer Intelligence Overview"
)

# =====================================================
# LOAD DATA
# =====================================================

df = get_data()

df = preprocess(df)

df = add_all_features(df)

# =====================================================
# LOAD SEGMENT MODEL
# =====================================================

try:

    scaler = joblib.load(
        "saved_models/scaler.pkl"
    )

    segment_model = joblib.load(
        "saved_models/segment_model.pkl"
    )

    segment_features = df[
        [
            "AnnualIncome",
            "SpendingScore",
            "PurchaseFrequency"
        ]
    ]

    scaled = scaler.transform(
        segment_features
    )

    df["Segment"] = segment_model.predict(
        scaled
    )

except:

    df["Segment"] = 0

# =====================================================
# SEGMENT LABELS
# =====================================================

segment_names = {
    0: "Premium",
    1: "Loyal",
    2: "At Risk",
    3: "New"
}

df["SegmentName"] = (
    df["Segment"]
    .map(segment_names)
)

# =====================================================
# KPI CALCULATIONS
# =====================================================

total_customers = len(df)

total_revenue = (
    df["MonthlyCharges"]
    .sum()
)

avg_clv = (
    df["CLV"]
    .mean()
)

avg_loyalty = (
    df["LoyaltyScore"]
    .mean()
)

avg_satisfaction = (
    df["SatisfactionScore"]
    .mean()
)

avg_spending = (
    df["SpendingScore"]
    .mean()
)

churn_rate = (
    df["Churn"]
    .mean()
) * 100

retention_rate = (
    100 - churn_rate
)

high_value_customers = len(
    df[
        df["CustomerValue"]
        >
        df["CustomerValue"].median()
    ]
)

# =====================================================
# KPI ROW 1
# =====================================================

section_title("Business KPIs")

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
            "Retention %",
            f"{retention_rate:.1f}%"
        )
    ]
)

# =====================================================
# KPI ROW 2
# =====================================================

metric_row(
    [
        (
            "Avg Satisfaction",
            f"{avg_satisfaction:.2f}"
        ),
        (
            "Avg Loyalty",
            f"{avg_loyalty:.2f}"
        ),
        (
            "Avg Spending",
            f"{avg_spending:.2f}"
        ),
        (
            "High Value Customers",
            high_value_customers
        )
    ]
)

st.divider()

# =====================================================
# CHARTS
# =====================================================

col1, col2 = st.columns(2)

# =====================================================
# SEGMENT DISTRIBUTION
# =====================================================

with col1:

    st.subheader(
        "🎯 Customer Segments"
    )

    fig_segments = px.pie(
        df,
        names="SegmentName"
    )

    st.plotly_chart(
        fig_segments,
        use_container_width=True
    )

# =====================================================
# CHURN DISTRIBUTION
# =====================================================

with col2:

    st.subheader(
        "⚠ Churn Overview"
    )

    churn_data = pd.DataFrame(
        {
            "Status": [
                "Active",
                "Churned"
            ],
            "Count": [
                len(
                    df[df["Churn"] == 0]
                ),
                len(
                    df[df["Churn"] == 1]
                )
            ]
        }
    )

    fig_churn = px.pie(
        churn_data,
        names="Status",
        values="Count"
    )

    st.plotly_chart(
        fig_churn,
        use_container_width=True
    )

# =====================================================
# REVENUE BY SEGMENT
# =====================================================

st.subheader(
    "💰 Revenue Contribution by Segment"
)

segment_revenue = (
    df.groupby(
        "SegmentName"
    )["MonthlyCharges"]
    .sum()
    .reset_index()
)

fig_revenue = px.bar(
    segment_revenue,
    x="SegmentName",
    y="MonthlyCharges",
    text_auto=True
)

st.plotly_chart(
    fig_revenue,
    use_container_width=True
)

# =====================================================
# TOP CUSTOMERS
# =====================================================

st.subheader(
    "🏆 Top Customers"
)

top_customers = (
    df.sort_values(
        "CustomerValue",
        ascending=False
    )
    .head(10)
)

st.dataframe(
    top_customers[
        [
            "CustomerID",
            "AnnualIncome",
            "MonthlyCharges",
            "CLV",
            "CustomerValue",
            "LoyaltyScore"
        ]
    ],
    use_container_width=True
)

# =====================================================
# CUSTOMER INSIGHTS
# =====================================================

st.subheader(
    "👥 Customer Insights"
)

c1, c2, c3 = st.columns(3)

with c1:

    st.info(f"""
    Average Income

    ₹{df['AnnualIncome'].mean():,.0f}

    Average Spending Score

    {avg_spending:.2f}
    """)

with c2:

    st.success(f"""
    Loyalty Score

    {avg_loyalty:.2f}

    Satisfaction Score

    {avg_satisfaction:.2f}
    """)

with c3:

    st.warning(f"""
    Churn Rate

    {churn_rate:.2f}%

    Retention Rate

    {retention_rate:.2f}%
    """)

# =====================================================
# EXECUTIVE SUMMARY
# =====================================================

st.divider()

st.subheader(
    "🤖 Executive AI Insights"
)

top_segment = (
    segment_revenue
    .sort_values(
        "MonthlyCharges",
        ascending=False
    )
    .iloc[0]["SegmentName"]
)

st.success(f"""
### Business Summary

Revenue Generated:
₹{total_revenue:,.0f}

Highest Revenue Segment:
{top_segment}

Customer Retention:
{retention_rate:.2f}%

Average Customer Lifetime Value:
₹{avg_clv:,.0f}

Growth Opportunities:

• Increase engagement among medium-spending customers

• Upsell loyal customers

• Run retention campaigns for at-risk customers

• Improve satisfaction to reduce churn

• Expand premium customer programs
""")

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "AI-Driven Customer Analytics Platform | Executive Dashboard"
)