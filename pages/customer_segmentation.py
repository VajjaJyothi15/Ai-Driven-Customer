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
    page_title="Customer Segmentation",
    page_icon="🎯",
    layout="wide"
)

load_css()

page_header(
    "🎯 Customer Segmentation",
    "K-Means Based Customer Persona Discovery"
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
# GENERATE SEGMENTS
# =====================================================

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

segment_names = {
    0: "💎 Premium Customers",
    1: "⭐ Loyal Customers",
    2: "⚠ At-Risk Customers",
    3: "🆕 New Customers"
}

df["Persona"] = (
    df["Segment"]
    .map(segment_names)
)

# =====================================================
# KPI CALCULATIONS
# =====================================================

premium_count = len(
    df[df["Segment"] == 0]
)

loyal_count = len(
    df[df["Segment"] == 1]
)

risk_count = len(
    df[df["Segment"] == 2]
)

new_count = len(
    df[df["Segment"] == 3]
)

# =====================================================
# SEGMENT KPIs
# =====================================================

section_title("Segment Overview")

metric_row(
    [
        (
            "Premium",
            premium_count
        ),
        (
            "Loyal",
            loyal_count
        ),
        (
            "At-Risk",
            risk_count
        ),
        (
            "New",
            new_count
        )
    ]
)

st.divider()

# =====================================================
# SEGMENT DISTRIBUTION
# =====================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader(
        "Segment Distribution"
    )

    fig_pie = px.pie(
        df,
        names="Persona"
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )

# =====================================================
# REVENUE CONTRIBUTION
# =====================================================

with col2:

    st.subheader(
        "Revenue by Segment"
    )

    revenue = (
        df.groupby("Persona")
        ["MonthlyCharges"]
        .sum()
        .reset_index()
    )

    fig_bar = px.bar(
        revenue,
        x="Persona",
        y="MonthlyCharges",
        text_auto=True
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )

# =====================================================
# 3D CLUSTER VISUALIZATION
# =====================================================

st.subheader(
    "3D Customer Clusters"
)

fig_3d = px.scatter_3d(
    df,
    x="AnnualIncome",
    y="SpendingScore",
    z="PurchaseFrequency",
    color="Persona",
    hover_data=[
        "CustomerID"
    ]
)

st.plotly_chart(
    fig_3d,
    use_container_width=True
)

# =====================================================
# SEGMENT COMPARISON
# =====================================================

section_title(
    "Segment Performance"
)

comparison = (
    df.groupby("Persona")
    [
        [
            "AnnualIncome",
            "SpendingScore",
            "PurchaseFrequency",
            "CustomerValue",
            "CLV"
        ]
    ]
    .mean()
    .round(2)
)

st.dataframe(
    comparison,
    use_container_width=True
)

# =====================================================
# PERSONA EXPLORER
# =====================================================

section_title(
    "Persona Explorer"
)

selected_persona = st.selectbox(
    "Select Customer Segment",
    list(segment_names.values())
)

filtered_df = df[
    df["Persona"]
    == selected_persona
]

st.write(
    f"Customers Found: {len(filtered_df)}"
)

st.dataframe(
    filtered_df[
        [
            "CustomerID",
            "AnnualIncome",
            "SpendingScore",
            "PurchaseFrequency",
            "CustomerValue",
            "CLV"
        ]
    ],
    use_container_width=True
)

# =====================================================
# SEGMENT RECOMMENDATIONS
# =====================================================

section_title(
    "🤖 Segment Intelligence"
)

st.success("""
### 💎 Premium Customers

• Highest revenue potential

• Offer VIP programs

• Upsell premium services

---

### ⭐ Loyal Customers

• Reward loyalty

• Encourage referrals

• Cross-sell products

---

### ⚠ At-Risk Customers

• Immediate retention campaigns

• Satisfaction improvement plans

• Discount incentives

---

### 🆕 New Customers

• Onboarding programs

• Engagement campaigns

• Product education
""")

# =====================================================
# SEGMENT SUMMARY
# =====================================================

section_title(
    "Executive Segment Summary"
)

largest_segment = (
    df["Persona"]
    .value_counts()
    .idxmax()
)

highest_revenue_segment = (
    revenue.sort_values(
        "MonthlyCharges",
        ascending=False
    )
    .iloc[0]["Persona"]
)

st.info(f"""
Largest Segment:
{largest_segment}

Highest Revenue Segment:
{highest_revenue_segment}

Total Segments:
4

Business Opportunities:

• Focus on retaining high-value customers

• Increase engagement among new customers

• Reduce churn within at-risk groups

• Expand premium offerings
""")

# =====================================================
# DOWNLOAD REPORT
# =====================================================

section_title(
    "Download Report"
)

csv = (
    filtered_df
    .to_csv(index=False)
    .encode("utf-8")
)

st.download_button(
    "⬇ Download Segment Report",
    csv,
    f"{selected_persona}_report.csv",
    "text/csv"
)

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "AI-Driven Customer Analytics Platform | Customer Segmentation"
)