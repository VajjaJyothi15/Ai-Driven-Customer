import streamlit as st
import pandas as pd
import numpy as np
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
    page_title="Revenue Intelligence",
    page_icon="💰",
    layout="wide"
)

load_css()

page_header(
    "💰 Revenue Intelligence",
    "Revenue Analytics, Customer Value & Growth Opportunities"
)

# =====================================================
# LOAD DATA
# =====================================================

df = get_data()
df = preprocess(df)
df = add_all_features(df)

# =====================================================
# KPI CALCULATIONS
# =====================================================

total_revenue = round(
    df["MonthlyCharges"].sum(),
    0
)

avg_revenue = round(
    df["MonthlyCharges"].mean(),
    0
)

avg_clv = round(
    df["CLV"].mean(),
    0
)

max_customer_value = round(
    df["CustomerValue"].max(),
    0
)

high_value_customers = len(
    df[
        df["CustomerValue"]
        >
        df["CustomerValue"].median()
    ]
)

# =====================================================
# KPI SECTION
# =====================================================

section_title(
    "Revenue Overview"
)

metric_row(
    [
        (
            "Total Revenue",
            f"₹{total_revenue:,.0f}"
        ),
        (
            "Avg Revenue",
            f"₹{avg_revenue:,.0f}"
        ),
        (
            "Avg CLV",
            f"₹{avg_clv:,.0f}"
        ),
        (
            "Top Customer Value",
            f"₹{max_customer_value:,.0f}"
        )
    ]
)

metric_row(
    [
        (
            "High Value Customers",
            high_value_customers
        ),
        (
            "Avg Income",
            f"₹{df['AnnualIncome'].mean():,.0f}"
        ),
        (
            "Avg Spending",
            round(
                df["SpendingScore"].mean(),
                2
            )
        ),
        (
            "Avg Loyalty",
            round(
                df["LoyaltyScore"].mean(),
                2
            )
        )
    ]
)

st.divider()

# =====================================================
# TOP CUSTOMERS
# =====================================================

section_title(
    "🏆 Top Revenue Customers"
)

top_customers = (
    df.sort_values(
        "CustomerValue",
        ascending=False
    )
    .head(20)
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
# REVENUE FORECAST
# =====================================================

section_title(
    "Revenue Forecast"
)

months = np.arange(1, 13)

forecast = (
    avg_revenue *
    len(df)
) * (
    1 + (months * 0.03)
)

forecast_df = pd.DataFrame({
    "Month": months,
    "ForecastRevenue": forecast
})

fig_forecast = px.line(
    forecast_df,
    x="Month",
    y="ForecastRevenue",
    markers=True
)

st.plotly_chart(
    fig_forecast,
    use_container_width=True
)

# =====================================================
# REVENUE OPPORTUNITY
# =====================================================

section_title(
    "Revenue Opportunity Analysis"
)

fig_opportunity = px.scatter(
    df,
    x="CustomerValue",
    y="CLV",
    size="MonthlyCharges",
    hover_data=[
        "CustomerID"
    ]
)

st.plotly_chart(
    fig_opportunity,
    use_container_width=True
)

# =====================================================
# CUSTOMER VALUE ANALYSIS
# =====================================================

section_title(
    "Customer Value Insights"
)

high_value = len(
    df[
        df["CustomerValue"]
        >
        df["CustomerValue"].quantile(0.75)
    ]
)

medium_value = len(
    df[
        (
            df["CustomerValue"]
            >
            df["CustomerValue"].quantile(0.50)
        )
        &
        (
            df["CustomerValue"]
            <=
            df["CustomerValue"].quantile(0.75)
        )
    ]
)

standard_value = len(df) - (
    high_value +
    medium_value
)

c1, c2, c3 = st.columns(3)

with c1:

    st.success(f"""
    High Value Customers

    {high_value}

    Premium Revenue Potential
    """)

with c2:

    st.info(f"""
    Medium Value Customers

    {medium_value}

    Upsell Opportunities
    """)

with c3:

    st.warning(f"""
    Standard Customers

    {standard_value}

    Engagement Opportunity
    """)

# =====================================================
# BUSINESS RECOMMENDATIONS
# =====================================================

section_title(
    "🤖 AI Revenue Recommendations"
)

st.success("""
### Revenue Growth Strategy

• Focus on high-value customers

• Launch premium membership plans

• Improve customer retention

• Increase cross-selling opportunities

• Target medium-value customers for upselling

• Strengthen loyalty programs

• Offer personalized product recommendations

• Increase engagement among low-spending customers
""")

# =====================================================
# EXECUTIVE SUMMARY
# =====================================================

section_title(
    "Executive Revenue Summary"
)

st.info(f"""
Total Revenue:
₹{total_revenue:,.0f}

Average Customer Lifetime Value:
₹{avg_clv:,.0f}

High Value Customers:
{high_value_customers}

Growth Opportunities:

• Increase premium customer base

• Improve retention strategies

• Expand cross-sell campaigns

• Strengthen loyalty initiatives

Expected Revenue Growth:
+3% Monthly Projection
""")

# =====================================================
# DOWNLOAD REPORT
# =====================================================

section_title(
    "Download Revenue Report"
)

csv = (
    top_customers
    .to_csv(index=False)
    .encode("utf-8")
)

st.download_button(
    "⬇ Download Revenue Report",
    csv,
    "revenue_report.csv",
    "text/csv"
)

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "AI-Driven Customer Analytics Platform | Revenue Intelligence"
)