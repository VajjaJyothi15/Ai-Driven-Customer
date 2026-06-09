import streamlit as st
import pandas as pd

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
    page_title="AI Recommendation Center",
    page_icon="🤖",
    layout="wide"
)

load_css()

page_header(
    "🤖 AI Recommendation Center",
    "Executive Decision Support & Business Growth Intelligence"
)

# =====================================================
# LOAD DATA
# =====================================================

df = get_data()
df = preprocess(df)
df = add_all_features(df)

# =====================================================
# BUSINESS HEALTH SCORE
# =====================================================

avg_satisfaction = df["SatisfactionScore"].mean()
avg_loyalty = df["LoyaltyScore"].mean()
avg_engagement = df["EngagementScore"].mean()

churn_rate = (
    df["Churn"].mean()
) * 100

# Custom business score
business_health_score = (
    (
        avg_satisfaction * 10
    )
    +
    (
        avg_loyalty * 2
    )
    +
    (
        avg_engagement
    )
    -
    (
        churn_rate
    )
)

business_health_score = max(
    0,
    min(
        round(
            business_health_score,
            2
        ),
        100
    )
)

# =====================================================
# OVERVIEW
# =====================================================

section_title(
    "Business Overview"
)

metric_row(
    [
        (
            "Business Health",
            f"{business_health_score}/100"
        ),
        (
            "Customers",
            len(df)
        ),
        (
            "Revenue",
            f"₹{df['MonthlyCharges'].sum():,.0f}"
        ),
        (
            "Churn %",
            f"{churn_rate:.2f}%"
        )
    ]
)

st.divider()

# =====================================================
# HEALTH INTERPRETATION
# =====================================================

if business_health_score >= 80:

    st.success("""
    Excellent Business Performance

    Customer engagement is strong.

    Retention is healthy.

    Revenue opportunities exist through upselling.
    """)

elif business_health_score >= 60:

    st.warning("""
    Moderate Business Performance

    Customer retention can be improved.

    Loyalty initiatives recommended.
    """)

else:

    st.error("""
    Business Health Requires Attention

    Focus on retention and engagement.

    Reduce churn risk immediately.
    """)

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Retention",
        "Revenue",
        "Engagement",
        "Executive Plan"
    ]
)

# =====================================================
# RETENTION TAB
# =====================================================

with tab1:

    st.subheader(
        "Customer Retention Strategy"
    )

    high_risk_customers = len(
        df[
            df["RiskScore"]
            >
            df["RiskScore"].quantile(0.75)
        ]
    )

    st.info(f"""
    High-Risk Customers:
    {high_risk_customers}

    Recommended Actions:

    • Personalized retention campaigns

    • Loyalty rewards

    • Dedicated customer support

    • Satisfaction improvement programs

    • Special discount offers
    """)

# =====================================================
# REVENUE TAB
# =====================================================

with tab2:

    st.subheader(
        "Revenue Growth Strategy"
    )

    high_value_customers = len(
        df[
            df["CustomerValue"]
            >
            df["CustomerValue"].quantile(0.75)
        ]
    )

    st.success(f"""
    High Value Customers:
    {high_value_customers}

    Recommended Actions:

    • Upsell premium products

    • Introduce subscription plans

    • Cross-sell related products

    • Launch VIP customer programs

    • Create premium membership tiers
    """)

# =====================================================
# ENGAGEMENT TAB
# =====================================================

with tab3:

    st.subheader(
        "Customer Engagement Strategy"
    )

    low_engagement = len(
        df[
            df["EngagementScore"]
            <
            df["EngagementScore"].median()
        ]
    )

    st.warning(f"""
    Low Engagement Customers:
    {low_engagement}

    Recommended Actions:

    • Personalized recommendations

    • Email campaigns

    • Mobile notifications

    • Referral programs

    • Reward-based engagement
    """)

# =====================================================
# EXECUTIVE PLAN TAB
# =====================================================

with tab4:

    st.subheader(
        "Executive Action Plan"
    )

    st.success("""
    ### Short-Term Actions

    • Reduce churn risk

    • Improve satisfaction

    • Increase customer engagement

    ---

    ### Mid-Term Actions

    • Expand loyalty programs

    • Launch premium services

    • Increase cross-selling

    ---

    ### Long-Term Actions

    • Improve customer lifetime value

    • Grow premium customer base

    • Build predictive retention programs

    • Enhance AI-driven personalization
    """)

# =====================================================
# AI EXECUTIVE SUMMARY
# =====================================================

section_title(
    "🤖 AI Executive Summary"
)

total_revenue = round(
    df["MonthlyCharges"].sum(),
    0
)

avg_clv = round(
    df["CLV"].mean(),
    0
)

st.info(f"""
Business Health Score:
{business_health_score}/100

Total Customers:
{len(df)}

Total Revenue:
₹{total_revenue:,.0f}

Average Customer Lifetime Value:
₹{avg_clv:,.0f}

Key Priorities:

1. Improve retention

2. Increase engagement

3. Expand premium customer programs

4. Grow customer lifetime value

5. Strengthen loyalty initiatives
""")

# =====================================================
# DOWNLOAD EXECUTIVE REPORT
# =====================================================

section_title(
    "Download Executive Report"
)

report_df = pd.DataFrame({
    "Metric": [
        "Business Health Score",
        "Customers",
        "Revenue",
        "Average CLV",
        "Churn Rate"
    ],
    "Value": [
        business_health_score,
        len(df),
        total_revenue,
        avg_clv,
        round(churn_rate, 2)
    ]
})

csv = report_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "⬇ Download Executive Report",
    csv,
    "executive_report.csv",
    "text/csv"
)

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "AI-Driven Customer Analytics Platform | AI Recommendation Center"
)