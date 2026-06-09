import pandas as pd


def format_currency(value):
    return f"₹{value:,.0f}"


def churn_rate(df):

    return round(
        (
            df["Churn"].sum()
            /
            len(df)
        ) * 100,
        2
    )


def retention_rate(df):

    return round(
        (
            (len(df) - df["Churn"].sum())
            /
            len(df)
        ) * 100,
        2
    )


def total_revenue(df):

    return df["MonthlyCharges"].sum()


def avg_clv(df):

    return round(
        df["CLV"].mean(),
        2
    )