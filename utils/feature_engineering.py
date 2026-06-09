import pandas as pd


def add_all_features(df):

    df = df.copy()

    # Customer Lifetime Value
    df["CLV"] = (
        df["MonthlyCharges"]
        *
        df["Tenure"]
    )

    # Loyalty Score
    df["LoyaltyScore"] = (
        df["Tenure"] * 0.5
        +
        df["PurchaseFrequency"] * 0.3
        +
        df["SatisfactionScore"] * 0.2
    )

    # Engagement Score
    df["EngagementScore"] = (
        df["PurchaseFrequency"] * 0.6
        +
        df["SpendingScore"] * 0.4
    )

    # Customer Value
    df["CustomerValue"] = (
        df["AnnualIncome"] * 0.3
        +
        df["SpendingScore"] * 0.3
        +
        df["CLV"] * 0.4
    )

    # Risk Score
    df["RiskScore"] = (
        (100 - df["SatisfactionScore"] * 10)
        +
        (100 - df["SpendingScore"])
    )

    # Customer Health Score
    df["CustomerHealthScore"] = (
        200 - df["RiskScore"]
    )

    return df