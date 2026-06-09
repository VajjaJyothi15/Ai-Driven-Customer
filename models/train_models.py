import os
import sys
import joblib

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)

from database.db_operations import get_data
from utils.preprocessing import preprocess
from utils.feature_engineering import add_all_features
from models.segmentation import train_segmentation
from models.churn_prediction import get_churn_model

print("Loading Data...")

df = get_data()

print(f"Dataset Shape: {df.shape}")

df = preprocess(df)

df = add_all_features(df)

df = train_segmentation(df)

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

X = df[feature_columns]

y = df["Churn"]

model = get_churn_model()

model.fit(X, y)

os.makedirs(
    "saved_models",
    exist_ok=True
)

joblib.dump(
    model,
    "saved_models/churn_model.pkl"
)

print("Segmentation Model Saved")
print("Churn Model Saved")
print("Training Complete")