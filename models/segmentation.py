from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os


def train_segmentation(df):

    features = df[
        [
            "AnnualIncome",
            "SpendingScore",
            "PurchaseFrequency"
        ]
    ]

    scaler = StandardScaler()

    scaled = scaler.fit_transform(
        features
    )

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    df["Segment"] = (
        kmeans.fit_predict(
            scaled
        )
    )

    os.makedirs(
        "saved_models",
        exist_ok=True
    )

    joblib.dump(
        scaler,
        "saved_models/scaler.pkl"
    )

    joblib.dump(
        kmeans,
        "saved_models/segment_model.pkl"
    )

    return df