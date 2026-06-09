from sklearn.ensemble import RandomForestClassifier


def get_churn_model():

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )

    return model