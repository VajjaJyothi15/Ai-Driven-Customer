import pandas as pd


def preprocess(df):

    df = df.copy()

    if "Gender" in df.columns:

        df["Gender"] = (
            df["Gender"]
            .map(
                {
                    "Male": 1,
                    "Female": 0
                }
            )
        )

    df.fillna(0, inplace=True)

    return df