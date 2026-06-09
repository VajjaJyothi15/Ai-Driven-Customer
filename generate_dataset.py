import pandas as pd
import numpy as np

np.random.seed(42)

n = 2000

data = {
    "CustomerID": range(1, n + 1),
    "Age": np.random.randint(18, 70, n),
    "Gender": np.random.choice(["Male", "Female"], n),
    "AnnualIncome": np.random.randint(20000, 150000, n),
    "SpendingScore": np.random.randint(1, 100, n),
    "Tenure": np.random.randint(1, 120, n),
    "MonthlyCharges": np.random.randint(500, 10000, n),
    "PurchaseFrequency": np.random.randint(1, 50, n),
    "SatisfactionScore": np.random.randint(1, 10, n),
    "Churn": np.random.choice([0, 1], n, p=[0.8, 0.2])
}

df = pd.DataFrame(data)

df.to_csv("dataset/customer_data.csv", index=False)

print("Dataset Generated Successfully")