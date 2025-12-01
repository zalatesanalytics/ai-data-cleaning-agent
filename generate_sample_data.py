import pandas as pd
import numpy as np
import random
from pathlib import Path

np.random.seed(42)
random.seed(42)

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

def inject_anomalies(df, anomaly_fraction=0.15):
    """Randomly inject missing values, extreme values, and weird strings."""
    n = len(df)
    anomaly_indices = np.random.choice(df.index, int(n * anomaly_fraction), replace=False)

    for idx in anomaly_indices:
        col = random.choice(df.columns.tolist())
        # For numeric-like columns
        if pd.api.types.is_numeric_dtype(df[col]):
            df.loc[idx, col] = random.choice([9999, -50, None])
        else:
            df.loc[idx, col] = random.choice(["Unknown", None, "???"])
    return df

# -------------------------
# Dataset 1: 200 rows
# Columns: ID, name, age, sex, eduction, inc1, inc2, employed, food security, consume meat
# -------------------------
n1 = 200
df1 = pd.DataFrame({
    "ID": range(1, n1 + 1),
    "name": [f"Person_{i}" for i in range(1, n1 + 1)],
    "age": np.random.choice([8, 15, 18, 25, 40, 70, None, 999], n1),
    "sex": np.random.choice(["Male", "Female", None], n1),
    "eduction": np.random.choice(["None", "Primary", "Secondary", "College", "University", None], n1),
    "inc1": np.random.choice([np.random.randint(100, 2000), None, -20], n1),
    "inc2": np.random.choice([np.random.randint(50, 1500), None, 5000], n1),
    "employed": np.random.choice(["Yes", "No", "Unknown"], n1),
    "food security": np.random.choice(["Secure", "Insecure"], n1),
    "consume meat": np.random.choice(["Yes", "No", "Always", None], n1),
})

df1 = inject_anomalies(df1, anomaly_fraction=0.15)
df1.to_csv(data_dir / "dataset1_dirty.csv", index=False)
print("Saved data/dataset1_dirty.csv")

# -------------------------
# Dataset 2: 420 rows
# Columns: ID, name of hhs, Age, Gender, education, income1, income2, employed,
#          secure food, eat meat, region, area, status
# -------------------------
n2 = 420
df2 = pd.DataFrame({
    "ID": range(1, n2 + 1),
    "name of hhs": [f"HH_{i}" for i in range(1, n2 + 1)],
    "Age": np.random.choice([8, 17, 25, 50, 80, None], n2),
    "Gender": np.random.choice(["Male", "Female", "Other", None], n2),
    "education": np.random.choice(["None", "Primary", "Secondary", "College", "University", None], n2),
    "income1": np.random.choice([np.random.randint(100, 4000), None, -100], n2),
    "income2": np.random.choice([np.random.randint(50, 2500), None, 9999], n2),
    "employed": np.random.choice(["Yes", "No", "Unknown"], n2),
    "secure food": np.random.choice(["Secure", "Insecure"], n2),
    "eat meat": np.random.choice(["Yes", "No", "Always"], n2),
    "region": np.random.choice(["North", "South", "East", "West", "Unknown"], n2),
    "area": np.random.choice(["Urban", "Rural", "Mixed"], n2),
    "status": np.random.choice(["Active", "Inactive", "Unknown"], n2),
})

df2 = inject_anomalies(df2, anomaly_fraction=0.15)
df2.to_csv(data_dir / "dataset2_dirty.csv", index=False)
print("Saved data/dataset2_dirty.csv")

# -------------------------
# Dataset 3: 320 rows
# Columns: IDnumber, name, age, sex, eduction, incomes1, household income,
#          Employ, foodsecurity, consmeat, eat bread, eat checkin
# -------------------------
n3 = 320
df3 = pd.DataFrame({
    "IDnumber": range(1, n3 + 1),
    "name": [f"Person_{i}" for i in range(1, n3 + 1)],
    "age": np.random.choice([7, 15, 22, 60, 90, None], n3),
    "sex": np.random.choice(["Male", "Female", "Unknown"], n3),
    "eduction": np.random.choice(["None", "Primary", "Secondary", "College", "University", None], n3),
    "incomes1": np.random.choice([np.random.randint(100, 5000), None, -30], n3),
    "household income": np.random.choice([np.random.randint(300, 8000), None, 20000], n3),
    "Employ": np.random.choice(["Yes", "No", "Unknown"], n3),
    "foodsecurity": np.random.choice(["Secure", "Insecure"], n3),
    "consmeat": np.random.choice(["Yes", "No", "Daily"], n3),
    "eat bread": np.random.choice(["Yes", "No", None], n3),
    "eat checkin": np.random.choice(["Yes", "No", "Always"], n3),
})

df3 = inject_anomalies(df3, anomaly_fraction=0.15)
df3.to_csv(data_dir / "dataset3_dirty.csv", index=False)
print("Saved data/dataset3_dirty.csv")

print("All sample dirty datasets generated.")
