import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("data/aadhaar_sample.csv")

df["area"] = df["area"].map({"Urban":1,"Rural":0})
df["gender"] = df["gender"].map({"Male":1,"Female":0})

# Create labels again
df["needs_update"] = df["last_update_year"] < 2015
df["fraud"] = df["updates_count"] > 6

X = df[["age","area","biometric_failed","updates_count"]]

model_update = RandomForestClassifier().fit(X, df["needs_update"])
model_fraud = RandomForestClassifier().fit(X, df["fraud"])

df["update_pred"] = model_update.predict(X)
df["fraud_pred"] = model_fraud.predict(X)

# --- POLICY LOGIC ---

recommendations = []

# 1️⃣ New Centres
district_load = df.groupby("district")["updates_count"].sum()
for district, load in district_load.items():
    if load > 12:
        recommendations.append(f"Open 2 new Aadhaar centres in {district}")

# 2️⃣ Update Drives
women_update = df[(df["gender"]==0) & (df["update_pred"]==1)]
districts = women_update["district"].value_counts()
for d in districts.index:
    recommendations.append(f"Run Aadhaar update drive for women in {d}")

# 3️⃣ Fraud Audits
fraud_centers = df[df["fraud_pred"]==1]["center_id"].unique()
for c in fraud_centers:
    recommendations.append(f"Audit Aadhaar centre {c} for potential fraud")

# --- OUTPUT ---

print("\n🏛 Aadhaar Policy Recommendations:\n")
for r in recommendations:
    print("•", r)
