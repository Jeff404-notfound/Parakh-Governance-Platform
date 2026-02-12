import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/aadhaar_sample.csv")

df["area"] = df["area"].map({"Urban":1,"Rural":0})
df["gender"] = df["gender"].map({"Male":1,"Female":0})

X = df[["age","area","biometric_failed","updates_count"]]

# Re-train quickly
df["needs_update"] = df["last_update_year"] < 2015
df["fraud"] = df["updates_count"] > 6

model_update = RandomForestClassifier().fit(X, df["needs_update"])
model_fraud = RandomForestClassifier().fit(X, df["fraud"])

df["update_prediction"] = model_update.predict(X)
df["fraud_prediction"] = model_fraud.predict(X)

print("\n🧠 Aadhaar AI Predictions:\n")
print(df[["district","age","update_prediction","fraud_prediction"]])
load = df.groupby("district")["updates_count"].sum()
print("\n📈 Future Centre Load Risk:")
print(load.sort_values(ascending=False))
