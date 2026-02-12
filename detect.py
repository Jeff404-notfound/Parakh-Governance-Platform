import pandas as pd

df = pd.read_csv("data/aadhaar_sample.csv")

# 1️⃣ Suspicious Update Patterns
suspicious_centers = df.groupby("center_id")["updated"].sum()
suspicious_centers = suspicious_centers[suspicious_centers > 3]

print("\n🚨 Suspicious Centres (too many updates):")
print(suspicious_centers)

# 2️⃣ Overloaded Centres
center_load = df["center_id"].value_counts()
overloaded = center_load[center_load > 3]

print("\n🔥 Overloaded Centres:")
print(overloaded)

# 3️⃣ Under-Served Districts
district_updates = df.groupby("district")["updated"].mean()
under_served = district_updates[district_updates < 0.4]

print("\n📉 Under-served Districts:")
print(under_served)

# 4️⃣ High-Risk Demographics
df["risk"] = ((df["age"] > 60) & (df["biometric_failed"] == 1))
risk_group = df[df["risk"] == True]

print("\n⚠ High-risk Citizens:")
print(risk_group[["district","age","gender","biometric_failed"]])
