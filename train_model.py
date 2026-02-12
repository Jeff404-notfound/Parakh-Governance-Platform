import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/aadhaar_sample.csv")

# Rule-based labels
# Needs update if last update > 10 years
df["needs_update"] = df["last_update_year"] < 2015

# Fraud if too many updates
df["fraud"] = df["updates_count"] > 6

# Convert categorical to numbers
df["area"] = df["area"].map({"Urban":1,"Rural":0})
df["gender"] = df["gender"].map({"Male":1,"Female":0})

features = ["age","area","biometric_failed","updates_count"]

X = df[features]

y_update = df["needs_update"]
y_fraud = df["fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y_update, test_size=0.3)

model_update = RandomForestClassifier()
model_update.fit(X_train, y_train)

model_fraud = RandomForestClassifier()
model_fraud.fit(X, y_fraud)

print("Models trained successfully")
