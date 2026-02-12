from flask import Flask, render_template, jsonify, request, redirect
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

app = Flask(__name__)

# ✅ Absolute path setup (VERY IMPORTANT FOR RENDER)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "aadhaar_sample.csv")
HELPDESK_PATH = os.path.join(BASE_DIR, "data", "helpdesk.csv")


# ------------------ AI + DATA LOADER ------------------

def load_and_train():

    df = pd.read_csv(DATA_PATH)

    df["area"] = df["area"].map({"Urban": 1, "Rural": 0})
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    df["needs_update"] = df["last_update_year"] < 2015
    df["fraud"] = df["updates_count"] > 6

    X = df[["age", "area", "biometric_failed", "updates_count"]]

    model_update = RandomForestClassifier(random_state=42).fit(X, df["needs_update"])
    model_fraud = RandomForestClassifier(random_state=42).fit(X, df["fraud"])

    df["update_pred"] = model_update.predict(X)
    df["fraud_pred"] = model_fraud.predict(X)

    return df, model_update, model_fraud


df, model_update, model_fraud = load_and_train()


# ------------------ ROUTES ------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/update")
def update_page():
    return render_template("update.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/help")
def help_page():
    return render_template("help.html")


# ✅ HELP DESK
@app.route("/help_submit", methods=["POST"])
def help_submit():

    ticket = {
        "name": request.form["name"],
        "aadhaar": request.form["aadhaar"],
        "type": request.form["type"],
        "issue": request.form["issue"]
    }

    # Create file if not exists
    if not os.path.exists(HELPDESK_PATH):
        df_help = pd.DataFrame(columns=["name", "aadhaar", "type", "issue"])
    else:
        df_help = pd.read_csv(HELPDESK_PATH)

    df_help = pd.concat([df_help, pd.DataFrame([ticket])], ignore_index=True)
    df_help.to_csv(HELPDESK_PATH, index=False)

    return render_template("help_success.html")


# ✅ DATA UPDATE
@app.route("/submit", methods=["POST"])
def submit():

    global df, model_update, model_fraud

    new = {
        "state": request.form["state"],
        "district": request.form["district"],
        "gender": request.form["gender"],
        "age": int(request.form["age"]),
        "area": request.form["area"],
        "center_id": request.form["center"],
        "last_update_year": int(request.form["year"]),
        "biometric_failed": int(request.form["bio"]),
        "updates_count": int(request.form["count"])
    }

    data = pd.read_csv(DATA_PATH)

    data = pd.concat([data, pd.DataFrame([new])], ignore_index=True)
    data.to_csv(DATA_PATH, index=False)

    # retrain models
    df, model_update, model_fraud = load_and_train()

    return redirect("/")


@app.route("/stats")
def stats():
    return jsonify({
        "state": df["state"].value_counts().to_dict(),
        "gender": df["gender"].value_counts().to_dict()
    })


@app.route("/problems")
def problems():

    under_served = df.groupby("district")["update_pred"].mean()
    under = under_served[under_served > 0.6].index.tolist()

    fraud_centers = df[df["fraud_pred"] == 1]["center_id"].unique().tolist()

    return jsonify({
        "under_served": under,
        "fraud_centers": fraud_centers
    })


@app.route("/policy")
def policy():

    recs = []

    load = df.groupby("district")["updates_count"].sum()

    for d,o, l in load.items():
        if l > 12:
            recs.append(f"Open new Aadhaar centres in {d}")

    women = df[(df["gender"] == 0) & (df["update_pred"] == 1)]

    for d in women["district"].unique():
        recs.append(f"Run women Aadhaar update drive in {d}")

    for c in df[df["fraud_pred"] == 1]["center_id"].unique():
        recs.append(f"Audit Aadhaar centre {c}")

    return jsonify(recs)


# ------------------ RUN SERVER ------------------

if __name__ == "__main__":
    app.run(debug=True)
