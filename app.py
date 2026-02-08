from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder="templates")

# --------------------------------------------------
# Load trained model safely
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"model.joblib not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# Home page – input form
# --------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# --------------------------------------------------
# Prediction logic
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect inputs from form
        year = float(request.form["year"])
        annual_cost = float(request.form["annual_cost_healthy_diet_usd"])
        veg_cost = float(request.form["cost_vegetables_ppp_usd"])
        fruit_cost = float(request.form["cost_fruits_ppp_usd"])
        total_food_cost = float(request.form["total_food_components_cost"])

        # Arrange inputs EXACTLY as training order
        features = np.array([[
    year,
    annual_cost,
    veg_cost,
    fruit_cost,
    total_food_cost
]])



        # Prediction
        prediction = model.predict(features)

        return render_template(
            "result.html",
            prediction=round(prediction[0], 2)
        )

    except Exception as e:
        return f"Prediction Error: {e}"

# --------------------------------------------------
# Run app
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
