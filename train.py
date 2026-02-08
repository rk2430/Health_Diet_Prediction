import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------
# 1. Load dataset
# ---------------------------------------
df = pd.read_csv("data.csv")

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

print("Columns:", df.columns)

# ---------------------------------------
# 2. Select required columns
# ---------------------------------------
required_columns = [
    "year",
    "annual_cost_healthy_diet_usd",
    "cost_vegetables_ppp_usd",
    "cost_fruits_ppp_usd",
    "total_food_components_cost",
    "cost_healthy_diet_ppp_usd"
]

df = df[required_columns]

# ---------------------------------------
# 3. Check NaN values BEFORE filling
# ---------------------------------------
print("\nNaN values before filling:")
print(df.isna().sum())

# ---------------------------------------
# 4. Fill NaN values (NUMERIC COLUMNS)
# ---------------------------------------
df.fillna(df.mean(numeric_only=True), inplace=True)

# ---------------------------------------
# 5. Check NaN values AFTER filling
# ---------------------------------------
print("\nNaN values after filling:")
print(df.isna().sum())

# ---------------------------------------
# 6. Split features and target
# ---------------------------------------
X = df[
    [
        "year",
        "annual_cost_healthy_diet_usd",
        "cost_vegetables_ppp_usd",
        "cost_fruits_ppp_usd",
        "total_food_components_cost"
    ]
]

y = df["cost_healthy_diet_ppp_usd"]

# ---------------------------------------
# 7. Train-test split
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------
# 8. Train model
# ---------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel expects features:", model.n_features_in_)

# ---------------------------------------
# 9. Evaluate model
# ---------------------------------------
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# ---------------------------------------
# 10. Save model
# ---------------------------------------
joblib.dump(model, "model.joblib")
print("\nmodel.joblib saved successfully")
