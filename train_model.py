import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import matplotlib.pyplot as plt

import joblib

# =========================
# 1. LOAD DATA
# =========================

train_df = pd.read_csv("train_car.csv")
test_df = pd.read_csv("test_car.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# =========================
# 2. SPLIT X AND Y
# =========================

X_train = train_df.drop("Price_log", axis=1)
y_train = train_df["Price_log"]

X_test = test_df.drop("Price_log", axis=1)
y_test = test_df["Price_log"]

# =========================
# 3. LINEAR REGRESSION
# =========================

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

# =========================
# 4. RANDOM FOREST
# =========================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

# =========================
# 5. XGBOOST
# =========================

xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

# =========================
# 6. EVALUATION FUNCTION
# =========================

def evaluate_model(name, y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(mean_squared_error(
    y_true,
    y_pred
))

    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}")
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R2  :", r2)

    return mae, rmse, r2

# =========================
# 7. EVALUATE ALL MODELS
# =========================

linear_mae, linear_rmse, linear_r2 = evaluate_model(
    "Linear Regression",
    y_test,
    linear_pred
)

rf_mae, rf_rmse, rf_r2 = evaluate_model(
    "Random Forest",
    y_test,
    rf_pred
)

xgb_mae, xgb_rmse, xgb_r2 = evaluate_model(
    "XGBoost",
    y_test,
    xgb_pred
)

# =========================
# 8. COMPARE RESULTS
# =========================

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "XGBoost"
    ],
    "MAE": [
        linear_mae,
        rf_mae,
        xgb_mae
    ],
    "RMSE": [
        linear_rmse,
        rf_rmse,
        xgb_rmse
    ],
    "R2 Score": [
        linear_r2,
        rf_r2,
        xgb_r2
    ]
})

print("\n====================")
print("MODEL COMPARISON")
print("====================")

print(results)

# =========================
# 9. SAVE BEST MODEL
# =========================

joblib.dump(xgb_model, "car_price_model.pkl")

print("\nModel saved successfully!")

# =========================
# 10. FEATURE IMPORTANCE
# =========================

importance = xgb_model.feature_importances_

features = X_train.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP IMPORTANT FEATURES")
print(importance_df.head(10))

# =========================
# 11. PLOT FEATURE IMPORTANCE
# =========================

top_features = importance_df.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title("Top 10 Feature Importance")

plt.gca().invert_yaxis()

plt.show()

# =========================
# 12. ACTUAL VS PREDICTED
# =========================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, xgb_pred)

plt.xlabel("Actual Price_log")
plt.ylabel("Predicted Price_log")

plt.title("Actual vs Predicted")

plt.show()

# =========================
# 13. PREDICT REAL PRICE
# =========================

sample_prediction_log = xgb_pred[0]

sample_real_price = np.exp(sample_prediction_log)

print("\nExample Real Price Prediction:")
print(sample_real_price)