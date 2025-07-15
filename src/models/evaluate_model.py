# src/models/evaluate_model.py

import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import json

# Load test data (CSV format)
X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv").values.ravel()

# Load trained model
model = joblib.load("models/model.pkl")

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Save metrics
metrics = {"mse": mse, "r2": r2}
with open("metrics/scores.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save predictions
pred_df = pd.DataFrame({
    "y_true": y_test,
    "y_pred": predictions
})
pred_df.to_csv("data/predictions.csv", index=False)
