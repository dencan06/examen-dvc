# src/models/train_model.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load scaled training data
X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").values.ravel()

# Load best parameters
best_params = joblib.load("models/best_params.pkl")

# Train model
model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "models/model.pkl")
