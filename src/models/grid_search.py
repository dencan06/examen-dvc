# src/models/grid_search.py

import pandas as pd
import yaml
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Load data
X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").values.ravel()

# Load parameters from YAML
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

grid_params = params["grid_search"]["param_grid"]

# Replace YAML null with Python None if needed
for key in grid_params:
    grid_params[key] = [None if v is None else v for v in grid_params[key]]

# Initialize model
model = RandomForestRegressor(random_state=42)

# Grid Search
grid_search = GridSearchCV(model, grid_params, cv=5, scoring="r2", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Save best params
joblib.dump(grid_search.best_params_, "models/best_params.pkl")