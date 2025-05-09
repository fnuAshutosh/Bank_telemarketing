import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os

# Get the project root directory
project_root = Path(__file__).parent.parent.absolute()

# Load the model to check its features
model_path = os.path.join(project_root, 'models', 'xgboost_final.joblib')
model = joblib.load(model_path)

# Get expected feature names
if hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'feature_names'):
    expected_features = model.get_booster().feature_names
    print(f"Found {len(expected_features)} features in model:")
    print(expected_features)

# Save the exact feature names to disk
features_path = os.path.join(project_root, 'models', 'exact_features.txt')
with open(features_path, 'w') as f:
    for feature in expected_features:
        f.write(f"{feature}\n")
print(f"Saved exact feature names to {features_path}")