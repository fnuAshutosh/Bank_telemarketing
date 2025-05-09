import joblib
from pathlib import Path

# Get project root
project_root = Path(__file__).parent.parent.absolute()

# Load the model
model_path = project_root / "models" / "xgboost_final.joblib"
model = joblib.load(model_path)

# Print feature names used during training
if hasattr(model, 'feature_names_'):
    print("Model feature names:", model.feature_names_)
else:
    print("Model doesn't have feature_names_ attribute")
    
# If using XGBoost, we can also check feature_names directly
if hasattr(model, 'get_booster'):
    booster = model.get_booster()
    if hasattr(booster, 'feature_names'):
        print("Booster feature names:", booster.feature_names)