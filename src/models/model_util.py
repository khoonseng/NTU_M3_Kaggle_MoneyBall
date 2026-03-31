import joblib
import os

def save_model(model, model_name, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{model_name}.pkl")
    joblib.dump(model, path)

def load_model(model_name, model_dir):
    path = os.path.join(model_dir, f"{model_name}.pkl")
    return joblib.load(path)