import pandas as pd
from datetime import datetime
import os

def log_experiment(results, params, model_name, cv_results=None, filepath="experiments/results.csv"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    record = {
        "timestamp": datetime.now(),
        "model": model_name,
        **params,
        **results
    }

    # Add best CV scores if available
    if cv_results:
        for metric, value in cv_results.items():
            record[f"cv_best_{metric}"] = value

    df = pd.DataFrame([record])

    if not os.path.exists(filepath):
        df.to_csv(filepath, index=False)
    else:
        df.to_csv(filepath, mode="a", header=False, index=False)