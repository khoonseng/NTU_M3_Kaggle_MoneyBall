import pandas as pd
import numpy as np
import os
from pathlib import Path

def run_prediction(predictions, features, pipeline):
    # Prepare Submission
    predict_X = predictions[features].copy()
    predict_preds = pipeline.predict(predict_X)

    if hasattr(pipeline, "named_steps"):
        # Get the last step's estimator
        final_estimator = pipeline.steps[-1][1]  # steps is a list of (name, estimator) tuples

        # Get the model's class name
        model_name = final_estimator.__class__.__name__
    else:
        model_name = "ensemble"
    
    # Build submission in the same format as submission.csv
    submission_df = pd.DataFrame({
        'ID': predictions['ID'],
        'W': np.round(predict_preds).astype(int)
    })

    PROJECT_ROOT = Path(os.environ.get("LOCAL_DATA_DIR", Path.cwd())).resolve()    
    LOCAL_SUBMISSION_FILE = PROJECT_ROOT.home().name + "_submission_predict_" + model_name + ".csv"
    submission_path = PROJECT_ROOT / "submission" / LOCAL_SUBMISSION_FILE
    submission_df.to_csv(submission_path, index=False)
    print(f"Kaggle submission saved to {submission_path}")