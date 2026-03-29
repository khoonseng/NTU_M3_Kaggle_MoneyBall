import pandas as pd
from sklearn.model_selection import train_test_split

from src.pipelines.preprocessing import build_preprocessor
from src.models.prediction import run_prediction
from scripts.train_model import train_model

def main():
    # -------------------------
    # 1. Load data
    # -------------------------
    data_df = pd.read_csv("data/data.csv")
    predict_df = pd.read_csv("data/predict.csv")

    # Display basic information about the datasets
    print(f"Data set shape: {data_df.shape}")
    print(f"Predict set shape: {predict_df.shape}")

    # Select only the default features from DATA_DESCRIPTION.md
    default_features = [
        # Basic Statistics
        'G', 'R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB', 'CS', 'HBP', 'SF',
        'RA', 'ER', 'ERA', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA', 'SOA',
        'E', 'DP', 'FP', 'attendance', 'BPF', 'PPF',
        
        # Derived Features
        'R_per_game', 'RA_per_game', 'mlb_rpg',
        
        # Era Indicators
        'era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8',
        
        # Decade Indicators
        'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950',
        'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990', 'decade_2000', 'decade_2010'
    ]

    # Filter features that exist in both datasets
    available_features = [col for col in default_features if col in data_df.columns and col in predict_df.columns]
    print(f"Number of available default features: {len(available_features)}")

    # Separate features and target variable
    X = data_df[available_features]
    y = data_df['W']
    print(X.dtypes)

    # Perform the split (adjust test_size / random_state as needed)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,      # 20% for testing
        random_state=42    # ensures reproducibility
    )

    # -------------------------
    # 2. Preprocessing
    # -------------------------
    # Scale features
    # Identify columns to exclude from scaling (one-hot encoded and label columns)
    one_hot_cols = [col for col in X_train.columns if col.startswith(('era_', 'decade_'))]
    other_cols = [col for col in X_train.columns if col not in one_hot_cols]

    preprocessor = build_preprocessor(other_cols, None)

    # -------------------------
    # 3. Baseline: Linear Regression
    # -------------------------
    lr_pipeline = train_model(
        model_name="linear", 
        preprocessor=preprocessor, 
        X_train=X_train, 
        y_train=y_train, 
        X_test=X_test, 
        y_test=y_test,
        use_GridSearch=False
    )
    run_prediction(predict_df, available_features, lr_pipeline)

    # -------------------------
    # 4. Lasso with GridSearchCV
    # -------------------------
    lasso_pipeline = train_model(
        model_name="lasso", 
        preprocessor=preprocessor, 
        X_train=X_train, 
        y_train=y_train, 
        X_test=X_test, 
        y_test=y_test,
        use_GridSearch=True
    )
    run_prediction(predict_df, available_features, lasso_pipeline)

    # -------------------------
    # 5. Ridge with GridSearchCV
    # -------------------------
    ridge_pipeline = train_model(
        model_name="ridge", 
        preprocessor=preprocessor, 
        X_train=X_train, 
        y_train=y_train, 
        X_test=X_test, 
        y_test=y_test,
        use_GridSearch=True
    )
    run_prediction(predict_df, available_features, ridge_pipeline)

if __name__ == "__main__":
    main()