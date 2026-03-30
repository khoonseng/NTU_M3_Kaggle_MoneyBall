import pandas as pd
from sklearn.model_selection import train_test_split

from src.pipelines.preprocessing import build_preprocessor
from src.models.train import run_all_models

def add_advanced_metrics(df):
    df_advanced = df.copy()

    # Normalizing Runs (R), Home Runs (HR) and Strikeouts (SO) per game
    df_advanced['R_per_G'] = df_advanced['R'] / df_advanced['G']
    df_advanced['HR_per_G'] = df_advanced['HR'] / df_advanced['G']
    df_advanced['SO_per_G'] = df_advanced['SO'] / df_advanced['G']

    # 2. Inning-based Pitching Metrics
    # First, convert IPouts to fractional Innings Pitched (IP)
    df_advanced['IP'] = df_advanced['IPouts'] / 3

    # Compute WHIP: (Walks Allowed + Hits Allowed) / Innings Pitched
    # Column keys: BBA (Walks Allowed), HA (Hits Allowed)
    df_advanced['WHIP'] = (df_advanced['BBA'] + df_advanced['HA']) / df_advanced['IP']

    # Compute/Verify ERA: (Earned Runs * 9) / Innings Pitched
    # Column keys: ER (Earned Runs Allowed)
    # df_advanced['computed_ERA'] = (df_advanced['ER'] * 9) / df_advanced['IP']

    # 3. Efficiency Metrics (Ratio-based, not game-based)
    # Batting Average (BA): Hits divided by At Bats
    df_advanced['BA'] = df_advanced['H'] / df_advanced['AB']

    # Steps to calculate OPS (OBP + SLG)
    # Step 1: Calculate Singles (optional but good for visibility)
    df_advanced['1B'] = df_advanced['H'] - (df_advanced['2B'] + df_advanced['3B'] + df_advanced['HR'])

    # Step 2: Calculate Total Bases (the numerator for SLG)
    df_advanced['Total_Bases'] = df_advanced['1B'] + (2 * df_advanced['2B']) + (3 * df_advanced['3B']) + (4 * df_advanced['HR'])

    # Step 3: Calculate Slugging Percentage (SLG)
    df_advanced['SLG'] = df_advanced['Total_Bases'] / df_advanced['AB']

    # Step 4: Calculate On-Base Percentage (OBP) 
    df_advanced['OBP'] = (df_advanced['H'] + df_advanced['BB']) / (df_advanced['AB'] + df_advanced['BB'])

    # Step 5: Calculate On-Base Plus Slugging (OPS)
    df_advanced['OPS'] = df_advanced['OBP'] + df_advanced['SLG']

    # Remove original features after adding metrics
    # original_features = ['R', 'HR', 'SO', 'IPouts', 'BBA', 'HA', 'ER', 'H', 'AB', '2B', '3B', 'BB']
    # df_advanced = df_advanced.drop(original_features, axis=1)

    return df_advanced

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

    # Advanced metrics
    advanced_metrics = ['R_per_G', 'HR_per_G', 'SO_per_G', 'IP', 'WHIP', 'BA', 'OBP', 'SLG', 'OPS']

    # Combine all features
    all_features = default_features + advanced_metrics

    data_df = add_advanced_metrics(data_df)
    predict_df = add_advanced_metrics(predict_df)

    # Filter features that exist in both datasets
    available_features = [col for col in all_features if col in data_df.columns and col in predict_df.columns]
    print(f"Total number of features: {len(available_features)}")
    print(f"Number of default features: {len([c for c in default_features if c in available_features])}")
    print(f"Number of advanced metrics: {len([c for c in advanced_metrics if c in available_features])}")

    # convert boolean to integers
    # bool_cols = [col for col in available_features if col.startswith(('era_', 'decade_'))]
    # data_df[bool_cols] = data_df[bool_cols].astype(int)
    # predict_df[bool_cols] = predict_df[bool_cols].astype(int)

    # Separate features and target variable
    X = data_df[available_features]
    y = data_df['W']
    # print("X data types:\n", X.dtypes)

    # Remove era/decade columns
    # one_hot_cols = [col for col in X.columns if col.startswith(('era_', 'decade_'))]
    # X = X.drop(one_hot_cols, axis=1)
    # print("X data types:\n", X.dtypes)

    # import sys
    # sys.exit("test...")

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

    # preprocessor = build_preprocessor(num_features=other_cols, cat_features=one_hot_cols)
    preprocessor = build_preprocessor(num_features=other_cols, cat_features=None)

    # -------------------------
    # 3. Run pipelines and predictions
    # -------------------------
    trained_pipelines = run_all_models(
        preprocessor=preprocessor, 
        X_train=X_train, 
        y_train=y_train, 
        X_test=X_test, 
        y_test=y_test,
        predict_df=predict_df,
        available_features=available_features
    )
    # print(trained_pipelines)

if __name__ == "__main__":
    main()