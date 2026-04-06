import pandas as pd
from sklearn.model_selection import train_test_split

from src.pipelines.preprocessing import build_linear_preprocessor, build_tree_preprocessor, build_pca_preprocessor
from src.models.train import run_all_models, run_ensemble
from src.models.prediction import run_prediction
from src.config.config import CONFIG

def add_advanced_metrics(df):
    df_advanced = df.copy()

    # Normalizing 9 offensive stats per game
    df_advanced['R_per_G'] = df_advanced['R'] / df_advanced['G']
    df_advanced['AB_per_G'] = df_advanced['AB'] / df_advanced['G']    
    df_advanced['H_per_G'] = df_advanced['H'] / df_advanced['G']
    df_advanced['2B_per_G'] = df_advanced['2B'] / df_advanced['G']
    df_advanced['3B_per_G'] = df_advanced['3B'] / df_advanced['G']
    df_advanced['HR_per_G'] = df_advanced['HR'] / df_advanced['G']
    df_advanced['BB_per_G'] = df_advanced['BB'] / df_advanced['G']
    df_advanced['SO_per_G'] = df_advanced['SO'] / df_advanced['G']
    df_advanced['SB_per_G'] = df_advanced['SB'] / df_advanced['G']

    # Normalizing 10 pitching stats per game
    df_advanced['RA_per_G'] = df_advanced['RA'] / df_advanced['G']
    df_advanced['ER_per_G'] = df_advanced['ER'] / df_advanced['G']
    df_advanced['CG_per_G'] = df_advanced['CG'] / df_advanced['G']
    df_advanced['SHO_per_G'] = df_advanced['SHO'] / df_advanced['G']
    df_advanced['SV_per_G'] = df_advanced['SV'] / df_advanced['G']
    df_advanced['IPouts_per_G'] = df_advanced['IPouts'] / df_advanced['G']
    df_advanced['HA_per_G'] = df_advanced['HA'] / df_advanced['G']
    df_advanced['HRA_per_G'] = df_advanced['HRA'] / df_advanced['G']
    df_advanced['BBA_per_G'] = df_advanced['BBA'] / df_advanced['G']
    df_advanced['SOA_per_G'] = df_advanced['SOA'] / df_advanced['G']

    # Normalizing 2 defensive stats per game
    df_advanced['E_per_G'] = df_advanced['E'] / df_advanced['G']
    df_advanced['DP_per_G'] = df_advanced['DP'] / df_advanced['G']

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
    # normalize per-game stats
    df_advanced['1B_per_G'] = df_advanced['1B'] / df_advanced['G'] 

    # Step 2: Calculate Total Bases (the numerator for SLG)
    df_advanced['Total_Bases'] = df_advanced['1B'] + (2 * df_advanced['2B']) + (3 * df_advanced['3B']) + (4 * df_advanced['HR'])
    df_advanced['Total_Bases_per_G'] = df_advanced['Total_Bases'] / df_advanced['G']

    # Step 3: Calculate Slugging Percentage (SLG)
    df_advanced['SLG'] = df_advanced['Total_Bases'] / df_advanced['AB']

    # Step 4: Calculate On-Base Percentage (OBP) 
    df_advanced['OBP'] = (df_advanced['H'] + df_advanced['BB']) / (df_advanced['AB'] + df_advanced['BB'])

    # Step 5: Calculate On-Base Plus Slugging (OPS)
    df_advanced['OPS'] = df_advanced['OBP'] + df_advanced['SLG']

    # Interaction between pitching and defense
    df_advanced['WHIP_x_FP'] = df_advanced['WHIP'] * df_advanced['FP'] # Fielding Percentage

    # Interaction between offense and era-normalization
    df_advanced['offense_index'] = df_advanced['R_per_G'] / df_advanced['mlb_rpg']
    df_advanced['OPS_x_offense_index'] = df_advanced['OPS'] * df_advanced['offense_index']

    # df_advanced['weighted_production'] = (3 * df_advanced['OBP']) + df_advanced['SLG']

    # # 1. Sort the data by Team and Year to ensure chronological order
    # df_advanced = df_advanced.sort_values(by=['teamID', 'yearID'])

    # # 2. Create 'Lag' features (the value from the previous season)
    # # groupby ensures that the first year of a team's existence results in NaN
    # df_advanced['OPS_lag1'] = df_advanced.groupby('teamID')['OPS'].shift(1)
    # df_advanced['ERA_lag1'] = df_advanced.groupby('teamID')['ERA'].shift(1)

    # # 3. Calculate the Delta (Current Year - Previous Year)
    # # Positive delta for OPS indicates offensive improvement [1]
    # df_advanced['OPS_delta_yoy'] = df_advanced['OPS'] - df_advanced['OPS_lag1']

    # # Negative delta for ERA indicates pitching improvement (fewer runs allowed)
    # df_advanced['ERA_delta_yoy'] = df_advanced['ERA'] - df_advanced['ERA_lag1']

    # Example for franchise-level statistics
    # franchise_stats = df_advanced.groupby('franchID').agg({
    #                                     'R': ['mean', 'std', 'min', 'max'],
    #                                     'RA': ['mean', 'std'],
    #                                     'OPS': ['mean', 'std']
    #                                 })
    # # Flatten the multi-level column names
    # franchise_stats.columns = ['_'.join(col).strip() for col in franchise_stats.columns.values]
    # # Merge these new features back into the main DataFrame
    # df_advanced = df_advanced.merge(franchise_stats, on='franchID', how='left')


    # Remove original features after adding metrics [Leave the removal of features to Lasso instead of manually removing them]
    # original_offensive_features = ['R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB']
    # original_pitching_features = ['RA', 'ER', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA', 'SOA']
    # original_defensive_features = ['E', 'DP']
    # original_features = original_offensive_features + original_pitching_features + original_defensive_features
    # df_advanced = df_advanced.drop(original_features, axis=1)

    return df_advanced

def add_era_metrics(df):
    df_advanced = df.copy()
    era_cols = [col for col in df_advanced.columns if col.startswith("era_")]
    df_advanced["era_label"] = df_advanced[era_cols].idxmax(axis=1)
    assert (df_advanced[era_cols].sum(axis=1) == 1).all(), "Invalid era encoding detected!"

    # power and speed factor
    df_advanced['era_power_factor'] = df_advanced['HR'] / df_advanced['H']
    df_advanced['era_speed_factor'] = df_advanced['SB'] / df_advanced['R']
 
    return df_advanced

def main():
    # -------------------------
    # 1. Load data
    # -------------------------
    data_df = pd.read_csv("data/data_year_team_franchise.csv")
    predict_df = pd.read_csv("data/predict_year_team_franchise.csv")

    # Display basic information about the datasets
    print(f"Data set shape: {data_df.shape}")
    print(f"Predict set shape: {predict_df.shape}")

    # Select only the default features from DATA_DESCRIPTION.md
    default_features = [
        # Basic Statistics
        'G', 'R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB', 'CS', 'HBP', 'SF',
        'RA', 'ER', 'ERA', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA', 'SOA',
        'E', 'DP', 'FP', 'attendance', 'BPF', 'PPF', 'yearID', 'teamID', 'franchID',
        
        # Derived Features
        'R_per_game', 'RA_per_game', 'mlb_rpg',
        
        # Era Indicators
        'era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8',
        
        # Decade Indicators
        'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950',
        'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990', 'decade_2000', 'decade_2010'
    ]

    # Advanced metrics
    advanced_metrics = ['IP', 'WHIP', 'BA', 'OBP', 'SLG', 'OPS']
    offensive_per_game_metrics = ['R_per_G', 'AB_per_G', 'H_per_G', '2B_per_G', '3B_per_G', 'HR_per_G', 'BB_per_G', 
                                  'SO_per_G', 'SB_per_G', '1B_per_G', 'Total_Bases_per_G', 'WHIP_x_FP', 'offense_index', 
                                  'OPS_x_offense_index', 'weighted_production', 'OPS_delta_yoy', 'ERA_delta_yoy']
    pitching_per_game_metrics = ['RA_per_G', 'ER_per_G', 'CG_per_G', 'SHO_per_G', 'SV_per_G', 'IPouts_per_G', 'HA_per_G', 
                                 'HRA_per_G', 'BBA_per_G', 'SOA_per_G'] 
    defensive_per_game_metrics = ['E_per_G', 'DP_per_G']
    # franchise_metrics = ['R_mean', 'R_std', 'R_min', 'R_max', 'RA_mean', 'RA_std', 'OPS_mean', 'OPS_std']
    era_metrics = ['era_label', 'era_power_factor', 'era_power_factor_mean', 'era_power_factor_relative', 'era_speed_factor', 'era_speed_factor_mean', 'era_speed_factor_relative']
    normalize_metrics = offensive_per_game_metrics + pitching_per_game_metrics + defensive_per_game_metrics + era_metrics

    # Combine all features
    all_features = default_features + advanced_metrics + normalize_metrics

    data_df = add_advanced_metrics(df=data_df)
    predict_df = add_advanced_metrics(df=predict_df)

    # data_df = add_era_metrics(df=data_df)
    # predict_df = add_era_metrics(df=predict_df)

    # Remove unused columns
    unused_cols = [col for col in data_df.columns if col.startswith(('teamID', 'decade_label'))]
    data_df = data_df.drop(unused_cols, axis=1)
    era_decade_cols = ['era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8',
                        'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950',
                        'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990', 'decade_2000', 'decade_2010']
    data_df.drop(columns=era_decade_cols, inplace=True)

    # Filter features that exist in both datasets
    available_features = [col for col in all_features if col in data_df.columns and col in predict_df.columns]
    print(f"Total number of features: {len(available_features)}")
    print(f"Number of default features: {len([c for c in default_features if c in available_features])}")
    print(f"Number of advanced metrics: {len([c for c in advanced_metrics if c in available_features])}")
    print(f"Number of normalized metrics: {len([c for c in normalize_metrics if c in available_features])}")

    # convert boolean to integers
    # bool_cols = [col for col in available_features if col.startswith(('era_', 'decade_'))]
    # data_df[bool_cols] = data_df[bool_cols].astype(int)
    # predict_df[bool_cols] = predict_df[bool_cols].astype(int)

    # print(data_df.head())

    # Separate features and target variable
    X = data_df[available_features]
    y = data_df['W']
    # print(X.info())
    # import sys
    # sys.exit("test...")
    
    # Perform the split (adjust test_size / random_state as needed)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,      # 20% for testing
        random_state=42    # ensures reproducibility
    )

    # X_train["era_power_factor_mean"] = X_train.groupby("era_label")["era_power_factor"].transform("mean")
    # X_train["era_power_factor_relative"] = X_train['era_power_factor'] / X_train["era_power_factor_mean"]
    # X_train["era_speed_factor_mean"] = X_train.groupby("era_label")["era_speed_factor"].transform("mean")
    # X_train["era_speed_factor_relative"] = X_train['era_speed_factor'] / X_train["era_speed_factor_mean"]
    # X_test["era_power_factor_mean"] = None
    # X_test["era_power_factor_relative"] = None
    # X_test["era_speed_factor_mean"] = None
    # X_test["era_speed_factor_relative"] = None
    # predict_df["era_power_factor_mean"] = None
    # predict_df["era_power_factor_relative"] = None
    # predict_df["era_speed_factor_mean"] = None
    # predict_df["era_speed_factor_relative"] = None
    # available_features += ["era_power_factor_mean", "era_power_factor_relative", "era_speed_factor_mean", "era_speed_factor_relative"]

    # Example for franchise-level statistics
    franchise_stats = X_train.groupby('franchID').agg({
                                        'OPS': ['mean']
                                    })
    # # Flatten the multi-level column names
    # franchise_stats.columns = ['_'.join(col).strip() for col in franchise_stats.columns.values]
    # Merge these new features back into the main DataFrame
    # X_train = X_train.merge(franchise_stats, on='franchID', how='left')
    # X_train["relative_OPS"] = X_train["OPS"] / X_train["OPS_mean"]
    # X_test["OPS_mean"] = None
    # X_test["relative_OPS"] = None
    # predict_df["OPS_mean"] = None
    # predict_df["relative_OPS"] = None
    # available_features += ["OPS_mean", "relative_OPS"]
    
    # print(X_train.info())
    # import sys
    # sys.exit("test...")

    # -------------------------
    # 2. Preprocessing
    # -------------------------
    # Scale features
    # Identify columns to exclude from scaling (one-hot encoded and label columns)
    other_cols = [col for col in X_train.columns if col.startswith(('franchID', 'teamID', 'decade_label', 'era_label'))]
    categorical_cols = [col for col in X_train.columns if col.startswith(('era_', 'decade_')) and col not in other_cols and col not in era_metrics]
    numerical_cols = [col for col in X_train.columns if col not in categorical_cols and col not in other_cols]
    # print(f"other_cols:", other_cols)
    # print(f"categorical_cols:", categorical_cols)
    # print(f"numerical_cols:", numerical_cols)
    # print(X_train[numerical_cols].info())
    # import sys
    # sys.exit("test...")

    # preprocessor = build_preprocessor(num_features=numerical_cols, cat_features=categorical_cols)
    preprocessors = {
        "linear": build_linear_preprocessor(num_features=numerical_cols, cat_features=None),
        "tree": build_tree_preprocessor(num_features=numerical_cols, cat_features=None),
        "pca": build_pca_preprocessor(num_features=numerical_cols, cat_features=None)
    }

    # -------------------------
    # 3. Run pipelines and predictions
    # -------------------------
    trained_pipelines = run_all_models(
        preprocessors=preprocessors, 
        X_train=X_train, 
        y_train=y_train, 
        X_test=X_test, 
        y_test=y_test,
        predict_df=predict_df,
        available_features=available_features
    )
    # print(trained_pipelines)

    ensemble_config = CONFIG["ensemble"]
    if ensemble_config["enabled"]:
        # Number of features for meta-model is simply the number of base model predictions
        n_features = len(ensemble_config["base_models"])
        ensemble_model = run_ensemble(
            X_train=X_train, 
            y_train=y_train, 
            X_test=X_test, 
            y_test=y_test, 
            config=ensemble_config, 
            n_features=n_features
        )

        run_prediction(
            predictions=predict_df, 
            features=available_features, 
            pipeline=ensemble_model)

if __name__ == "__main__":
    main()