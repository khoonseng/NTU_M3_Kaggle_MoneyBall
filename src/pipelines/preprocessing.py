from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer

def convert_bool_to_int(df):
    # Ensure we are working on a copy to avoid SettingWithCopy warnings
    return df.astype(int)

def build_preprocessor(num_features, cat_features=None):
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('to_int', FunctionTransformer(convert_bool_to_int, feature_names_out="one-to-one")),
        ("imputer", SimpleImputer(strategy="most_frequent"))
        # ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    if cat_features is not None:
        preprocessor = ColumnTransformer([
            ("num", numerical_pipeline, num_features),
            ("cat", categorical_pipeline, cat_features)
        ])
    else:
        preprocessor = ColumnTransformer([
            ("num", numerical_pipeline, num_features)
        ])
    
    return preprocessor