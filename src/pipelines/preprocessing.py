from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA

def convert_bool_to_int(df):
    # Ensure we are working on a copy to avoid SettingWithCopy warnings
    return df.astype(int)

def build_linear_preprocessor(num_features, cat_features=None):
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('to_int', FunctionTransformer(func=convert_bool_to_int, feature_names_out="one-to-one")),
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

def build_tree_preprocessor(num_features, cat_features=None):
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ('to_int', FunctionTransformer(func=convert_bool_to_int, feature_names_out="one-to-one")),
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

def build_pca_preprocessor(num_features, cat_features, n_components=0.95):
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, svd_solver="full", random_state=42))
    ])

    categorical_pipeline = Pipeline([
        ('to_int', FunctionTransformer(func=convert_bool_to_int, feature_names_out="one-to-one")),
        ("imputer", SimpleImputer(strategy="most_frequent"))
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