import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def initial_preprocessing(df, output_column):
    df = df.copy()
    df.dropna(subset=[output_column], inplace=True)
    for col in df.columns:
        if 'id' in col.lower() and df[col].nunique() / len(df) > 0.9:
            print(f"Dropping likely ID column: {col}")
            df.drop(columns=[col], inplace=True)

    return df


def numerical_imputation(df, output_column):
    df = df.copy()
    num_cols = df.select_dtypes(include=["number"]).columns
    feature_cols = [c for c in num_cols if c != output_column]

    if len(feature_cols) > 0:
        imputer = SimpleImputer(strategy="median")
        df[feature_cols] = imputer.fit_transform(df[feature_cols])

    for col in feature_cols:
        if df[col].skew() > 1 or df[col].skew() < -1:
            df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
    
    return df


def categorical_imputation(df, output_column):
    df = df.copy()
    cols_to_drop = []

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    cat_cols = [c for c in cat_cols if c != output_column]

    for col in cat_cols:
        unique_ratio = df[col].nunique() / len(df)
        if df[col].value_counts(normalize=True).empty:
            top_val_freq = 0
        else:
            top_val_freq = df[col].value_counts(normalize=True).iloc[0]

        if unique_ratio > 0.2:
            cols_to_drop.append(col)
            print(f"Dropping {col}: High Cardinality")
        elif top_val_freq > 0.95:
            cols_to_drop.append(col)
            print(f"Dropping {col}: Quasi-Constant")
        
        else:
            missing_pct = df[col].isnull().mean()
            if missing_pct < 0.05:
                if not df[col].mode().empty:
                    most_frequent = df[col].mode()[0]
                    df[col] = df[col].fillna(most_frequent)
            else:
                df[col] = df[col].fillna("Missing")

    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
    
    return df