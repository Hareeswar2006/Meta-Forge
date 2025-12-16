import pandas as pd
import numpy as np
import os
import sys
import json
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

class RegressionPreprocessor:

    def __init__(self, target_column, dataset_id="default"):
        self.target_column = target_column
        self.dataset_id = dataset_id
        self.numeric_cols = []
        self.cat_cols = []
        self.dropped_cols = set()
        self.numeric_medians = {}
        self.outlier_caps = {}
        self.log_cols = [] 
        self.cat_impute_values = {} 
        self.encoder = None
        self.scaler = StandardScaler()
        self.final_feature_names = []
        self.n_rows_before = 0
        self.n_rows_after = 0


    def fit(self, df):
        df = df.copy()
        self.n_rows_before = len(df)

        # 1. DROP ROWS WITH MISSING TARGET
        df.dropna(subset=[self.target_column], inplace=True)

        # 2. DETECT ID / INDEX / ARTIFACT COLUMNS
        ID_KEYWORDS = ["id", "index", "row", "level", "unnamed"]

        for col in df.columns:
            col_lower = col.lower()

            # Case 1: obvious index artifacts
            if col_lower.startswith("unnamed") or col_lower in ["index", "level_0"]:
                self.dropped_cols.add(col)
                continue

            # Case 2: ID-like name + numeric behavior
            if any(k in col_lower for k in ID_KEYWORDS):
                if pd.api.types.is_numeric_dtype(df[col]):
                    unique_ratio = df[col].nunique() / len(df)

                    if unique_ratio > 0.9:
                        if df[col].is_monotonic_increasing or df[col].is_monotonic_decreasing:
                            self.dropped_cols.add(col)


            if "id" in col_lower and df[col].nunique() / len(df) > 0.9:
                self.dropped_cols.add(col)

        # APPLY DROPS
        df = df.drop(columns=list(self.dropped_cols), errors="ignore")
        self.n_rows_after = len(df)

        # 3. SEPARATE FEATURES
        feature_df = df.drop(columns=[self.target_column], errors="ignore")

        self.numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
        self.cat_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()

        # 4. NUMERIC STATS
        for col in self.numeric_cols:
            median = df[col].median()
            self.numeric_medians[col] = median

            temp = df[col].fillna(median)
            lower = temp.quantile(0.01)
            upper = temp.quantile(0.99)
            self.outlier_caps[col] = (lower, upper)

            capped = np.clip(temp, lower, upper)
            if capped.skew() > 1 or capped.skew() < -1:
                self.log_cols.append(col)

        # 5. CATEGORICAL STATS
        cols_to_remove = []
        for col in self.cat_cols:
            unique_ratio = df[col].nunique() / len(df)
            try:
                top_freq = df[col].value_counts(normalize=True).iloc[0]
            except IndexError:
                top_freq = 0

            if unique_ratio > 0.2 or top_freq > 0.95:
                self.dropped_cols.add(col)
                cols_to_remove.append(col)
            else:
                self.cat_impute_values[col] = (
                    df[col].mode()[0] if not df[col].mode().empty else "Missing"
                )

        self.cat_cols = [c for c in self.cat_cols if c not in cols_to_remove]

        # 6. FIT ENCODER
        if self.cat_cols:
            temp_cat = df[self.cat_cols].copy()
            for col in self.cat_cols:
                temp_cat[col] = temp_cat[col].fillna(self.cat_impute_values[col]).astype(str)

            self.encoder = OneHotEncoder(
                drop="first",
                sparse_output=False,
                handle_unknown="ignore",
                dtype=int
            )
            self.encoder.fit(temp_cat)

        # 7. BUILD FINAL FEATURE MATRIX
        temp_df = df.copy()
        temp_df = temp_df.drop(columns=[self.target_column], errors="ignore")

        for col in self.numeric_cols:
            temp_df[col] = temp_df[col].fillna(self.numeric_medians[col])
            low, high = self.outlier_caps[col]
            temp_df[col] = np.clip(temp_df[col], low, high)
            if col in self.log_cols:
                temp_df[col] = np.sign(temp_df[col]) * np.log1p(np.abs(temp_df[col]))

        if self.cat_cols and self.encoder:
            temp_cat = temp_df[self.cat_cols].copy()
            for col in self.cat_cols:
                temp_cat[col] = temp_cat[col].fillna(self.cat_impute_values[col]).astype(str)

            encoded = self.encoder.transform(temp_cat)
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.encoder.get_feature_names_out(self.cat_cols),
                index=temp_df.index
            )

            temp_df = pd.concat([temp_df.drop(columns=self.cat_cols), encoded_df], axis=1)

        self.final_feature_names = temp_df.columns.tolist()

        # 8. FIT SCALER
        self.scaler.fit(temp_df)


    def transform(self, df):
        df = df.copy()

        # 1. DROP TARGET NA (IF PRESENT)
        if self.target_column in df.columns:
            df.dropna(subset=[self.target_column], inplace=True)

        # 2. DROP LEARNED COLUMNS
        df.drop(columns=list(self.dropped_cols), errors="ignore", inplace=True)

        # 3. NUMERIC TRANSFORMS
        for col in self.numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(self.numeric_medians[col])
                low, high = self.outlier_caps[col]
                df[col] = np.clip(df[col], low, high)
                if col in self.log_cols:
                    df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

        # 4. CATEGORICAL TRANSFORMS
        if self.cat_cols and self.encoder:
            temp_cat = df[self.cat_cols].copy()
            for col in self.cat_cols:
                temp_cat[col] = temp_cat[col].fillna(self.cat_impute_values[col]).astype(str)

            encoded = self.encoder.transform(temp_cat)
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.encoder.get_feature_names_out(self.cat_cols),
                index=df.index
            )

            df = pd.concat([df.drop(columns=self.cat_cols), encoded_df], axis=1)

        # 5. SCALE
        df[self.final_feature_names] = self.scaler.transform(
            df[self.final_feature_names]
        )

        # 6. MOVE TARGET TO END
        if self.target_column in df.columns:
            df = df[self.final_feature_names + [self.target_column]]
        else:
            df = df[self.final_feature_names]

        return df


    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
    

    def save(self, directory="models/preprocessors"):
        os.makedirs(directory, exist_ok=True)
        base_path = os.path.join(directory, f"{self.dataset_id}")
        
        # 1. Pickle the whole object
        with open(f"{base_path}_preprocessor.pkl", "wb") as f:
            pickle.dump(self, f)
            
        # 2. JSON Summary
        meta = {
            "dataset_id": self.dataset_id,
            "target": self.target_column,
            "rows_raw": self.n_rows_before,
            "rows_clean": self.n_rows_after,
            "dropped_columns": sorted(list(self.dropped_cols)),
            "log_transformed": self.log_cols,
            "numeric_cols_count": len(self.numeric_cols),
            "categorical_cols_count": len(self.cat_cols),
            "features_out": len(self.final_feature_names) if self.final_feature_names else "Not set"
        }
        with open(f"{base_path}_preprocess_meta.json", "w") as f:
            json.dump(meta, f, indent=4)
        
        print(f"Artifacts saved to {base_path}...")
    

# def initial_preprocessing(df, output_column):
#     df = df.copy()
#     df.dropna(subset=[output_column], inplace=True)
#     id_cols_to_drop = []

#     for col in df.columns:
#         if 'id' in col.lower() and df[col].nunique() / len(df) > 0.9:
#             print(f"Dropping likely ID column: {col}")
#             id_cols_to_drop.append(col)

#     df.drop(columns=id_cols_to_drop, inplace=True)

#     return df


# def numerical_imputation(df, output_column):
#     df = df.copy()
#     num_cols = df.select_dtypes(include=["number"]).columns
#     feature_cols = [c for c in num_cols if c != output_column]

#     if len(feature_cols) > 0:
#         imputer = SimpleImputer(strategy="median")
#         df[feature_cols] = imputer.fit_transform(df[feature_cols])

#     for col in feature_cols:

#         lower_limit = df[col].quantile(0.01)
#         upper_limit = df[col].quantile(0.99)
#         df[col] = np.clip(df[col], lower_limit, upper_limit)

#         if df[col].skew() > 1 or df[col].skew() < -1:
#             df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
    
#     return df


# def categorical_imputation(df, output_column):
#     df = df.copy()
#     cols_to_drop = []

#     cat_cols = df.select_dtypes(include=["object", "category"]).columns
#     cat_cols = [c for c in cat_cols if c != output_column]

#     for col in cat_cols:
#         unique_ratio = df[col].nunique() / len(df)
#         if df[col].value_counts(normalize=True).empty:
#             top_val_freq = 0
#         else:
#             top_val_freq = df[col].value_counts(normalize=True).iloc[0]

#         if unique_ratio > 0.2:
#             cols_to_drop.append(col)
#             print(f"Dropping {col}: High Cardinality")
#         elif top_val_freq > 0.95:
#             cols_to_drop.append(col)
#             print(f"Dropping {col}: Quasi-Constant")
        
#         else:
#             missing_pct = df[col].isnull().mean()
#             if missing_pct < 0.05:
#                 if not df[col].mode().empty:
#                     most_frequent = df[col].mode()[0]
#                     df[col] = df[col].fillna(most_frequent)
#             else:
#                 df[col] = df[col].fillna("Missing")

#     if cols_to_drop:
#         df.drop(columns=cols_to_drop, inplace=True)
#         cat_cols = [c for c in cat_cols if c not in cols_to_drop]

#     if len(cat_cols)>0:

#         encoder = OneHotEncoder(drop='first', sparse_output=False, dtype=int)
#         encoded_array = encoder.fit_transform(df[cat_cols])
#         feature_names = encoder.get_feature_names_out(cat_cols)
#         encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)
#         df = pd.concat([df, encoded_df], axis=1)
#         df.drop(columns=cat_cols, inplace=True)
    
#     return df


# def final_scaling(df, output_column):

#     df = df.copy()
#     feature_cols = [c for c in df.columns if c != output_column]
    
#     if len(feature_cols) > 0:
#         scaler = StandardScaler()
#         df[feature_cols] = scaler.fit_transform(df[feature_cols])
        
#     return df
