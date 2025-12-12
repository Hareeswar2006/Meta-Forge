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
        self.dropped_cols =[]
        self.numeric_medians = {}
        self.outlier_caps = {}
        self.log_cols = [] 
        self.cat_impute_values = {} 
        self.encoder = None
        self.scaler = None
        self.final_feature_names = []
        self.n_rows_before = 0
        self.n_rows_after = 0


    def fit(self, df):
        df = df.copy()
        self.n_rows_before = len(df)
        
        # 1. CLEAN TARGET & ID DROPPING LOGIC
        df.dropna(subset=[self.target_column], inplace=True)
        
        for col in df.columns:
            if 'id' in col.lower() and df[col].nunique() / len(df) > 0.9:
                self.dropped_cols.append(col)
        
        cols_to_keep = [c for c in df.columns if c not in self.dropped_cols]
        df = df[cols_to_keep]
        self.n_rows_after = len(df)

        # 2. SEPARATE TYPES
        feature_df = df.drop(columns=[self.target_column], errors='ignore')
        
        self.numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
        self.cat_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()

        # 3. NUMERIC PROCESSING
        for col in self.numeric_cols:
            median = df[col].median()
            self.numeric_medians[col] = median
            
            temp_series = df[col].fillna(median)
            
            lower = temp_series.quantile(0.01)
            upper = temp_series.quantile(0.99)
            self.outlier_caps[col] = (lower, upper)
            
            capped_series = np.clip(temp_series, lower, upper)
            if capped_series.skew() > 1 or capped_series.skew() < -1:
                self.log_cols.append(col)

        # 4. CATEGORICAL PROCESSING
        cols_to_remove_cat = []
        for col in self.cat_cols:
            unique_ratio = df[col].nunique() / len(df)
            try:
                top_val_freq = df[col].value_counts(normalize=True).iloc[0]
            except IndexError:
                top_val_freq = 0
            
            if unique_ratio > 0.2:
                self.dropped_cols.append(col)
                cols_to_remove_cat.append(col)
            elif top_val_freq > 0.95:
                self.dropped_cols.append(col)
                cols_to_remove_cat.append(col)
            else:
                if not df[col].mode().empty:
                    self.cat_impute_values[col] = df[col].mode()[0]
                else:
                    self.cat_impute_values[col] = "Missing"

        self.cat_cols = [c for c in self.cat_cols if c not in cols_to_remove_cat]
        
        if self.cat_cols:
            temp_cat = df[self.cat_cols].copy()
            for col in self.cat_cols:
                temp_cat[col] = temp_cat[col].fillna(self.cat_impute_values[col])
                
            self.encoder = OneHotEncoder(drop='first', sparse_output=False, dtype=int, handle_unknown='ignore')
            self.encoder.fit(temp_cat)

    
    def transform(self, df):
        df = df.copy()
        
        # 1. Drop Rows (Target) & IDs
        if self.target_column in df.columns:
            df.dropna(subset=[self.target_column], inplace=True)
            
        df.drop(columns=[c for c in self.dropped_cols if c in df.columns], inplace=True, errors='ignore')
        
        # 2. NUMERICS
        present_num = [c for c in self.numeric_cols if c in df.columns]
        
        for col in present_num:
            # Impute
            df[col] = df[col].fillna(self.numeric_medians.get(col, 0))
            
            # Cap
            if col in self.outlier_caps:
                lower, upper = self.outlier_caps[col]
                df[col] = np.clip(df[col], lower, upper)
            
            # Log
            if col in self.log_cols:
                df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

        # 3. CATEGORICALS
        present_cat = [c for c in self.cat_cols if c in df.columns]
        
        if present_cat:
            for col in present_cat:
                val = self.cat_impute_values.get(col, "Missing")
                df[col] = df[col].fillna(val)
                df[col] = df[col].astype(str)

            # Encode
            if self.encoder:
                encoded_arr = self.encoder.transform(df[present_cat])
                feat_names = self.encoder.get_feature_names_out(present_cat)
                encoded_df = pd.DataFrame(encoded_arr, columns=feat_names, index=df.index)
                
                df = pd.concat([df, encoded_df], axis=1)
                df.drop(columns=present_cat, inplace=True)

        # 4. SCALING
        feature_cols = [c for c in df.columns if c != self.target_column]
        self.final_feature_names = feature_cols
        
        if self.scaler:
            df[feature_cols] = self.scaler.transform(df[feature_cols])
        
        #STEP 5: MOVE TARGET TO THE END ---
        if self.target_column in df.columns:
            cols_ordered = [c for c in df.columns if c != self.target_column] + [self.target_column]
            df = df[cols_ordered]
            
        return df


    def fit_transform(self, df):
        self.fit(df)
        transformed_df = self.transform(df)

        feature_cols = [c for c in transformed_df.columns if c != self.target_column]
        
        self.scaler = StandardScaler()
        transformed_df[feature_cols] = self.scaler.fit_transform(transformed_df[feature_cols])
        
        return transformed_df
    

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
            "dropped_columns": self.dropped_cols,
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
