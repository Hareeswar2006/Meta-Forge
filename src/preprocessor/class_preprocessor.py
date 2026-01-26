import pandas as pd
import numpy as np
import os
import json
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ClassificationPreprocessor:

    def __init__(self, target_column, dataset_id="default"):
        self.target_column = target_column
        self.dataset_id = dataset_id

        self.numeric_cols = []
        self.cat_cols = []
        self.dropped_cols = set()
        self.zero_var_cols = set()

        self.numeric_medians = {}
        self.outlier_caps = {}

        self.cat_impute_values = {}
        self.encoder = None
        self.scaler = StandardScaler()

        self.final_feature_names = []

        self.n_rows_before = 0
        self.n_rows_after = 0
        

    def fit(self, df):
        df = df.copy()
        self.n_rows_before = len(df)

        # Drop rows with missing target
        df.dropna(subset=[self.target_column], inplace=True)
        self.n_rows_after = len(df)

        # Drop ID-like columns
        ID_KEYWORDS = ["id", "index", "row", "level", "unnamed"]

        for col in df.columns:
            col_lower = col.lower()

            if col_lower.startswith("unnamed") or col_lower in ["index", "level_0"]:
                self.dropped_cols.add(col)
                continue

            if any(k in col_lower for k in ID_KEYWORDS):
                if pd.api.types.is_numeric_dtype(df[col]):
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio > 0.9:
                        if df[col].is_monotonic_increasing or df[col].is_monotonic_decreasing:
                            self.dropped_cols.add(col)

        df.drop(columns=list(self.dropped_cols), inplace=True, errors="ignore")

        # Feature separation (AFTER drops)
        feature_df = df.drop(columns=[self.target_column], errors="ignore")

        self.numeric_cols = feature_df.select_dtypes(
            include=["number", "bool"]
        ).columns.tolist()

        self.cat_cols = feature_df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Numeric statistics
        for col in self.numeric_cols:
            median = df[col].median()
            self.numeric_medians[col] = median

            temp = df[col].fillna(median)
            lower = temp.quantile(0.01)
            upper = temp.quantile(0.99)
            self.outlier_caps[col] = (lower, upper)

        # Categorical pruning & imputation
        cols_to_remove = []

        for col in self.cat_cols:
            unique_ratio = df[col].nunique(dropna=True) / len(df)
            top_freq = df[col].value_counts(normalize=True).iloc[0] if not df[col].mode().empty else 0

            if unique_ratio > 0.2 or top_freq > 0.95:
                self.dropped_cols.add(col)
                cols_to_remove.append(col)
            else:
                self.cat_impute_values[col] = (
                    df[col].mode()[0] if not df[col].mode().empty else "Missing"
                )

        self.cat_cols = [c for c in self.cat_cols if c not in cols_to_remove]
        df.drop(columns=cols_to_remove, inplace=True, errors="ignore")

        # One-Hot Encoding
        if self.cat_cols:
            temp_cat = df[self.cat_cols].copy()
            for col in self.cat_cols:
                temp_cat[col] = temp_cat[col].fillna(
                    self.cat_impute_values[col]
                ).astype(str)

            self.encoder = OneHotEncoder(
                drop="first",
                sparse_output=False,
                handle_unknown="ignore",
                min_frequency=0.01,
                max_categories=20
            )
            self.encoder.fit(temp_cat)

        # Build final feature matrix
        temp_df = df.drop(columns=[self.target_column], errors="ignore")

        # Numeric handling
        for col in self.numeric_cols:
            if col in temp_df.columns:
                temp_df[col] = temp_df[col].fillna(self.numeric_medians[col])
                low, high = self.outlier_caps[col]
                temp_df[col] = np.clip(temp_df[col], low, high)

        # Categorical encoding
        if self.cat_cols and self.encoder:
            temp_cat = temp_df[self.cat_cols].copy()
            for col in self.cat_cols:
                temp_cat[col] = temp_cat[col].fillna(
                    self.cat_impute_values[col]
                ).astype(str)

            encoded = self.encoder.transform(temp_cat)
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.encoder.get_feature_names_out(self.cat_cols),
                index=temp_df.index
            )

            temp_df = pd.concat(
                [temp_df.drop(columns=self.cat_cols), encoded_df],
                axis=1
            )

        # Drop zero-variance columns
        zero_var_cols = temp_df.columns[temp_df.nunique() <= 1]
        if len(zero_var_cols) > 0:
            self.zero_var_cols = set(zero_var_cols)
            temp_df.drop(columns=zero_var_cols, inplace=True)

        temp_df = temp_df.replace([np.inf, -np.inf], np.nan)

        all_nan_cols = temp_df.columns[temp_df.isna().all()]
        if len(all_nan_cols) > 0:
            temp_df.drop(columns=all_nan_cols, inplace=True)

        if temp_df.isna().any().any():
            raise ValueError(
                f"[Preprocessor Error] NaNs remain after preprocessing.\n"
                f"Columns: {temp_df.columns[temp_df.isna().any()].tolist()}"
            )

        self.final_feature_names = temp_df.columns.tolist()
        self.scaler.fit(temp_df.astype(float))


    def transform(self, df):
        df = df.copy()

        if self.target_column in df.columns:
            df.dropna(subset=[self.target_column], inplace=True)

        df.drop(columns=list(self.dropped_cols), errors="ignore", inplace=True)
        df.drop(columns=list(self.zero_var_cols), errors="ignore", inplace=True)

        # Numeric
        for col in self.numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(self.numeric_medians[col])
                low, high = self.outlier_caps[col]
                df[col] = np.clip(df[col], low, high)

        # Categorical
        if self.cat_cols and self.encoder:
            temp_cat = df[self.cat_cols].copy()
            for col in self.cat_cols:
                temp_cat[col] = temp_cat[col].fillna(
                    self.cat_impute_values[col]
                ).astype(str)

            encoded = self.encoder.transform(temp_cat)
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.encoder.get_feature_names_out(self.cat_cols),
                index=df.index
            )

            df = pd.concat(
                [df.drop(columns=self.cat_cols), encoded_df],
                axis=1
            )

        df[self.final_feature_names] = self.scaler.transform(
            df[self.final_feature_names].astype(float)
        )

        if self.target_column in df.columns:
            df = df[self.final_feature_names + [self.target_column]]
        else:
            df = df[self.final_feature_names]

        return df


    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)


    def save(self, directory="models/preprocessors/classification"):
        os.makedirs(directory, exist_ok=True)
        base_path = os.path.join(directory, f"{self.dataset_id}")

        with open(f"{base_path}_preprocessor.pkl", "wb") as f:
            pickle.dump(self, f)

        meta = {
            "dataset_id": self.dataset_id,
            "target": self.target_column,
            "rows_raw": self.n_rows_before,
            "rows_clean": self.n_rows_after,
            "dropped_columns": sorted(list(self.dropped_cols)),
            "zero_variance_cols_dropped": sorted(list(self.zero_var_cols)),
            "numeric_cols_count": len(self.numeric_cols),
            "categorical_cols_count": len(self.cat_cols),
            "features_out": len(self.final_feature_names)
        }

        with open(f"{base_path}_preprocess_meta.json", "w") as f:
            json.dump(meta, f, indent=4)

        print(f"Classification preprocessing artifacts saved to {base_path}")
