import os
import json
import hashlib
import pandas as pd
import numpy as np
from classification.classification_analyzer import classification_meta
from utils.id_creation import save_uploaded_dataset
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
META_CSV = os.path.join(BASE_DIR, "data", "meta", "meta_class.csv")
TARGETS_JSON = os.path.join(BASE_DIR, "datasets", "targets.json")

HEADER_ORDER = [
    "dataset_id","source","target_column","timestamp",

    #  Dataset Structure 
    "num_rows","num_features","num_numeric_features","num_categorical_features",
    "numeric_feature_ratio","categorical_feature_ratio",
    "num_constant_features","num_near_constant_features",
    "num_high_cardinality_categoricals",
    "rows_to_features_ratio",

    # Missingness 
    "total_missing_ratio","features_missing_ratio","max_feature_missing_ratio",
    "num_features_missing_gt_10pct","num_features_missing_gt_30pct",
    "num_features_missing_gt_50pct","mean_feature_missing_ratio",
    "std_feature_missing_ratio","num_fully_observed_features",
    "num_almost_empty_features","has_heavy_missingness_flag",
    "has_widespread_missingness_flag",

    # Numeric Feature Distribution 
    "mean_numeric_variance","median_numeric_variance","pct_low_variance_features",
    "mean_numeric_skewness","pct_highly_skewed_features",
    "mean_numeric_kurtosis","pct_heavy_tailed_features",
    "mean_outlier_ratio","max_outlier_ratio",
    "mean_zero_ratio","pct_zero_inflated_features",

    # Categorical Feature Distribution 
    "mean_categorical_cardinality","max_categorical_cardinality",
    "pct_high_cardinality_categoricals","mean_categorical_entropy",
    "pct_low_entropy_categoricals","mean_dominance_ratio",
    "pct_highly_dominant_categoricals","mean_rare_category_ratio",
    "pct_features_with_rare_categories","std_categorical_cardinality",

    # Target Analysis
    "num_classes","is_binary","majority_class_ratio","minority_class_ratio",
    "imbalance_ratio","target_entropy","normalized_target_entropy",
    "num_rare_classes","rare_class_ratio","num_singleton_classes",
    "target_missing_ratio","effective_sample_size",

    # Target / Correlation Analysis
    "mean_abs_corr_target",
    "max_corr_target",
    "mean_inter_feature_corr",

    # Meta Label
    "best_model_label"
]

def validate_meta_flat(meta):
    if not isinstance(meta, dict):
        raise ValueError("Meta output must be a dictionary")

    bad = [k for k, v in meta.items() if isinstance(v, (dict, list, tuple, set))]
    if bad:
        raise ValueError(f"Nested values found in meta data: {bad}")

    return True


def build_row_dict(meta, dataset_id, source, target_column):
    row = {}

    row["dataset_id"] = dataset_id
    row["source"] = source
    row["target_column"] = target_column
    row["timestamp"] = datetime.now().isoformat()

    for key in HEADER_ORDER:
        if key in ("dataset_id", "source", "target_column", "timestamp"):
            continue
        if key in meta:
            row[key] = meta[key]
        else:
            if any(x in key for x in ["pct", "ratio", "mean", "std", "num", "count"]):
                row[key] = 0.0
            else:
                row[key] = ""

    row.setdefault("best_model_label", "Nil")
    return row


def coerce_row(row):

    def coerce_value(v):
        if pd.isna(v):
            return ""
        if isinstance(v, (np.floating, np.float64, np.float32)):
            return float(v)
        if isinstance(v, (np.integer, np.int64, np.int32)):
            return int(v)
        if isinstance(v, (np.bool_, bool)):
            return bool(v)
        return str(v) if isinstance(v, (pd.Timestamp,)) else v

    return {k: coerce_value(v) for k, v in row.items()}


def create_meta_file(path, header, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([row], columns=header)
    df.to_csv(path, index=False)
    print(f"Created classification meta file with dataset_id={row['dataset_id']}")


def append_if_new(path, header, row):
    # read existing
    df = pd.read_csv(path)

    if list(df.columns) != header:
        raise RuntimeError(f"Header mismatch!\nExisting: {list(df.columns)}\nExpected: {header}")

    # check duplicate dataset_id
    if str(row["dataset_id"]) in df["dataset_id"].astype(str).values:
        print(f"Dataset_id {row['dataset_id']} already exists — skipping.")
        return False

    # append and save
    tmp = path + ".tmp"
    df2 = pd.concat([df, pd.DataFrame([row], columns=header)], ignore_index=True)
    df2.to_csv(tmp, index=False)
    os.replace(tmp, path)

    print(f"Appended dataset_id={row['dataset_id']} to classification meta file")
    return True


def verify_last_row(path):
    df = pd.read_csv(path)
    last = df.tail(1).to_dict(orient="records")[0]
    print("Header:", list(df.columns))
    print("\nLast row preview:\n", last)


def run_classification_meta_writer():
    with open(TARGETS_JSON, "r") as f:
        targets = json.load(f)["classification"]

    for filename, target_col in targets.items():
        print(f"\n[META] Processing {filename}")

        uploaded_path = os.path.join(BASE_DIR, "datasets", "classification", filename)

        if not os.path.exists(uploaded_path):
            print(f"[SKIP] File not found: {filename}")
            continue

        # Create raw dataset + dataset_id
        DATA_PATH, dataset_id = save_uploaded_dataset(
            uploaded_path, "classification"
        )

        # Compute meta-features
        meta = classification_meta(DATA_PATH, target_col)

        if not validate_meta_flat(meta):
            print(f"[ERROR] Invalid meta for {filename}")
            continue

        row = build_row_dict(meta, dataset_id, DATA_PATH, target_col)
        row = coerce_row(row)

        if not os.path.exists(META_CSV):
            create_meta_file(META_CSV, HEADER_ORDER, row)
        else:
            append_if_new(META_CSV, HEADER_ORDER, row)


if __name__ == "__main__":
    run_classification_meta_writer()