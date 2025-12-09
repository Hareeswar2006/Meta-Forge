import os
import hashlib
import pandas as pd
import numpy as np
from regression.reg_analyzer import reg_meta
from utils.id_creation import save_uploaded_dataset
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
META_CSV = os.path.join(BASE_DIR, "data", "meta", "meta_reg.csv")

uploaded_path = os.path.join(BASE_DIR, "Travel.csv")
DATA_PATH, id = save_uploaded_dataset(uploaded_path)

column_name = "MonthlyIncome"
meta = reg_meta(DATA_PATH, column_name)



HEADER_ORDER = [
  "dataset_id","source","target_column","timestamp",
  "n_rows","n_cols","n_numeric_cols","n_categorical_cols","numerical_ratio","missing_ratio",
  "pca_component_1","pca_component_2","pca_component_3",
  "avg_skew","pct_skew_gt_1","avg_kur","pct_heavy_tailed",
  "avg_std","pct_low_variance","avg_outlier_pct","pct_columns_with_outliers",
  "avg_pct_zero","avg_pct_pos","avg_unique_ratio","pct_high_cardinality_cols",
  "target_mean","target_std","target_skewness_val","target_kurtosis_val",
  "target_outlier_pct","target_missing_ratio","target_unique_ratio","target_n_unique",
  "best_model_label"
]

def validate_meta_flat(meta):
    if not isinstance(meta,dict):
        raise ValueError("meta data must be a dict")
    bad =[k for k,v in meta.items() if isinstance(v,(dict,list,tuple,set))]
    if bad:
        raise ValueError(f"Nested values found in the meta data")
    return True


def build_row_dict(meta, id, source, target_column):
    row={}

    row["dataset_id"] = id
    row["source"] = source
    row["target_column"] = target_column
    row["timestamp"] = datetime.now().isoformat()

    for key in HEADER_ORDER:
        if key in ("dataset_id", "source", "target_column", "timestamp"):
            continue
        if key in meta:
            row[key] = meta[key]
        else:
            row[key] = 0.0 if ("pct" in key or "avg" in key or "std" in key or "mean" in key or "ratio" in key or "n_" in key or "count" in key) else ""

    row.setdefault("best_model_label", "")

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
    
    return {k: coerce_value(v) for k,v in row.items()}


def create_meta_file(path, header, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([row], columns=header)
    df.to_csv(path, index=False)
    print(f"Created meta file and wrote dataset_id={row['dataset_id']}")


def append_if_new(path, header, row):
    # read existing
    df = pd.read_csv(path)
    # ensure header matches (simple check)
    existing_cols = list(df.columns)
    if existing_cols != header:
        raise RuntimeError(f"Header mismatch!\nExisting: {existing_cols}\nExpected: {header}")

    # check duplicate dataset_id
    if str(row['dataset_id']) in df['dataset_id'].astype(str).values:
        print(f"Dataset_id {row['dataset_id']} already present — skipping append.")
        return False

    # append and save (atomic-ish: write to temp then replace)
    tmp = path + ".tmp"
    df2 = pd.concat([df, pd.DataFrame([row], columns=header)], ignore_index=True)
    df2.to_csv(tmp, index=False)
    os.replace(tmp, path)
    print(f"Appended dataset_id={row['dataset_id']} to {path}")
    return True


def verify_last_row(path):
    df = pd.read_csv(path)
    last = df.tail(1).to_dict(orient='records')[0]
    print("Header:", list(df.columns))
    print("\n\nLast row preview:", last)


if __name__ == "__main__":
    if (validate_meta_flat(meta)):
        row = build_row_dict(meta, id, DATA_PATH, column_name)
        coerce_row(row)
        if not os.path.exists(META_CSV):
            create_meta_file(META_CSV, HEADER_ORDER, row)
            verify_last_row(META_CSV)
        else:
            appended = append_if_new(META_CSV, HEADER_ORDER, row)
            if appended:
                verify_last_row(META_CSV)
    else:
        print("[ERROR] Meta Dictionary is not flattened......")