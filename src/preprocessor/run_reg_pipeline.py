import os
import pandas as pd
import warnings
from preprocessor.reg_preprocessor import RegressionPreprocessor


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# READ FROM META FILE
META_CSV = os.path.join(BASE_DIR, "data", "meta", "meta_reg.csv")

# RAW DATA
RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "regression")

# OUTPUT DIRS
PREP_DIR = os.path.join(BASE_DIR, "models", "preprocessors", "regression")
OUT_DIR = os.path.join(BASE_DIR, "data", "preprocessed", "regression")

os.makedirs(PREP_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

warnings.filterwarnings("ignore", category=RuntimeWarning)


def run_preprocessing():
    if not os.path.exists(META_CSV):
        print(f"[ERROR] Meta file not found at {META_CSV}")
        print("Please run reg_meta_writer.py first!")
        return

    print(f"[PREP] Reading registry: {META_CSV}")
    meta_df = pd.read_csv(META_CSV)

    if "target_column" not in meta_df.columns:
        print("[ERROR] 'target_column' missing from meta_reg.csv")
        return

    print(f"[PREP] Found {len(meta_df)} datasets in registry.")

    success_count = 0

    for _, row in meta_df.iterrows():
        dataset_id = str(row["dataset_id"])
        target_col = str(row["target_column"]).strip()

        raw_path = os.path.join(RAW_DIR, f"{dataset_id}.csv")
        clean_csv_path = os.path.join(OUT_DIR, f"{dataset_id}_clean.csv")

        # Skip if already processed
        if os.path.exists(clean_csv_path):
            print(f"[SKIP] {dataset_id} already preprocessed")
            success_count += 1
            continue

        # Check if raw file exists
        if not os.path.exists(raw_path):
            print(f"[WARN] Raw file missing: {raw_path} (Skipping)")
            continue

        print(f"\n[PREP] Processing {dataset_id} (Target: '{target_col}')")

        try:
            # Load Data
            df_raw = pd.read_csv(raw_path)

            # Initialize Preprocessor
            preprocessor = RegressionPreprocessor(target_column=target_col,dataset_id=dataset_id)

            # Fit & Transform
            df_clean = preprocessor.fit_transform(df_raw)

            # Save Artifacts
            preprocessor.save(PREP_DIR)
            df_clean.to_csv(clean_csv_path, index=False)

            print(f" -> Success! Saved to {clean_csv_path}")
            success_count += 1

        except Exception as e:
            print(f" -> [ERROR] Failed preprocessing {dataset_id}: {e}")
            continue

    print("--------------------------------------------------------------------------")
    print(f"[DONE] Successfully preprocessed {success_count}/{len(meta_df)} datasets.")


if __name__ == "__main__":
    run_preprocessing()
