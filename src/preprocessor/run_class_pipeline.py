import os
import pandas as pd
from preprocessor.class_preprocessor import ClassificationPreprocessor
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# READ FROM META FILE
META_CSV = os.path.join(BASE_DIR, "data", "meta", "meta_class.csv")

# RAW DATA 
RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "classification")

# OUTPUT DIRS
PREP_DIR = os.path.join(BASE_DIR, "models", "preprocessors", "classification")
OUT_DIR = os.path.join(BASE_DIR, "data", "preprocessed", "classification")

os.makedirs(PREP_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def run_preprocessing():
    if not os.path.exists(META_CSV):
        print(f"[ERROR] Meta file not found at {META_CSV}")
        print("Please run classification_meta_writer.py first!")
        return

    # Load the Meta-Data Registry
    print(f"[PREP] Reading registry: {META_CSV}")
    meta_df = pd.read_csv(META_CSV)
    
    # Check if 'target_column' exists
    if 'target_column' not in meta_df.columns:
        print("[ERROR] 'target_column' missing from meta_class.csv")
        return

    print(f"[PREP] Found {len(meta_df)} datasets in registry.")

    success_count = 0

    for index, row in meta_df.iterrows():
        dataset_id = row['dataset_id']
        target_col = row['target_column']
        
        # Construct path to the HASHED file
        raw_path = os.path.join(RAW_DIR, f"{dataset_id}.csv")
        clean_csv_path = os.path.join(OUT_DIR, f"{dataset_id}_clean.csv")

        # Skip if already done
        if os.path.exists(clean_csv_path):
            print(f"[SKIP] {dataset_id} already preprocessed")
            success_count += 1
            continue

        # Check if file exists
        if not os.path.exists(raw_path):
            print(f"[WARN] Raw file missing: {raw_path} (Skipping)")
            continue

        print(f"\n[PREP] Processing {dataset_id} (Target: '{target_col}')")

        try:
            # 1. Load Data
            df_raw = pd.read_csv(raw_path)

            # 2. Initialize Preprocessor
            preprocessor = ClassificationPreprocessor(target_column=target_col, dataset_id=dataset_id)

            # 3. Fit & Transform
            df_clean = preprocessor.fit_transform(df_raw)

            # 4. Save Artifacts
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