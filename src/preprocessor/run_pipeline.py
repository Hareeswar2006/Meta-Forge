import pandas as pd
import pickle
import os
from preprocessor.reg_preprocessor import RegressionPreprocessor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "db9f62a1c5d2e84b.csv")
TARGET = "price"
DATASET_ID = "db9f62a1c5d2e84b"
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "preprocessors")
DATA_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "preprocessed")
    
# 1. LOAD DATA
print("Loading Data...")
df_raw = pd.read_csv(DATA_PATH)

# 2. RUN PIPELINE
print("Running Preprocessing Pipeline...")
preprocessor = RegressionPreprocessor(target_column=TARGET, dataset_id=DATASET_ID)
df_clean = preprocessor.fit_transform(df_raw)

# 3. SAVE ARTIFACTS & DATA
print("Saving Artifacts...")
preprocessor.save(OUTPUT_DIR)

os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
clean_csv_path = f"{DATA_OUTPUT_DIR}/{DATASET_ID}_clean.csv"
df_clean.to_csv(clean_csv_path, index=False)
print(f"Cleaned Data saved to {clean_csv_path}")