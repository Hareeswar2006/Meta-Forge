import os
import sys 
import json
import pickle
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREPROCESSOR_DIR = os.path.join(BASE_DIR, "models", "preprocessors")
PREPROCESSED_DIR = os.path.join(BASE_DIR, "data", "preprocessed")
MODEL_DIR = os.path.join(BASE_DIR, "models", "best_models")
BENCH_ARCHIVE_DIR = os.path.join(BASE_DIR, "models", "bench_archive")
META_REG_CSV = os.path.join(BASE_DIR, "data", "meta", "meta_reg.csv")
SIMPLE_MODELS = ["LinearRegression", "Ridge", "Lasso"]
R2_SIMPLICITY_THRESHOLD = 0.02
RMSE_GUARD_THRESHOLD = 0.05



def load_dataset_info(dataset_id):
    meta_path = f"{PREPROCESSOR_DIR}/{dataset_id}_preprocess_meta.json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}. Run preprocessing first.")
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
        target_col = meta["target"]

    data_path = f"{PREPROCESSED_DIR}/{dataset_id}_clean.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Clean data not found: {data_path}")
    
    df = pd.read_csv(data_path)

    return df, target_col


from sklearn.model_selection import cross_val_score, KFold

def train_evaluate_models(X, y):
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1, max_iter=15000),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print(f"\nBenchmarking {len(models)} models with 5-Fold CV...")
    
    for name, model in models.items():
        neg_mse_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-neg_mse_scores)
        
        r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
        
        avg_rmse = np.mean(rmse_scores)
        avg_r2 = np.mean(r2_scores)
        
        print(f"  -> {name}: CV-RMSE={avg_rmse:.4f}, CV-R2={avg_r2:.4f}")

        model.fit(X, y)
        
        results.append({
            "model_name": name,
            "model_obj": model, 
            "rmse": avg_rmse,   
            "r2": avg_r2
        })

    results_sorted = sorted(results, key=lambda x: x['rmse'])
    return results_sorted


def save_artifacts(dataset_id, best_result, all_results):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(BENCH_ARCHIVE_DIR, exist_ok=True)
    
    best_model_path = f"{MODEL_DIR}/{dataset_id}_best_model.pkl"
    with open(best_model_path, "wb") as f:
        pickle.dump(best_result['model_obj'], f)
    
    json_results = []
    for res in all_results:
        res_copy = res.copy()
        del res_copy['model_obj']
        json_results.append(res_copy)
        
    bench_path = f"{BENCH_ARCHIVE_DIR}/{dataset_id}_bench.json"
    with open(bench_path, "w") as f:
        json.dump(json_results, f, indent=4)
        
    print(f"\nBest Model saved to: {best_model_path}")
    print(f"Full Benchmark saved to: {bench_path}")


def update_meta_log(dataset_id, best_result):
    if not os.path.exists(META_REG_CSV):
        print("Warning: meta_reg.csv not found. Skipping metadata update.")
        return

    df_meta = pd.read_csv(META_REG_CSV)
    df_meta['best_model_label'] = df_meta['best_model_label'].astype(object)
    if 'best_model_label' not in df_meta.columns:
        df_meta['best_model_label'] = None
        df_meta['best_model_label'] = df_meta['best_model_label'].astype('object')

    if dataset_id in df_meta['dataset_id'].values:
        idx = df_meta[df_meta['dataset_id'] == dataset_id].index[0]
        
        df_meta.loc[idx, 'best_model_label'] = best_result['model_name']

        df_meta.to_csv(META_REG_CSV, index=False)
        print(f"meta_reg.csv updated for {dataset_id}")
    else:
        print(f"Dataset ID {dataset_id} not found in meta_reg.csv")


def select_smart_winner(results):
    # 1. Sort by R²
    results_sorted = sorted(results, key=lambda x: x['r2'], reverse=True)
    absolute_winner = results_sorted[0]

    # 2. Find best SIMPLE model by R²
    best_simple = None
    for res in results_sorted:
        if res['model_name'] in SIMPLE_MODELS:
            best_simple = res
            break

    # 3. If no simple model exists OR best model is already simple
    if best_simple is None or absolute_winner['model_name'] in SIMPLE_MODELS:
        return absolute_winner

    # 4. Compute relative R² improvement
    r2_gain = absolute_winner['r2'] - best_simple['r2']

    print("Decision Logic:")
    print(f"   Best Simple:  {best_simple['model_name']} (R²: {best_simple['r2']:.4f}, RMSE: {best_simple['rmse']:.4f})")
    print(f"   Best Complex: {absolute_winner['model_name']} (R²: {absolute_winner['r2']:.4f}, RMSE: {absolute_winner['rmse']:.4f})")
    print(f"   R² Gain:      {r2_gain:.4f} (Threshold: {R2_SIMPLICITY_THRESHOLD:.2f})")

    # 5. Simplicity rule based on R²
    if r2_gain < R2_SIMPLICITY_THRESHOLD:
        print("   R² gain is small. Choosing SIMPLE model.")
        return best_simple

    # 6. RMSE sanity check
    rmse_improvement = (best_simple['rmse'] - absolute_winner['rmse']) / best_simple['rmse']

    if rmse_improvement < RMSE_GUARD_THRESHOLD:
        print("   RMSE gain is also small. Choosing SIMPLE model.")
        return best_simple

    print("   R² gain is significant. Choosing COMPLEX model.")
    return absolute_winner
  


def run_bench(dataset_id):
    print(f"Starting Bench Training for: {dataset_id}")
    
    df, target_col = load_dataset_info(dataset_id)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    results = train_evaluate_models(X, y)

    winner = select_smart_winner(results)
    
    print(f"\n WINNER: {winner['model_name']} (RMSE: {winner['rmse']:.4f})")
    
    save_artifacts(dataset_id, winner, results)
    
    update_meta_log(dataset_id, winner)


if __name__ == "__main__":
    if not os.path.exists(PREPROCESSED_DIR):
        print(f"Directory not found: {PREPROCESSED_DIR}")
        exit()

    if os.path.exists(META_REG_CSV):
        df_meta = pd.read_csv(META_REG_CSV)
        df_meta['best_model_label'] = df_meta['best_model_label'].astype('object')
    else:
        df_meta = pd.DataFrame(columns=['dataset_id', 'best_model_label'])

    clean_files = [f for f in os.listdir(PREPROCESSED_DIR) if f.endswith("_clean.csv")]
    
    if not clean_files:
        print(f"No clean datasets found in {PREPROCESSED_DIR}. Please run preprocessing first.")
    else:
        print(f"Found {len(clean_files)} datasets. Checking status...")
        
        for file_name in clean_files:
            dataset_id = file_name.replace("_clean.csv", "")
            
            is_done = False
            if dataset_id in df_meta['dataset_id'].values:
                row = df_meta[df_meta['dataset_id'] == dataset_id].iloc[0]

                if pd.notna(row.get('best_model_label')) and row.get('best_model_label') != "":
                    is_done = True

            if is_done:
                print(f"Skipping {dataset_id}: Already benchmarked (Winner: {row['best_model_label']})")
                continue
            
            print(f"--------------------------------------------------")
            try:
                run_bench(dataset_id)
            except Exception as e:
                print(f"Failed to run bench for {dataset_id}")
                print(f"Error: {e}")
            print(f"--------------------------------------------------\n")