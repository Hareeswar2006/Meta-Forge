import os
import json
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PREPROCESSOR_DIR = os.path.join(BASE_DIR, "models", "preprocessors", "classification")
PREPROCESSED_DIR = os.path.join(BASE_DIR, "data", "preprocessed","classification")
MODEL_DIR = os.path.join(BASE_DIR, "models", "best_models","classification")
BENCH_ARCHIVE_DIR = os.path.join(BASE_DIR, "models", "bench_archive", "classification")
META_CLASS_CSV = os.path.join(BASE_DIR, "data", "meta", "meta_class.csv")


SIMPLE_MODELS = ["LogisticRegression"]

F1_SIMPLICITY_THRESHOLD = 0.02
AUC_GUARD_THRESHOLD = 0.02

MAX_ROWS = 50000
CV_SPLITS = 5


def load_dataset_info(dataset_id):
    meta_path = f"{PREPROCESSOR_DIR}/{dataset_id}_preprocess_meta.json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing preprocess meta for {dataset_id}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    target_col = meta["target"]

    data_path = f"{PREPROCESSED_DIR}/{dataset_id}_clean.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing clean dataset for {dataset_id}")

    df = pd.read_csv(data_path)
    return df, target_col


def train_evaluate_models(X, y):
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            n_jobs=-1
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    results = []
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)

    is_binary = y.nunique() == 2

    print(f"\nBenchmarking {len(models)} models ({'Binary' if is_binary else 'Multiclass'})")

    for name, model in models.items():

        f1_scores = cross_val_score(
            model,
            X,
            y,
            cv=skf,
            scoring="f1_weighted"
        )
        avg_f1 = f1_scores.mean()

        avg_auc = None
        if is_binary:
            auc_scores = cross_val_score(
                model,
                X,
                y,
                cv=skf,
                scoring="roc_auc"
            )
            avg_auc = auc_scores.mean()

        model.fit(X, y)

        print(f"  -> {name}: F1={avg_f1:.4f}", end="")
        if avg_auc is not None:
            print(f", AUC={avg_auc:.4f}")
        else:
            print("")

        results.append({
            "model_name": name,
            "model_obj": model,
            "f1": avg_f1,
            "auc": avg_auc
        })

    return sorted(results, key=lambda x: x["f1"], reverse=True)


def select_smart_winner(results, is_binary):
    absolute_winner = results[0]

    best_simple = None
    for res in results:
        if res["model_name"] in SIMPLE_MODELS:
            best_simple = res
            break

    if best_simple is None or absolute_winner["model_name"] in SIMPLE_MODELS:
        return absolute_winner

    f1_gain = absolute_winner["f1"] - best_simple["f1"]

    print("\nDecision Logic:")
    print(f"  Best Simple:  {best_simple['model_name']} (F1={best_simple['f1']:.4f})")
    print(f"  Best Complex: {absolute_winner['model_name']} (F1={absolute_winner['f1']:.4f})")
    print(f"  F1 Gain: {f1_gain:.4f} (Threshold={F1_SIMPLICITY_THRESHOLD})")

    if f1_gain < F1_SIMPLICITY_THRESHOLD:
        print("  → Downgrading to SIMPLE model (F1 gain too small)")
        return best_simple

    if is_binary:
        auc_gain = absolute_winner["auc"] - best_simple["auc"]
        print(f"  AUC Gain: {auc_gain:.4f} (Threshold={AUC_GUARD_THRESHOLD})")

        if auc_gain < AUC_GUARD_THRESHOLD:
            print("  → Downgrading to SIMPLE model (AUC gain too small)")
            return best_simple

    print("  → Choosing COMPLEX model")
    return absolute_winner


def save_artifacts(dataset_id, winner, all_results):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(BENCH_ARCHIVE_DIR, exist_ok=True)

    with open(f"{MODEL_DIR}/{dataset_id}_best_model.pkl", "wb") as f:
        pickle.dump(winner["model_obj"], f)

    json_results = []
    for r in all_results:
        r_copy = r.copy()
        r_copy.pop("model_obj")
        json_results.append(r_copy)

    with open(f"{BENCH_ARCHIVE_DIR}/{dataset_id}_bench.json", "w") as f:
        json.dump(json_results, f, indent=4)

    print(f"\nSaved best model → {winner['model_name']}")


def update_meta_log(dataset_id, winner):
    if not os.path.exists(META_CLASS_CSV):
        print("meta_class.csv not found — skipping update")
        return

    df_meta = pd.read_csv(META_CLASS_CSV)
    df_meta["best_model_label"] = df_meta["best_model_label"].astype(object)

    if dataset_id in df_meta["dataset_id"].values:
        idx = df_meta[df_meta["dataset_id"] == dataset_id].index[0]
        df_meta.loc[idx, "best_model_label"] = winner["model_name"]
        df_meta.to_csv(META_CLASS_CSV, index=False)
        print("meta_class.csv updated")


def run_bench(dataset_id):
    print(f"\nStarting Classification Bench: {dataset_id}")

    df, target_col = load_dataset_info(dataset_id)

    if len(df) > MAX_ROWS:
        print(f"Downsampling from {len(df)} → {MAX_ROWS}")
        df = df.sample(MAX_ROWS, random_state=42)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    results = train_evaluate_models(X, y)
    winner = select_smart_winner(results, is_binary=(y.nunique() == 2))

    save_artifacts(dataset_id, winner, results)
    update_meta_log(dataset_id, winner)


if __name__ == "__main__":

    if not os.path.exists(PREPROCESSED_DIR):
        print("Preprocessed directory not found.")
        exit()

    if os.path.exists(META_CLASS_CSV):
        df_meta = pd.read_csv(META_CLASS_CSV)
        df_meta["best_model_label"] = df_meta["best_model_label"].astype(object)
    else:
        df_meta = pd.DataFrame(columns=["dataset_id", "best_model_label"])

    clean_files = [
        f for f in os.listdir(PREPROCESSED_DIR)
        if f.endswith("_clean.csv")
    ]

    if not clean_files:
        print("No clean datasets found.")
        exit()

    print(f"Found {len(clean_files)} datasets. Checking status...")

    for file_name in clean_files:
        dataset_id = file_name.replace("_clean.csv", "")

        is_done = False
        if dataset_id in df_meta["dataset_id"].values:
            row = df_meta[df_meta["dataset_id"] == dataset_id].iloc[0]
            if pd.notna(row.get("best_model_label")) and row.get("best_model_label") != "":
                is_done = True

        if is_done:
            print(f"Skipping {dataset_id}: Already benchmarked "
                  f"(Winner: {row['best_model_label']})")
            continue

        print("-" * 60)
        try:
            run_bench(dataset_id)
        except Exception as e:
            print(f"Failed for {dataset_id}")
            print(f"Error: {e}")
        print("-" * 60)