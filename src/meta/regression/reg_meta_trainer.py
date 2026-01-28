import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
META_TRAIN_CSV = os.path.join(BASE_DIR, "data", "meta", "meta_reg.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "models", "meta", "regression")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

OUTPUT_COLUMN = "best_model_label"
DROP_COLS = ["dataset_id", "source", "target_column", "timestamp"]


def load_training_data():
    if not os.path.exists(META_TRAIN_CSV):
        raise FileNotFoundError("meta_reg_train.csv not found")

    df = pd.read_csv(META_TRAIN_CSV)

    if df.empty:
        raise ValueError("meta_reg_train.csv is empty")

    if df[OUTPUT_COLUMN].isna().any():
        raise ValueError("Training CSV contains unlabeled rows")

    return df


def split_features_labels(df):
    X = df.drop(columns=DROP_COLS + [OUTPUT_COLUMN])
    y = df[OUTPUT_COLUMN]

    if X.shape[1] == 0:
        raise ValueError("No meta-features available")

    return X, y


def preprocess(X, y):
    X = X.fillna(0.0).astype(float)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, le


def train_meta_model(X, y_encoded):
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y_encoded, cv=cv, scoring="f1_macro")

    model.fit(X, y_encoded)

    return model, float(scores.mean())


def save_artifacts(model, label_encoder, feature_names, cv_score):
    with open(os.path.join(ARTIFACT_DIR, "meta_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(ARTIFACT_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

    with open(os.path.join(ARTIFACT_DIR, "features.json"), "w") as f:
        json.dump(feature_names, f, indent=4)

    with open(os.path.join(ARTIFACT_DIR, "meta_metrics.json"), "w") as f:
        json.dump({"cv_f1_macro": cv_score}, f, indent=4)


def main():
    print("[META] Training regression meta-model")

    df = load_training_data()
    X, y = split_features_labels(df)
    X, y_encoded, le = preprocess(X, y)

    model, cv_score = train_meta_model(X, y_encoded)

    save_artifacts(
        model=model,
        label_encoder=le,
        feature_names=list(X.columns),
        cv_score=cv_score
    )

    print(f"[DONE] Meta-model trained | CV F1-macro = {cv_score:.4f}")


if __name__ == "__main__":
    main()
