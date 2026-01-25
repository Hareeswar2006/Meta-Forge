import pandas as pd
import numpy as np
from scipy.stats import entropy

def dataset_structure_analyzer(df, output_column):
    temp_df = df.copy()
    if output_column in temp_df.columns:
        temp_df = temp_df.drop(columns=[output_column])

    n_rows = temp_df.shape[0]
    n_features = temp_df.shape[1]

    if n_features == 0:
        return {
            "num_rows": float(n_rows),
            "num_features": 0.0,
            "num_numeric_features": 0.0,
            "num_categorical_features": 0.0,
            "numeric_feature_ratio": 0.0,
            "categorical_feature_ratio": 0.0,
            "num_constant_features": 0.0,
            "num_near_constant_features": 0.0,
            "num_high_cardinality_categoricals": 0.0,
            "rows_to_features_ratio": 0.0
        }

    numeric_df = temp_df.select_dtypes(include=["number"])
    categorical_df = temp_df.select_dtypes(include=["object", "category", "bool"])

    n_numeric = numeric_df.shape[1]
    n_categorical = categorical_df.shape[1]

    numeric_ratio = n_numeric / n_features
    categorical_ratio = n_categorical / n_features

    num_constant_features = (temp_df.nunique(dropna=False) == 1).sum()

    num_near_constant_features = 0
    for col in temp_df.columns:
        value_counts = temp_df[col].value_counts(dropna=False)
        if len(value_counts) > 0:
            dominant_ratio = value_counts.iloc[0] / n_rows
            if dominant_ratio >= 0.95:
                num_near_constant_features += 1

    if n_categorical > 0:
        categorical_cardinalities = categorical_df.nunique(dropna=True)
        num_high_cardinality_categoricals = (
            categorical_cardinalities > 50
        ).sum()
    else:
        num_high_cardinality_categoricals = 0

    rows_to_features_ratio = n_rows / n_features

    return {
        "num_rows": float(n_rows),
        "num_features": float(n_features),
        "num_numeric_features": float(n_numeric),
        "num_categorical_features": float(n_categorical),
        "numeric_feature_ratio": float(numeric_ratio),
        "categorical_feature_ratio": float(categorical_ratio),
        "num_constant_features": float(num_constant_features),
        "num_near_constant_features": float(num_near_constant_features),
        "num_high_cardinality_categoricals": float(num_high_cardinality_categoricals),
        "rows_to_features_ratio": float(rows_to_features_ratio)
    }


def missing_analyzer(df, output_column):
    temp_df = df.copy()
    if output_column in temp_df.columns:
        temp_df = temp_df.drop(columns=[output_column])

    n_features = temp_df.shape[1]

    if n_features == 0:
        return {
            "total_missing_ratio": 0.0,
            "features_missing_ratio": 0.0,
            "max_feature_missing_ratio": 0.0,
            "num_features_missing_gt_10pct": 0.0,
            "num_features_missing_gt_30pct": 0.0,
            "num_features_missing_gt_50pct": 0.0,
            "mean_feature_missing_ratio": 0.0,
            "std_feature_missing_ratio": 0.0,
            "num_fully_observed_features": 0.0,
            "num_almost_empty_features": 0.0,
            "has_heavy_missingness_flag": False,
            "has_widespread_missingness_flag": False
        }

    total_values = temp_df.size
    total_null_values = temp_df.isna().sum().sum()
    total_missing_ratio = total_null_values / total_values

    missing_ratios = temp_df.isna().mean()

    num_features_with_missing = (missing_ratios > 0).sum()
    features_missing_ratio = num_features_with_missing / n_features

    max_feature_missing_ratio = missing_ratios.max()

    num_features_missing_gt_10pct = (missing_ratios > 0.10).sum()
    num_features_missing_gt_30pct = (missing_ratios > 0.30).sum()
    num_features_missing_gt_50pct = (missing_ratios > 0.50).sum()

    mean_feature_missing_ratio = missing_ratios.mean()
    std_feature_missing_ratio = missing_ratios.std()

    num_fully_observed_features = (missing_ratios == 0).sum()
    num_almost_empty_features = (missing_ratios > 0.90).sum()

    has_heavy_missingness_flag = bool(num_features_missing_gt_50pct > 0)

    has_widespread_missingness_flag = bool(features_missing_ratio >= 0.3)

    return {
        "total_missing_ratio": float(total_missing_ratio),
        "features_missing_ratio": float(features_missing_ratio),
        "max_feature_missing_ratio": float(max_feature_missing_ratio),
        "num_features_missing_gt_10pct": float(num_features_missing_gt_10pct),
        "num_features_missing_gt_30pct": float(num_features_missing_gt_30pct),
        "num_features_missing_gt_50pct": float(num_features_missing_gt_50pct),
        "mean_feature_missing_ratio": float(mean_feature_missing_ratio),
        "std_feature_missing_ratio": float(std_feature_missing_ratio),
        "num_fully_observed_features": float(num_fully_observed_features),
        "num_almost_empty_features": float(num_almost_empty_features),
        "has_heavy_missingness_flag": has_heavy_missingness_flag,
        "has_widespread_missingness_flag": has_widespread_missingness_flag
    }


def numeric_feature_distribution_analyzer(df, output_column):
    temp_df = df.copy()
    if output_column in temp_df.columns:
        temp_df = temp_df.drop(columns=[output_column])

    default_output = {
        "mean_numeric_variance": 0.0,
        "median_numeric_variance": 0.0,
        "pct_low_variance_features": 0.0,
        "mean_numeric_skewness": 0.0,
        "pct_highly_skewed_features": 0.0,
        "mean_numeric_kurtosis": 0.0,
        "pct_heavy_tailed_features": 0.0,
        "mean_outlier_ratio": 0.0,
        "max_outlier_ratio": 0.0,
        "mean_zero_ratio": 0.0,
        "pct_zero_inflated_features": 0.0
    }

    n_features = temp_df.shape[1]
    if n_features == 0:
        return default_output

    numeric_df = temp_df.select_dtypes(include=["number"])
    n_numeric = numeric_df.shape[1]
    if n_numeric == 0:
        return default_output

    non_constant_numeric_df = numeric_df.loc[:, numeric_df.var() > 0]
    k = non_constant_numeric_df.shape[1]
    if k == 0:
        return default_output

    per_feature_variance = non_constant_numeric_df.var()
    mean_numeric_variance = per_feature_variance.mean()
    median_numeric_variance = per_feature_variance.median()
    pct_low_variance_features = ((per_feature_variance < 0.1).sum() / n_numeric)

    per_feature_skewness = non_constant_numeric_df.skew()
    mean_numeric_skewness = per_feature_skewness.abs().mean()
    pct_highly_skewed_features = ((per_feature_skewness.abs() > 1).sum() / n_numeric)

    per_feature_kurtosis = non_constant_numeric_df.kurtosis()
    mean_numeric_kurtosis = per_feature_kurtosis.mean()
    pct_heavy_tailed_features = ((per_feature_kurtosis > 3).sum() / n_numeric)

    outlier_ratios = []

    for col in non_constant_numeric_df.columns:
        Q1 = non_constant_numeric_df[col].quantile(0.25)
        Q3 = non_constant_numeric_df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outlier_ratio = (((non_constant_numeric_df[col] < lower) |(non_constant_numeric_df[col] > upper)).mean())
        outlier_ratios.append(outlier_ratio)

    mean_outlier_ratio = float(np.mean(outlier_ratios))
    max_outlier_ratio = float(np.max(outlier_ratios))

    zero_ratios = (non_constant_numeric_df == 0).mean()

    mean_zero_ratio = zero_ratios.mean()
    pct_zero_inflated_features = ((zero_ratios > 0.5).sum() / n_numeric)

    return {
        "mean_numeric_variance": float(mean_numeric_variance),
        "median_numeric_variance": float(median_numeric_variance),
        "pct_low_variance_features": float(pct_low_variance_features),
        "mean_numeric_skewness": float(mean_numeric_skewness),
        "pct_highly_skewed_features": float(pct_highly_skewed_features),
        "mean_numeric_kurtosis": float(mean_numeric_kurtosis),
        "pct_heavy_tailed_features": float(pct_heavy_tailed_features),
        "mean_outlier_ratio": float(mean_outlier_ratio),
        "max_outlier_ratio": float(max_outlier_ratio),
        "mean_zero_ratio": float(mean_zero_ratio),
        "pct_zero_inflated_features": float(pct_zero_inflated_features)
    }


def categorical_feature_distribution_analyzer(df, output_column):
    temp_df = df.copy()
    if output_column in temp_df.columns:
        temp_df = temp_df.drop(columns=[output_column])

    default_output = {
        "mean_categorical_cardinality": 0.0,
        "max_categorical_cardinality": 0.0,
        "std_categorical_cardinality": 0.0,
        "pct_high_cardinality_categoricals": 0.0,
        "mean_categorical_entropy": 0.0,
        "pct_low_entropy_categoricals": 0.0,
        "mean_dominance_ratio": 0.0,
        "pct_highly_dominant_categoricals": 0.0,
        "mean_rare_category_ratio": 0.0,
        "pct_features_with_rare_categories": 0.0
    }

    n_features = temp_df.shape[1]
    if n_features == 0:
        return default_output

    categorical_df = temp_df.select_dtypes(include=["object", "category", "bool"])
    n_categorical = categorical_df.shape[1]
    if n_categorical == 0:
        return default_output

    per_feature_cardinality = categorical_df.nunique(dropna=True)
    mean_categorical_cardinality = per_feature_cardinality.mean()
    max_categorical_cardinality = per_feature_cardinality.max()
    std_categorical_cardinality = per_feature_cardinality.std()
    pct_high_cardinality_categoricals = ((per_feature_cardinality > 50).sum() / n_categorical)


    per_feature_entropy = categorical_df.apply(lambda col: entropy(col.value_counts(normalize=True, dropna=True),base=2))
    mean_categorical_entropy = per_feature_entropy.mean()
    pct_low_entropy_categoricals = ((per_feature_entropy < 0.2).sum() / n_categorical)

    per_feature_dominance = categorical_df.apply(lambda col: col.value_counts(normalize=True, dropna=True).max())
    mean_dominance_ratio = per_feature_dominance.mean()
    pct_highly_dominant_categoricals = ((per_feature_dominance > 0.9).sum() / n_categorical)

    rare_threshold = 0.01
    per_feature_rare_ratio = categorical_df.apply(
        lambda col: (
            (col.value_counts(normalize=True, dropna=True) < rare_threshold).sum()
            / col.nunique(dropna=True)
            if col.nunique(dropna=True) > 0 else 0
        )
    )
    mean_rare_category_ratio = per_feature_rare_ratio.mean()
    pct_features_with_rare_categories = ((per_feature_rare_ratio > 0).sum() / n_categorical)

    return {
        "mean_categorical_cardinality": float(mean_categorical_cardinality),
        "max_categorical_cardinality": float(max_categorical_cardinality),
        "std_categorical_cardinality": float(std_categorical_cardinality),
        "pct_high_cardinality_categoricals": float(pct_high_cardinality_categoricals),
        "mean_categorical_entropy": float(mean_categorical_entropy),
        "pct_low_entropy_categoricals": float(pct_low_entropy_categoricals),
        "mean_dominance_ratio": float(mean_dominance_ratio),
        "pct_highly_dominant_categoricals": float(pct_highly_dominant_categoricals),
        "mean_rare_category_ratio": float(mean_rare_category_ratio),
        "pct_features_with_rare_categories": float(pct_features_with_rare_categories)
    }


def target_analyzer(df, output_column):
    temp_df = df.copy()

    default_output = {
        "num_classes": 0.0,
        "is_binary": 0.0,
        "majority_class_ratio": 0.0,
        "minority_class_ratio": 0.0,
        "imbalance_ratio": 0.0,
        "target_entropy": 0.0,
        "normalized_target_entropy": 0.0,
        "num_rare_classes": 0.0,
        "rare_class_ratio": 0.0,
        "num_singleton_classes": 0.0,
        "target_missing_ratio": 0.0,
        "effective_sample_size": 0.0
    }

    if output_column not in temp_df.columns:
        return default_output
    
    if temp_df.shape[0] == 0:
        return default_output

    target_df = temp_df[output_column]

    if target_df.isna().sum() == target_df.shape[0]:
        default_output["target_missing_ratio"] = 1.0
        return default_output
    
    if target_df.dropna().nunique() == 1:
        default_output["num_classes"] = 1.0
        return default_output
    
    num_classes = target_df.dropna().nunique()
    is_binary = 1.0 if num_classes == 2 else 0

    obs_labels = target_df.notna().sum()
    majority_class_ratio = target_df.value_counts().max() / obs_labels
    minority_class_ratio = target_df.value_counts().min() / obs_labels
    imbalance_ratio = target_df.value_counts().max() / target_df.value_counts().min()

    target_entropy = entropy(target_df.value_counts(normalize=True, dropna=True),base=2)
    normalized_entropy = target_entropy / np.log2(num_classes)

    freqs = target_df.value_counts(normalize = True)
    num_rare_classes = (freqs < 0.01).sum()
    rare_class_ratio = num_rare_classes / num_classes

    classes_count = target_df.value_counts()
    num_singleton_classes = (classes_count == 1).sum()

    target_missing_ratio = target_df.isnull().sum() / target_df.shape[0]

    effective_sample_size = 1.0 / np.square(freqs).sum()

    return {
        "num_classes": float(num_classes),
        "is_binary": float(is_binary),
        "majority_class_ratio": float(majority_class_ratio),
        "minority_class_ratio": float(minority_class_ratio),
        "imbalance_ratio": float(imbalance_ratio),
        "target_entropy": float(target_entropy),
        "normalized_target_entropy": float(normalized_entropy),
        "num_rare_classes": float(num_rare_classes),
        "rare_class_ratio": float(rare_class_ratio),
        "num_singleton_classes": float(num_singleton_classes),
        "target_missing_ratio": float(target_missing_ratio),
        "effective_sample_size": float(effective_sample_size)
    }


def classification_meta(path, column_name):
    df = pd.read_csv(path)
    target_column = column_name
    meta_vector = {}

    structure_features = dataset_structure_analyzer(df, target_column)
    meta_vector.update(structure_features)

    missing_features = missing_analyzer(df, target_column)
    meta_vector.update(missing_features)

    numeric_feature_distribution_features = numeric_feature_distribution_analyzer(df, target_column)
    meta_vector.update(numeric_feature_distribution_features)

    categorical_feature_distribution_features = categorical_feature_distribution_analyzer(df, target_column)
    meta_vector.update(categorical_feature_distribution_features)

    target_features = target_analyzer(df, target_column)
    meta_vector.update(target_features)

    return meta_vector
