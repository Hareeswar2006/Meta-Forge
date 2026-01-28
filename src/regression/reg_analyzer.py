import os
import json
import pandas as pd
import csv
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def _safe_div(n, d):
    return float(n / d) if d not in (0, 0.0) else 0.0


def _clean_float(x):
    if pd.isna(x) or np.isinf(x):
        return 0.0
    return float(x)


def stats_analyzer(df, output_column):
    if df is None or df.shape[0] == 0:
        return [0, 0, 0, 0.0]

    total_values = df.size
    total_null_values = int(df.isna().sum().sum())
    pct_total = _safe_div(total_null_values, total_values) * 100

    if output_column not in df.columns:
        n_rows = df.shape[0]
        n_cols = df.shape[1]

        numerical_df = df.select_dtypes(include="number")
        categorical_df = df.select_dtypes(include=["object", "category"])

        numerical_null = {c: int(numerical_df[c].isna().sum()) for c in numerical_df.columns}
        categorical_null = {c: int(categorical_df[c].isna().sum()) for c in categorical_df.columns}

        return [
            n_rows,
            n_cols,
            numerical_df.shape[1],
            categorical_df.shape[1],
            numerical_df,
            numerical_null,
            categorical_df,
            categorical_null,
            total_values,
            total_null_values,
            pct_total,
        ]

    return [df.shape[0], total_values, total_null_values, pct_total]


def numerical_skew_kurtosis(df):
    stats = {}
    eps = 1e-6

    if df is None or df.shape[1] == 0:
        return stats

    for col in df.columns:
        series = df[col].dropna()
        n_nonnull = series.shape[0]

        if n_nonnull < 3:
            stats[col] = {
                "skewness_val": 0.0,
                "skewness_direction": "None",
                "kurtosis_val": 0.0,
                "kurtosis_direction": "None",
                "n_missing": int(df[col].isna().sum()),
                "n_unique": int(series.nunique()),
            }
            continue

        skew_val = _clean_float(series.skew())
        kur_val = _clean_float(series.kurtosis())

        skew_dir = "pos" if skew_val > eps else "neg" if skew_val < -eps else "neutral"
        kur_dir = "lepto" if kur_val > 0.5 else "platy" if kur_val < -0.5 else "meso"

        stats[col] = {
            "skewness_val": skew_val,
            "skewness_direction": skew_dir,
            "kurtosis_val": kur_val,
            "kurtosis_direction": kur_dir,
            "n_missing": int(df[col].isna().sum()),
            "n_unique": int(series.nunique()),
        }

    return stats



def pca_analysis(df):
    if df is None or df.shape[0] < 2 or df.shape[1] == 0:
        return 0.0, 0.0, 0.0

    X = df.dropna(axis=1, thresh=int(0.35 * len(df)))
    if X.shape[1] == 0:
        return 0.0, 0.0, 0.0

    X = X.fillna(X.median(numeric_only=True))

    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_components = min(3, X_scaled.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)

        explained = pca.explained_variance_ratio_.tolist()
        while len(explained) < 3:
            explained.append(0.0)

        return tuple(_clean_float(v) for v in explained[:3])

    except Exception:
        return 0.0, 0.0, 0.0


def distribution_stats(df, col):
    series = df[col].dropna()
    if series.empty:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    mean = _clean_float(series.mean())
    std = _clean_float(series.std())
    minv = _clean_float(series.min())
    maxv = _clean_float(series.max())
    n_unique = int(series.nunique())

    return (
        mean,
        std,
        minv,
        maxv,
        _clean_float(maxv - minv),
        float(n_unique),
        _safe_div(n_unique, df.shape[0]),
    )


def outliers(df, col):
    series = df[col].dropna()
    if series.shape[0] < 4:
        return 0, 0.0

    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    count = int(((series < lb) | (series > ub)).sum())
    return count, _safe_div(count, series.shape[0])


def pct_and_constant_detection(df, col):
    series = df[col].dropna()
    total = len(series)

    if total == 0:
        return 0.0, 0.0, 0.0, True, True

    zero = (series == 0).sum()
    pos = (series > 0).sum()
    neg = (series < 0).sum()

    top_freq = series.value_counts(normalize=True).iloc[0]

    return (
        _safe_div(zero, total) * 100,
        _safe_div(pos, total) * 100,
        _safe_div(neg, total) * 100,
        series.nunique() == 1,
        top_freq >= 0.95,
    )


def aggregate_numeric_stats(col_analysis, col_stats, df):
    if not col_stats:
        return {k: 0.0 for k in [
            "avg_skew","pct_skew_gt_1","avg_kur","pct_heavy_tailed","avg_std",
            "pct_low_variance","avg_outlier_pct","max_outlier_pct",
            "pct_columns_with_outliers","avg_pct_zero","avg_pct_pos",
            "avg_unique_ratio","pct_high_cardinality_cols"
        ]}

    skew_vals = [v["skewness_val"] for v in col_stats.values()]
    kur_vals = [v["kurtosis_val"] for v in col_stats.values()]

    std_vals = [v["std"] for v in col_analysis.values()]
    out_vals = [v["outlier_pct"] for v in col_analysis.values()]
    zero_vals = [v["pct_zero"] for v in col_analysis.values()]
    pos_vals = [v["pct_pos"] for v in col_analysis.values()]
    uniq_vals = [v["unique_ratio"] for v in col_analysis.values()]

    n = len(col_analysis)

    return {
        "avg_skew": _clean_float(np.mean(skew_vals)),
        "pct_skew_gt_1": _safe_div(sum(abs(v) > 1 for v in skew_vals), n),
        "avg_kur": _clean_float(np.mean(kur_vals)),
        "pct_heavy_tailed": _safe_div(sum(v > 3 for v in kur_vals), n),
        "avg_std": _clean_float(np.mean(std_vals)),
        "pct_low_variance": _safe_div(sum(v < 1e-3 for v in std_vals), n),
        "avg_outlier_pct": _clean_float(np.mean(out_vals)),
        "max_outlier_pct": _clean_float(np.max(out_vals)),
        "pct_columns_with_outliers": _safe_div(sum(v > 0.01 for v in out_vals), n),
        "avg_pct_zero": _clean_float(np.mean(zero_vals)),
        "avg_pct_pos": _clean_float(np.mean(pos_vals)),
        "avg_unique_ratio": _clean_float(np.mean(uniq_vals)),
        "pct_high_cardinality_cols": _safe_div(sum(v > 0.2 for v in uniq_vals), n),
    }



def get_correlations(df, output_column):
    if output_column not in df.columns:
        return 0.0, 0.0, 0.0, 0.0

    temp = df.copy()

    for col in temp.select_dtypes(include=["object", "category"]).columns:
        temp[col] = pd.factorize(temp[col])[0]

    if not pd.api.types.is_numeric_dtype(temp[output_column]):
        return 0.0, 0.0, 0.0, 0.0

    X = temp.drop(columns=[output_column])
    y = temp[output_column]

    X = X.loc[:, X.nunique() > 1]
    if X.shape[1] == 0:
        return 0.0, 0.0, 0.0, 0.0

    tc = X.corrwith(y).abs().fillna(0)
    mean_t, max_t = tc.mean(), tc.max()

    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
    vals = upper.stack()

    return (
        _clean_float(vals.mean()) if not vals.empty else 0.0,
        _clean_float(vals.max()) if not vals.empty else 0.0,
        _clean_float(mean_t),
        _clean_float(max_t),
    )



def X_analysis(X, output_column):
    nrows,ncols,nnumerical,ncategorical,numerical_df,numerical_null,categorical_df,categorical_null,total_values,total_null_values,pct_total = stats_analyzer(X, output_column)
    numerical_stats = numerical_skew_kurtosis(numerical_df)
    pca_com1, pca_com2, pca_com3 = pca_analysis(numerical_df)

    numerical_col_analysis={}
    for col in numerical_df.columns:
        col_mean, col_std, col_min, col_max, col_range, col_n_unique, col_unique_ratio = distribution_stats(numerical_df,col)
        outlier_count, outlier_pct = outliers(numerical_df,col)
        pct_zero, pct_pos, pct_neg, is_constant, is_quasi_constant = pct_and_constant_detection(numerical_df,col)

        numerical_col_analysis[col]={"mean":col_mean,"std":col_std,"cv":float(col_std/col_mean)if col_mean!=0 else 0.0,
                                     "min":col_min,"max":col_max,"range":col_range,"n_unique":col_n_unique,"unique_ratio":col_unique_ratio,
                                     "outlier_count":outlier_count,"outlier_pct":outlier_pct,"pct_zero":pct_zero,"pct_pos":pct_pos,"pct_neg":pct_neg,
                                     "is_constant":is_constant,"is_quasi_constant":is_quasi_constant
                                     }
        
    meta_structure = aggregate_numeric_stats(numerical_col_analysis, numerical_stats,numerical_df)
    meta_structure["pca_component_1"] = pca_com1
    meta_structure["pca_component_2"] = pca_com2
    meta_structure["pca_component_3"] = pca_com3
    meta_structure["n_numeric_cols"] = nnumerical
    meta_structure["n_categorical_cols"] = ncategorical
    meta_structure["numerical_ratio"] = float(nnumerical/(nnumerical+ncategorical))
    meta_structure["n_rows"] = nrows
    meta_structure["n_cols"] = ncols
    return meta_structure


def Y_analysis(Y,column_name):
    n_rows,n_total,n_null,pct_missing = stats_analyzer(Y, column_name)
    target_mean,target_std,target_min,target_max,target_range,target_n_unique,target_unique_ratio = distribution_stats(Y,column_name)
    output_skew_kur_stats = numerical_skew_kurtosis(Y)
    target_outlier_count, target_outlier_pct = outliers(Y,column_name)
    target_meta = {}
    target_meta["target_mean"] = target_mean
    target_meta['target_std']=target_std
    target_meta["target_min"] = target_min
    target_meta["target_max"] = target_max
    target_meta["target_range"] = target_range
    target_meta["target_skewness_val"] = output_skew_kur_stats[column_name]["skewness_val"]
    target_meta["target_skewness_direction"] = output_skew_kur_stats[column_name]["skewness_direction"]
    target_meta["target_kurtosis_val"] = output_skew_kur_stats[column_name]["kurtosis_val"]
    target_meta["target_kurtosis_direction"] = output_skew_kur_stats[column_name]["kurtosis_direction"]
    target_meta["target_outlier_count"] = target_outlier_count
    target_meta["target_outlier_pct"] = target_outlier_pct
    target_meta["target_missing_ratio"] = float(output_skew_kur_stats[column_name]["n_missing"]/Y.size)
    target_meta["target_unique_ratio"]=target_unique_ratio
    target_meta['target_n_unique']=target_n_unique
    return target_meta


def reg_meta(path, column_name):
    df = pd.read_csv(path)

    if column_name not in df.columns:
        raise ValueError("Target column not found")

    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise ValueError("Regression target must be numeric")

    avg_corr, max_corr, mean_corr_target, max_corr_target = get_correlations(df, column_name)

    X = df.drop(columns=[column_name])
    Y = df[[column_name]]

    meta = X_analysis(X, column_name)
    meta["missing_ratio"] = _safe_div(df.isna().sum().sum(), df.size)

    meta.update({
        "avg_corr_features": avg_corr,
        "max_corr_features": max_corr,
        "mean_corr_with_target": mean_corr_target,
        "max_corr_with_target": max_corr_target,
    })

    meta.update(Y_analysis(Y, column_name))
    return meta
