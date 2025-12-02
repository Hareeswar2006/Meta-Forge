import os
import json
import pandas as pd
import csv
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def stats_analyzer(file_path):
    df=pd.read_csv(file_path)
    total_values=df.size
    total_null_values=df.isna().sum().sum()
    pct_total = (total_null_values/total_values)*100

    n_rows=df.shape[0]
    n_cols=df.shape[1]
    numerical_df = df.select_dtypes(include='number')
    categorical_df = df.select_dtypes(include=['object','category'])
    n_numerical_cols= numerical_df.shape[1]
    n_categorical_cols =categorical_df.shape[1]

    numerical_null={}
    categorical_null={}
    for column in numerical_df.columns:
        numerical_null[column] = int(numerical_df[column].isna().sum())
    for column in categorical_df.columns:
        categorical_null[column] = int(categorical_df[column].isna().sum())

    return [
            n_rows, n_cols, n_numerical_cols, n_categorical_cols, numerical_df, numerical_null,
            categorical_df, categorical_null, total_values, total_null_values, pct_total
            ]


def numerical_skew_kurtosis(df):
    numerical_stats={}
    eps = 1e-6
    for col in df.columns:
        n_nonnull = df[col].dropna().shape[0]
        skew_val=df[col].dropna().skew()
        kur_val=df[col].dropna().kurtosis()

        if n_nonnull>3:
            if skew_val>eps:
                skew_dir="pos"
            elif skew_val<-eps:
                skew_dir="neg"
            else:
                skew_dir="neutral"

            if kur_val<-0.5:
                kur_dir="platy"
            elif kur_val> 0.5:
                kur_dir="lepto"
            else:
                kur_dir="meso"
        else:
            skew_val=0
            skew_dir="None"
            kur_val=0
            kur_dir="None"

        numerical_stats[col]= {"skewness_val":float(skew_val),"skewness_direction":skew_dir,"kurtosis_val":float(kur_val),"kurtosis_direction":kur_dir,"n_missing": int(df[col].isna().sum()),"n_unique": int(df[col].nunique(dropna=True))}
    return numerical_stats


def pca_analysis(df):
    X = df.copy()
    threshold = 0.65
    X = X.dropna(axis=1, thresh=int((1 - threshold) * len(df)))
    if X.shape[1] == 0 or X.shape[0] < 2:
        return {"pca_component_1":0.0,"pca_component_2":0.0,"pca_component_3":0.0}
    
    for col in X.columns:
        X[col].fillna(float(X[col].median()), inplace = True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(3, X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    explained = pca.explained_variance_ratio_.tolist()

    while len(explained) < 3:
        explained.append(0.0)

    pca_meta = {
        "pca_component_1" : explained[0],
        "pca_component_2" : explained[1],
        "pca_component_3" : explained[2]
    }

    return pca_meta




if __name__ == "__main__":
    file_path="../data/Travel.csv"
    nrows,ncols,nnumerical,ncategorical,numerical_df,numerical_null,categorical_df,categorical_null,total_values,total_null_values,pct_total = stats_analyzer(file_path)
    numerical_stats = numerical_skew_kurtosis(numerical_df)
    pca_meta = pca_analysis(numerical_df)
    print("No of rows: ",nrows)
    print("No of cols: ",ncols)

    print("-----------------------------------------")

    print("No of numerial rows:",nnumerical)
    print("No of categorical rows:",ncategorical)

    print("-----------------------------------------")

    print("Numerical DF:\n",numerical_df)
    print("Numerical null columns: ", numerical_null)

    print("-----------------------------------------")

    print("Categorical DF:\n",categorical_df)
    print("Categorical null columns: ", categorical_null)

    print("-----------------------------------------")

    print("Total Values :",total_values)
    print("Total Null values :",total_null_values)
    print(" Total percentage null values :",pct_total)

    print("-----------------------------------------")

    print("\nNumerical columns statistics (Skewness and Kurtosis):\n\n\n",numerical_stats)

    print("-----------------------------------------")

    print("\n PCA Analysis:\n",pca_meta)
