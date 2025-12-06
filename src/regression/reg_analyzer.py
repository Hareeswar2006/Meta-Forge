import os
import json
import pandas as pd
import csv
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def stats_analyzer(df):
    total_values=df.size
    total_null_values=df.isna().sum().sum()
    pct_total = (total_null_values/total_values)*100

    if df.shape[1]>1:
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
    
    else:
        n_rows=df.shape[0]

    return [n_rows, total_values, total_null_values, pct_total]


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
    
    X = X.fillna(X.median(numeric_only=True))

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


def distribution_stats(df,column_name):
    target_mean = df[column_name].mean()
    target_std = df[column_name].std(axis=0)
    target_min = df[column_name].min()
    target_max = df[column_name].max()
    target_range = target_max - target_min
    target_n_unique = df[column_name].nunique()
    target_unique_ratio = target_n_unique / df.shape[0]

    return float(target_mean),float(target_std),float(target_min),float(target_max),float(target_range),float(target_n_unique),float(target_unique_ratio)


def outliers(df,column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    target_outlier_count = int(((df[column_name].notnull()) & ((df[column_name] < lower_bound) | (df[column_name] > upper_bound))).sum())
    target_outlier_pct = float(target_outlier_count / df[column_name].notnull().sum())

    return target_outlier_count, target_outlier_pct


def pct_and_constant_detection(numerical_df,col):
    pct_total = len(numerical_df[col])
    zero_count = (numerical_df[col] == 0).sum()
    pos_count=(numerical_df[col]>0).sum()
    neg_count=(numerical_df[col]<0).sum()
    
    pct_zero=float(zero_count/pct_total)*100
    pct_pos = float(pos_count/pct_total)*100
    pct_neg=float(neg_count/pct_total)*100

    is_constant = numerical_df[col].nunique(dropna=False)==1
    top_freq_ratio=numerical_df[col].value_counts(normalize=True,dropna=False).iloc[0]
    is_quasi_constant=bool(top_freq_ratio>=0.95)
    return pct_zero, pct_pos, pct_neg, is_constant, is_quasi_constant


def input_display(nnumerical,numerical_df,numerical_null,total_values,total_null_values,pct_total,numerical_stats,numerical_col_analysis,meta_structure):
    print("-----------------------------------------")

    print("No of numerial rows:",nnumerical)

    print("-----------------------------------------")

    print("Numerical DF:\n",numerical_df)
    print("Numerical null columns: ", numerical_null)

    print("-----------------------------------------")

    print("Total Values :",total_values)
    print("Total Null values :",total_null_values)
    print(" Total percentage null values :",pct_total)

    print("-----------------------------------------")

    print("\nNumerical columns statistics (Skewness and Kurtosis):\n\n\n",numerical_stats)

    print("-----------------------------------------")
    print("\n Numerical column wise analysis: \n",numerical_col_analysis)

    print("-----------------------------------------\n")
    print("-----Meta Structure----- \n",meta_structure)


def output_display(n_rows,n_total,n_null,pct_missing,target_mean,target_std,target_min,target_max,target_range,target_n_unique,target_unique_ratio,output_skew_kur_stats,target_outlier_count,target_outlier_pct):
    print("\n-------------------------------\n")
    print("No of rows: ",n_rows)
    print("Total Values : ",n_total)
    print("Total Null values : ",n_null)
    print("Total no of non null values: ",n_total-n_null)
    print(" Total percentage null values : ",pct_missing)

    print("\n-------------------------------\n")

    print("Target column mean: ",target_mean)
    print("Target column std: ", target_std)
    print("Target column maximum: ", target_max)
    print("Target column minimum: ", target_min)
    print("Target column range: ", target_range)

    print("\n-------------------------------\n")

    print("No of unique in target column: ", target_n_unique)
    print("Unique values percentage in target column: ", target_unique_ratio)

    print("\n-------------------------------\n")

    print("Target column skewness and kurtosis: ", output_skew_kur_stats)
    ishighly_skewed = 'Yes' if output_skew_kur_stats[output_column]['skewness_val']> 1 else 'No'
    print("\nHighy Skewed?? : ",ishighly_skewed)
    
    print("\n-------------------------------\n")

    print("Target column outlier count: ", target_outlier_count)
    print("Target column outlier percentage: ", target_outlier_pct)


def aggregate_numeric_stats(col_analysis, col_stats, df):
    skew_list,kur_list,std_list,outlier_list,pct_zero_list,pct_pos_list,unique_list=[],[],[],[],[],[],[]
    pos_count,neg_count,heavy_count,outlier_count,std_count,card_count=0,0,0,0,0,0

    for col in df.columns:
        for key in col_stats[col].keys():
            if key == "skewness_val":
                skew_list.append(col_stats[col][key])
                if col_stats[col][key]>1:
                    pos_count+=1
                elif col_stats[col][key]<-1:
                    neg_count+=1
            elif key == "kurtosis_val":
                kur_list.append(col_stats[col][key])
                if col_stats[col][key] > 3:
                    heavy_count+=1

        for key in col_analysis[col].keys():
            if key == "std":
                if col_analysis[col][key]<0.001:
                    std_count+=1
                std_list.append(col_analysis[col][key])
            elif key == "outlier_pct":
                if col_analysis[col][key]>0.01:
                    outlier_count+=1
                outlier_list.append(col_analysis[col][key])
            elif key == "pct_zero":
                pct_zero_list.append(col_analysis[col][key])
            elif key == "pct_pos":
                pct_pos_list.append(col_analysis[col][key])
            elif key == "unique_ratio":
                if col_analysis[col][key]>0.2:
                    card_count+=1
                unique_list.append(col_analysis[col][key])

    avg_skew, std_skew,pct_skew_gt_1,pct_skew_lt_1 = np.average(skew_list), np.std(skew_list), float(pos_count/len(col_stats.keys())), float(neg_count/len(col_stats.keys()))
    avg_kur, std_kur,pct_heavy_tailed = np.average(kur_list), np.std(kur_list), float(heavy_count/len(col_stats.keys()))
    avg_std, std_std, pct_low_variance = np.average(std_list), np.std(std_list), float(std_count/len(col_analysis.keys()))
    avg_outlier_pct, max_outlier_pct, pct_columns_with_outliers = np.average(outlier_list), np.max(outlier_list), float(outlier_count/len(col_analysis.keys()))
    avg_pct_zero, avg_pct_pos = np.average(pct_zero_list), np.average(pct_pos_list)
    avg_unq_ratio, pct_high_card_cols = np.average(unique_list), float(card_count/len(col_analysis.keys()))

    meta_structure = {
        "avg_skew": float(avg_skew),
        "std_skew": float(std_skew),
        "pct_skew_gt_1": float(pct_skew_gt_1),
        "pct_skew_lt_-1": float(pct_skew_lt_1),
        "avg_kur": float(avg_kur),
        "std_kur": float(std_kur),
        "pct_heavy_tailed": float(pct_heavy_tailed),
        "avg_std": float(avg_std),
        "std_of_std": float(std_std),
        "pct_low_variance":float(pct_low_variance),
        "avg_outlier_pct":float(avg_outlier_pct),
        "max_outlier_pct":float(max_outlier_pct),
        "pct_columns_with_outliers":float(pct_columns_with_outliers),
        "avg_pct_zero":float(avg_pct_zero),
        "avg_pct_pos":float(avg_pct_pos),
        "avg_unique_ratio": float(avg_unq_ratio),
        "pct_high_cardinality_cols":float(pct_high_card_cols)
    }

    return meta_structure


def X_analysis(X):
    #---------------------------------
    nrows,ncols,nnumerical,ncategorical,numerical_df,numerical_null,categorical_df,categorical_null,total_values,total_null_values,pct_total = stats_analyzer(X)
    numerical_stats = numerical_skew_kurtosis(numerical_df)
    # pca_meta = pca_analysis(numerical_df)

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
    #----------------------------------
    meta_structure = aggregate_numeric_stats(numerical_col_analysis, numerical_stats,numerical_df)
    meta_structure["n_numeric_rows"] = nnumerical
    meta_structure["n_categorical_rows"] = ncategorical
    meta_structure["numerical_ratio"] = float(nnumerical/(nnumerical+ncategorical))
    #----------------------------------
    input_display(nnumerical,numerical_df,numerical_null,total_values,total_null_values,pct_total,numerical_stats,numerical_col_analysis,meta_structure)

def Y_analysis(Y,column_name):
    n_rows,n_total,n_null,pct_missing = stats_analyzer(Y)
    target_mean,target_std,target_min,target_max,target_range,target_n_unique,target_unique_ratio = distribution_stats(Y,column_name)
    output_skew_kur_stats = numerical_skew_kurtosis(Y)
    target_outlier_count, target_outlier_pct = outliers(Y,column_name)
    output_display(n_rows,n_total,n_null,pct_missing,target_mean,target_std,target_min,target_max,target_range,target_n_unique,target_unique_ratio,output_skew_kur_stats,target_outlier_count,target_outlier_pct)

if __name__ == "__main__":
    file_path="../../data/Travel.csv"
    df = pd.read_csv(file_path)
    output_column= "MonthlyIncome"

    X=df.drop(columns=[output_column])
    Y=df[[output_column]]

    X_analysis(X)
    Y_analysis(Y,output_column)
    
