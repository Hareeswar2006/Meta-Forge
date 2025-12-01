import os
import json
import pandas as pd
import csv
import numpy as np


def stats_analyzer(file_path):
    df=pd.read_csv(file_path)
    total_values=df.size
    total_null_values=df.isna().sum().sum()
    pct_total = (total_null_values/total_values)*100

    n_rows=df.shape[0]
    n_cols=df.shape[1]
    numerical_df = df.select_dtypes(include='number')
    categorical_df = df.select_dtypes(include=['object','category'])
    n_numerical_rows = len(numerical_df)
    n_categorical_rows = len(categorical_df)

    numerical_null={}
    categorical_null={}
    for column in numerical_df.columns:
        numerical_null[column] = int(numerical_df[column].isna().sum())
    for column in categorical_df.columns:
        categorical_null[column] = int(categorical_df[column].isna().sum())

    return [
            n_rows, n_cols, n_numerical_rows, n_categorical_rows, numerical_df, numerical_null,
            categorical_df, categorical_null, total_values, total_null_values, pct_total
            ]

if __name__ == "__main__":
    file_path="../data/Travel.csv"
    nrows,ncols,nnumerical,ncategorical,numerical_df,numerical_null,categorical_df,categorical_null,total_values,total_null_values,pct_total = stats_analyzer(file_path)
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