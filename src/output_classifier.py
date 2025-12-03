import pandas as pd
import numpy as np
import csv
import pandas.api.types as ptypes 

def classify(file_path,column_name):
    if column_name is None or column_name == "None":
        return "clustering"
    
    df = pd.read_csv(file_path)
    output_column = df[column_name]
    print(output_column.dtype)
    unique_vals = int(output_column.nunique())
    is_numeric = ptypes.is_numeric_dtype(output_column)

    if not is_numeric:
        prob_type = "classification"

    elif unique_vals<=20:
        prob_type="classification"
    
    else:
        prob_type="regression"

    return prob_type

if __name__=="__main__":
    file_path = "../data/spam.csv"
    column_name = "v1"
    output_type = classify(file_path, column_name)

    print("Problem type:  ",output_type)