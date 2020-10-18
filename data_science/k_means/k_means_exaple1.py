# The data comes from:
# https://pythonprogramming.net/static/...


import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

df = pd.read_excel("titanic.xls")
print(df)

df.drop(["body", "name"], 1, inplace=True)
df.apply(pd.to_numeric, errors="ignore")
df.fillna(0, inplace=True)


def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_value_vals = {}

        def convert_to_int(val):
            return text_digit_value_vals

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_value_vals:
                    text_digit_value_vals
                    x += 1

        df[column] = list(map(convert_to_int, df[column]))
        return df


df = handle_non_numerical_data(df)
print(df)
