# The data comes from:
# https://pythonprogramming.net/static/...


import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing
import pandas as pd

df = pd.read_excel("titanic.xls")
original_df = pd.DataFrame.copy(df)

df.drop(["body", "name"], 1, inplace=True)

for c in df.columns.values:
    df[c] = pd.to_numeric(df[c], errors="ignore")
df.fillna(0, inplace=True)


def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))
    return df


df = handle_non_numerical_data(df)
print(df)

x_data = np.array(df.drop(["survived", "pclass"], 1).astype(float))
x_data = preprocessing.scale(x_data)
y_data = np.array(df["survived"])

clf = MeanShift()
clf.fit(x_data)

labels = clf.lables_
cluster_centers = clf.cluster_centers_

original_df["cluster_group"] = np.nan

for i in range(len(x_data)):
    original_df["cluster_group"].iloc[i] = labels[i]

survival_rates = {}
n_clusters_ = len(np.unique(labels))

for i in range(n_clusters_):
    temp_df = original_df[(original_df["cluster_group"] == float(i))]
    survival_cluster = temp_df[(temp_df["survived"] == 1)]
    survival_rate = len(survival_cluster) / len(temp_df)
    survival_rates[i] = survival_rate

print(survival_rates)
