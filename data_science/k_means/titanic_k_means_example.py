# The data comes from:
# https://pythonprogramming.net/static/...


import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

df = pd.read_excel("titanic.xls")
print(df)

df.drop(["body", "name"], 1, inplace=True)
# df.apply(pd.to_numeric, errors="ignore")
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

clf = KMeans(n_clusters=2)
clf.fit(x_data)

correct = 0
for i in range(len(x_data)):
    predict_me = np.array(x_data[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y_data[i]:
        correct += 1

print(correct / len(x_data))
