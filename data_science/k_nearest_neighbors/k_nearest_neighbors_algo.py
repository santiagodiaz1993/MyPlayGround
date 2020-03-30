''' 
In this excercise we are comparing a manually written k nearest neigthbor
algorithm to the one that comes built in into numpy. 



Eucliean distance: sqrt(The sum to n starting from i=1 (Ai - Pi)**2)

Example:
a = (1, 3)
p = (2, 5)
This means we plug it and get:
    sqrt((1-2)**2 + (3-5)**2)
    Notes: 
        inplace=True -> does operation but nothing is returned. When 
inplace=Flase it performs the opreation and returns a new copy of the data
'''

from math import  sqrt
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import style
from collections import Counter
import pandas as pd
import random

style.use('fivethirtyeight')

plot1 = [1, 3]
plot2 = [2, 5]

euclidean_distance = sqrt( (plot1[0] - plot2[0])**2 + (plot1[1] - plot2[1])**2)

dataset = {'k':[[1, 2], [2, 3], [3, 1]], 'r':[[6, 5], [7, 7], [8, 6]]}
new_feature = [5, 7]

for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0], ii[1], color=i)

plt.scatter(new_feature[0], new_feature[1])
#plt.show()

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        print('K is set to the calue less than total voting groups')
    distances = []
    for group in data:
        for feature in data[group]:    
            # eucliean_distance = sqrt(The sum to n starting from i=1 (Ai - Pi)**2)
            euclidean_distance = np.linalg.norm(np.array(feature) - np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result

result = k_nearest_neighbors(dataset, new_feature, k=3)
print('This is the result')
print(result)

[[plt.scatter(ii[0], ii[1], color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_feature[0], new_feature[1])
# plt.show()


df = pd.read_csv('breast-cancer.data')
df.replace('?', -9999, inplace=True)
df.drop(['id'], 1, inplace=True)
print(df)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

train_size = 0.2
train_set =


