''' 
Eucliean distance: sqrt(The sum to n starting from i=1 (Ai - Pi)**2)

a = (1, 3)

p = (2, 5)

This means we plug it and get:

    sqrt((1-2)**2 + (3-5)**2)

    '''
from math import  sqrt
import numpy as np
import matplotlib.pyplot as plt 
import warnings
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')

plot1 = [1, 3]
plot2 = [2, 5]

# numpy has a function that does this computation. It will be used bc is much faster.

euclidean_distance = sqrt( (plot1[0] - plot2[0])**2 + (plot1[1] - plot2[1])**2)

print('This is the distance between this two plots')
print(euclidean_distance)


dataset = {'k':[[1, 2], [2, 3], [3, 1]], 'r':[[6, 5], [7, 7], [8, 6]]}
new_features = [5, 7]

for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0], ii[1], color=i)

plt.scatter(new_features[0], new_features[1])
plt.show()

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnning.warn('K is set to the calue less than total voting groups')
        knnalgos
        return vote_result
