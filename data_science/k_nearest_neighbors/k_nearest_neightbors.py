"""
RECAP:
    With linear regression the objective was to create a model that best fits our data. 

    
OBJECTIVE:
    The objextive with classification is to create a model that best devides or seperates our data.

NOTES:
The objective of this model is to devide or separate our data. It separates points into seperate groups. It closters!. 

Classification != clustering. Clustering is assinging points to seperate groups.In classification the objective is to create a model that properly devides groups, but also predicts which group does a pint belong. A algorithm can be chosen for this.

K = the number of closest points. 

Then a percentage is calculated from the types of K nearest points. 

Confidence equals the percentage of same types points found within the closest  K points. 

Confidence does not equal accuracy. 

Euclid discanec is what used to estimate the closest K points. 

k nearest nightboors is not very good for scaling. Estimations can also be done in paralle

most macine learning algorithms want a category to be a number that represents a type.
"""
# before importing the file I first added column names 

import numpy as np 
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

# first lets clean up import, format and extract the information

df = pd.read_csv('breast-cancer.data')
print('This is the data set just imported and untouched')
print(df)

df.replace('?', -999999, inplace=True)
df.drop(['id'], 1, inplace=True)

# Making both of the arrays out of the dataframe

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])
             
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print(accuracy)  

example_measures = np.array([4, 2, 1, 1, 1, 2, 3, 2, 1])
example_measures = example_measures.reshape(1, -1)

prediction = clf.predict(example_measures)
print(prediction)
