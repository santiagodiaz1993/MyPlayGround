"""
RECAP:
    With linear regression the objective was to find the best fit line, and use it to predict
values.

OBJECTIVE:
    The objective with classification is to create a model that best devides or seperates
our data.

NOTES:
The objective of this model is to classify a given point of data  based on the type
of closest points. K is the number of points that the model will take into consideration
to assign a type. The type will be assigned based on the type of the majority of the
points. Each point has a degree of *confidence*. Confidence defferentiates from accuracy
in which confidence is the percentage of data points that fall under the expected
assigned type. Accuracy is the entires model (All data points) precision.

Example:
    If the 2 out of the 3 (K) closest points have the same expected atribute, then confidence for
for this data point is 66%

    Euclid discanec: is what used to estimate the closest K points. This is done by measuring the
distance between a given point and the remaining ones and picking the K closest ones. This
is a very power hungry process.(+/-) up to a gigabite of data it is still okay to use.

Note: Trainning and testing is the same.
Deffinition: Outlier: is an observation that lies an abnormal distance from the other values in a
random sample from a population.

"""

import numpy as np
import pandas as pd
import pickle
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split

df = pd.read_csv("breast-cancer.data")

df.replace("?", -999999, inplace=True)
df.drop(["id"], 1, inplace=True)

full_data = df.astype(float).values.tolist()

X = np.array(df.drop(["class"], 1))
y = np.array(df["class"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train)
print(y_train)

# clf = neighbors.KNeighborsClassifier()
# clf.fit(X_train, y_train)

# with open('k_nearest_classifier.pickle', 'wb') as f:
#     pickle.dump(clf, f)
pickle_in = open("k_nearest_classifier.pickle", "rb")
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
print("This is the accruacy:")
print(accuracy)

example_measures = np.array([4, 2, 1, 1, 1, 2, 3, 2, 1])
example_measures = example_measures.reshape(1, -1)

prediction = clf.predict(example_measures)
print("This is the prediction given the example measure:")
print(prediction)
