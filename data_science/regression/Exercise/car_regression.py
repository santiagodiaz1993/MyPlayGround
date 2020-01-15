'''
LINEAR REGRESSION MINI-PROJECT

Questions we want to answer:
   What is the price in the future?
Steps we need to take:
1. Clean all of the data that have missing cells. For this I
decided to drop all rows with missing values
2. Graph all data against the price to see what are the
factors that have the most interaction.
3. Replace all none numeric values in cell with numeric values.
for this I used a dictionary.
4. Randomized all data and selected a training and testing set. 
The testing set is 20% of the values out of the total values 
after dropping the rows in step 1.
5. I trained and tested the model. It has an accuracy of 77%. 
I decided to use all of the features avaliable.

note: A good pactice is to fill values with the avrage of the column. 
However, I did not do that in this excercise. 
'''''

import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import math
import pickle

df = pd.read_csv('imports-85.data')

'''Clean the data. I am dropping the rows that have a missing value in them'''
df = df[df.normalized_losses != '?']
df = df[df.num_of_doors != '?']
df = df[df.bore != '?']
df = df[df.stroke != '?']
df = df[df.horsepower != '?']
df = df[df.peak_rpm != '?']

print('This is the dataframe with all of the data cleaned')
print('The rows with missing values were dropped')
print(df)

''' Look for interaction between all factors against price '''
y = df['price']
y = pd.to_numeric(y)
y = np.array(y)
print('This will be the label')
print(y)

for factor in df:
    colum_X = df[factor].sort_values(ascending=True)
    plt.title(factor + ' vs price')
    plt.xlabel(factor)
    plt.ylabel('price')
    plt.scatter(colum_X, y)
    # plt.show()

print(df)
''' Pick all factors that will be used to predict the car's price based on the interaction
from all of the factors'''

X = df.drop(['price'], 1)
print('These will be the features')
print(X)

''' Now sample, randomize, experiment, train, test and create the model '''

make_dict = {'alfa-romero': 0, 
             'audi':1, 
             'bmw':2, 
             'chevrolet':3, 
             'dodge':4, 
             'honda':5, 
             'isuzu':6, 
             'jaguar':7,
             'mazda':8, 
             'mercedes-benz':9, 
             'mercury':10, 
             'mitsubishi':11, 
             'nissan':12, 
             'peugot':13, 
             'plymouth':14, 
             'porsche':15,
             'renault':16, 
             'saab':17, 
             'subaru':18, 
             'toyota':19, 
             'volkswagen':20, 
             'volvo':21,
            }

X['make'] = X.make.map(make_dict)

X['fuel_type'] = X.fuel_type.map({'diesel':0, 'gas':1})

X['aspiration'] = X.aspiration.map({'std':0, 'turbo':1})

X['num_of_doors'] = X.num_of_doors.map({'four':0, 'two':1})

X['body_style'] = X.body_style.map({'hardtop':0, 'wagon':1, 'sedan':2, 'hatchback':3, 'convertible':4})

X['drive_wheels'] = X.drive_wheels.map({'4wd':0, 'fwd':1, 'rwd': 2})

X['engine_location'] = X.engine_location.map({'front':0, 'rear':1})

X['engine_type'] = X.engine_type.map({'dohc':0, 'dohcv':1, 'l':2, 'ohc':3, 'ohcf': 4, 'ohcv':5, 'rotor': 6})

X['num_of_cylinders'] = X.num_of_cylinders.map({'eight':0, 'five':1, 'four':2, 'six':3, 'three':4, 'twelve':6, 'two': 7})

X['fuel_system'] = X.fuel_system.map({'1bbl':0, '2bbl':1, '4bbl':2, 'idi':3, 'mfi':4, 'mpfi': 5, 'spdi':6, 'spfi': 7})

print('This is the data frame with all variables converted to numerical values')
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)
   
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
print(accuracy)

