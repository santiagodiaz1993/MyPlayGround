'''
Questions we want to answer:
   What is the price in the future?
Steps we need to take:
Lable the information [done]
Inspect information and get most important attributes
Drop none significant information

note: A good pactice is to fill values with the avrage of the column. 
This will need to be done where the values are ?'
'''''

import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import math

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
accuracy = clf.score(X_test, y_test)
print(accuracy)

#plt.scatter(dfe, y)
plt.title('Price vs highway mpg')
plt.xlabel('Symboling')
plt.ylabel('Price')
# plt.show()


