# Definitions:
# Feature = an input
# Label = output
# abnormal peace of information. Likley not used to estimations.

import quandl, math
import matplotlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression


df = quandl.get('WIKI/GOOGL')
print('This is the full table')
print(df)

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
print('This is the subset dataframwork')
print(df)

df['HL_PCT'] = df['Adj. High'] - df['Adj. Close'] / df['Adj. Close'] 
print('This is the high low percentage')
print('This shows high - close / close * 100 under the column HL_PCT')
print(df)

df['PCT_change'] = df['Adj. Close'] - df['Adj. Open'] / df['Adj. Open']
print('This is the change percente')
print('This is shows close - Open / Open * 100 under the column PCT_change') 
print(df)

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print('This is the table that we will be working on')
print('The table includes the characteristics about the stock that matters the most')
print('The table incluse Close + high low percente + percentage change')
print(df)

forecast_col = 'Adj. Close'
# Replace empty cells with a random number. In this ase the number is -9999
# fillna --> replace n/a values with the number given. This is done not to 
# loose data
df.fillna(-9999, inplace=True)

# convert to int --> round number up --> 0.1 * length (number of rows) of the df
forecast_out = int(math.ceil(0.01*len(df)))

# shift the rows in the column 'Adj. Close' by forecast_out towards the future (move values of the column up)  
# forecast_out = 35 days
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)


# Summary:
# We have filtered/estimated the most meaningful features of the stock, 
# and then created a column called "label" which is a column with values
# shifter 35 days into the future. 


# create array from the df but without the lable column for X 
X = np.array(df.drop(['label'], 1))
# create array from the df but but only for the lable column for y 
y = np.array(df['label'])

# preprocessing.scale will normalize data this is likley done by making all data 
# points avaliable in a scale that linearly represent their value within a usually
# smaller range 
X = preprocessing.scale(X)

# train_test_split will randomize within variables
# test size is percentage that will be extracted.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# choose the algorithm. The algorithm is a class.

clf = LinearRegression()
# fit = train
clf.fit(X_train, y_train)
# score = test
accuracy = clf.score(X_test, y_test)

print(accuracy)