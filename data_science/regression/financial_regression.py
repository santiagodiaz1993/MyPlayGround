# Definitions:
# Feature = an input
# Label = output
# abnormal = peace of information far from the avrage and likley 
# not used to estimations.

# for each epoch
#     for each training data instance
#         propagate error through the network
#         adjust the weights
#         calculate the accuracy over training data
#     for each validation data instance
#         calculate the accuracy over the validation data (make sure is not over fitting)
#     if the threshold validation accuracy is met
#         exit training
#     else
#         continue training
# Training Set: this data set is used to adjust the weights on the neural network.
# Validation Set: this data set is used to minimize overfitting. You're not adjusting the 
# weights of the network with this data set, you're just verifying that any increase in 
# accuracy over the training data set actually yields an increase in accuracy over a data 
# set that has not been shown to the network before, or at least the network hasn't trained 
# on it. 
# Testing Set: this data set is used only for testing the final solution in order to confirm the actual predictive power of the network.

import quandl, math, datetime
import matplotlib as plt
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

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
print('New column with the highs lows percentage')
print('HL_PCT shows (high - close) / close * 100 under the column HL_PCT')
print(df)

df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100
print('New column with the change percente')
print('PCT_change shows (close - Open) / Open * 100 under the column PCT_change') 
print(df)

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print('This is the table of features that we will be working on')
print('The table includes the most relevant characteristics about the stock that matters the most')
print('The table incluse Close + high low percente + percentage change')
print(df)

forecast_col = 'Adj. Close'
# Replace empty (N/A) cells with -9999. 
# fillna --> replace n/a values with the number given. This is done not to 
# loose data
df.fillna(-9999, inplace=True)

# convert to int --> round number up --> 0.01 * length (number of rows) of the df
# this line spits out an integer
forecast_out = int(math.ceil(0.01*len(df)))

# shift the rows in the column 'Adj. Close' by forecast_out towards the future (move values of the column up)  
# forecast_out = 35 days
df['label'] = df[forecast_col].shift(-forecast_out)


# Summary:
# We have filtered and estimated the most meaningful/non-meaningful features of the stock, 
# and then created a label column called which is a column with values
# shifter 35 days into the future. 


# create array from the df but without the lable column for X 
X = np.array(df.drop(['label'], 1))
print('This is array X w/o column label')
print(X)

# preprocessing.scale will normalize data. This is likley done by making all data 
# points avaliable in a scale that linearly represent their value within a usually
# smaller range 
X = preprocessing.scale(X)
print('This is array X but after being normalized')
print(X)

# 
X = X[:-forecast_out]
print('taking the last # (forecast_out) of rows for all columns')
print('the result is:')
print(X)

X_lateley = X[-forecast_out:]
print('taking the first # (forecast_out) of rows for all columns')
print('the result is:')
print(X_lateley)

# get rid of n/a cells 
df.dropna(inplace=True)

# create array from the df but only for the lable column for y 
y = np.array(df['label'])
print('This is array y with only column label')

# train_test_split will randomize within variables
# test size is percentage that will be extracted.
# this line 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print('This is X train and X train')
print(X_train)
print(X_test)

# choose the algorithm. The algorithm is a class. Documentation for each 
# algorithm can be found in sklearn
clf = LinearRegression(n_jobs = 1)
# fit = train
clf.fit(X_train, y_train)
# score = test
accuracy = clf.score(X_test, y_test)

print(accuracy)

forecast_set = clf.predict(X_lateley)

print(forecast_set, accuracy, forecast_out)


df['forecast'] = np.nan

last_date = df.iloc[-1].name

last_unix = last_date.timestamp()

one_day = 8640

next_unix = last_unix + one_day

for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]