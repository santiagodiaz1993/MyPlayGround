'''
Questions we want to answer:
   What is the  in the future?
Steps we need to take:
Lable the information [done]
Inspect information and get most important attributes
Drop none significant information

'''''

import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv('imports-85.data')
print(df.shape)
print(df)

X = df['price'].replace('?', 0)
X = pd.to_numeric(X)
y = df['symboling']

print('This is the price matrix')
print(X)
print('This is the symboling')
print(y)

plt.scatter(X, y)
plt.title('Price vs Symboling')
plt.xlabel('Price')
plt.ylabel('symboling')
plt.show()
