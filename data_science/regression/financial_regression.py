import quandl
import pandas as pd


# user will be able to input the stock information.
df = quandl.get('WIKI/GOOGL')

df = df[['Open', 'Low']]

print(df)