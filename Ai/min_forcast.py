from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
from torch.autograd import Variable 
from math import sqrt
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import math
import parquet
import os
from datetime import datetime
import locale
import matplotlib.pyplot as plt
from matplotlib import style


print("begging of file test")
#getting data from amberdata files
#Data provided by Amberdata.io
dataset = pd.read_csv("/../min_data.csv", index_col = 'timestamp', parse_dates=True)
real_dataset = pd.read_csv("/../min_04_01_data.csv",index_col = 'timestamp', parse_dates=True)

print("Dataset loaded")
print(dataset.columns)

dataset['HL_PCT'] = (dataset['high'] - dataset['low']) / dataset['close'] * 100.0
dataset['PCT_change'] = (dataset['close'] - dataset['open']) / dataset['open'] * 100.0


df = dataset[['close', 'HL_PCT', 'PCT_change', 'volume']]
forecast_col = 'close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.5 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)


X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=5)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)

forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_date = datetime.strptime(last_date, '%Y-%m-%d %H:%M:%S %f') 
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['close'].plot()
df['Forecast'].plot()
#print(df['Forecast'].tolist())

count = 0

for i in df['Forecast'].tolist():
    if not (np.isnan(i)):
        print("{:.2f}".format((real_dataset.iloc[count]["close"]/i) * 100))
        count+=1
print(count)


plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

