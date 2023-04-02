from math import sqrt
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


print("beginning of file test")
#getting data from amberdata files
#Data provided by Amberdata.io
pd.options.display.max_rows = 100
dataset = pd.read_csv("../../historical_files/ETHUSDCdata.csv", index_col = 'timestamp', parse_dates=True)
print("Dataset loaded")
print(dataset.head(5))

plt.style.use('ggplot')
dataset['volume'].plot(label='CLOSE', title='ETH-USDC Volume')

#model input values
num_epochs = 1000 #1000 epochs
learning_rate = 0.001 #0.001 lr
input_size = 5 #number of features
hidden_size = 2 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers
num_classes = 1 #number of output classes