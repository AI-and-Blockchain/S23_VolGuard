from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from torch.autograd import Variable 
from torch.utils.data import Dataset, DataLoader
from math import sqrt
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import os
import parquet
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from flask import Flask, request, url_for, redirect, render_template, jsonify


print("begging of file test")
#getting data from amberdata files
#Data provided by Amberdata.io
dataset = pd.read_csv("../hourlyETHUSDCdata.csv", index_col = 'timestamp', parse_dates=True)
print("Dataset loaded")
#model input values
num_epochs = 100 #1000 epochs
learning_rate = 0.001 #0.001 lr
input_size = 4 #number of features
hidden_size = 2 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers
num_classes = 1 #number of output classes
device =  'cpu'
 
target_feature="close"
feature_list=list(dataset.columns.difference([target_feature]))

forecast_amount= 24
seq_length= 24
torch.manual_seed(101)
target = f"{target_feature}_lead{forecast_amount}"

dataset[target] = dataset[target_feature].shift(-10)


# dataset = dataset.iloc[:-10]


#Basic data labeling
test_start_date= "2023-04-05 20:00:00 000"
dataset_train= dataset.loc[:test_start_date].copy()
dataset_test= dataset.loc[test_start_date:].copy()
forecast_train= dataset.loc[test_start_date:].copy()

forecast_train[target] = forecast_train[target_feature].shift(20)
print("Test set fraction:", len(dataset_test) / len(dataset_train))


target_mean = dataset_train[target].mean()
target_stdev = dataset_train[target].std()

for c in dataset_train.columns:
    mean = dataset_train[c].mean()
    stdev = dataset_train[c].std()

    dataset_train[c] = (dataset_train[c] - mean) / stdev
    dataset_test[c] = (dataset_test[c] - mean) / stdev
    
# setting up dataset to work with pytorch dataloader
class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=seq_length):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()
        print("this is test")
        

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i): 
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]
    


train_dataset= SequenceDataset(dataset_train, target=target, features=feature_list, sequence_length= seq_length )
test_dataset= SequenceDataset(dataset_test, target=target, features=feature_list, sequence_length= seq_length)


# Setup the dataloader for the trainer
train_loader= DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader= DataLoader(test_dataset, batch_size=4, shuffle=False)
X, y = next(iter(train_loader))
print("Features shape:", X.shape)
print("Target shape:", y.shape)






class LSTM1(nn.Module):
    def __init__(self, num_sensors, hidden_units):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out
    
if os.path.isfile('./saved_models/hourlylstm2.pt'):
    print("model was found")
    lstm1 = LSTM1(num_sensors=len(feature_list), hidden_units=16)#our lstm class 
    model_state_dict = torch.load('./saved_models/hourlylstm2.pt') # loading the dictionary object
    lstm1.load_state_dict(model_state_dict) # load_state_dict() function takes a dictionary object, NOT a path to a saved object
    lstm1.eval() # since we need to use the model for inference
else:
    print("no saved model found")
    lstm1 = LSTM1(num_sensors=len(feature_list), hidden_units=16)#our lstm class 

# Defining loss function, and optimizing parameters
criterion = torch.nn.MSELoss()    
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate) 

# Trains model for set number of epochs
print("reached training")
def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    
    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")

def test_model(data_loader, model, loss_function):
    
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")


print("Untrained test\n--------")
# test_model(test_loader, lstm1, criterion)
print()

for ix_epoch in range(1):
    print(f"Epoch {ix_epoch}\n---------")
    # train_model(train_loader, lstm1, criterion, optimizer=optimizer)
    # test_model(test_loader, lstm1, criterion)
    print()
    
    
    

def predict(data_loader, model):
    step = 0
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
            for X, _ in data_loader:
                # print(step)
                # print(X.shape)
                # print(data_loader)
                # print("this is x in predict")
                # print(X)
                y_star = model(X)
                
                output = torch.cat((output, y_star), 0)
               
    
    return output

def predict_step(data_loader, model):
    step = 0
    output = torch.tensor([])
    model.eval()
    while step <= 100 :
        with torch.no_grad():
            for X, _ in data_loader:
                # print(step)
                # print(X.shape)
                # print(data_loader)
                # print("this is x in predict")
                # print(X)
                y_star = model(X)
                
                output = torch.cat((output, y_star), 0)
                
                step +=1
    
    return output

def predict_step2(model):
    step = 0
    
    model.eval()

    with torch.no_grad():
        predictions, _ = model(forecast_train[-forecast_amount:])
        print(predictions)
    return predictions

fc_dataset= SequenceDataset(forecast_train, target=target, features=feature_list, sequence_length= 1 )
forecast_loader = DataLoader(fc_dataset, batch_size=4, shuffle=False)

# def make_predictions_from_dataloader(model, unshuffled_dataloader):
#   model.eval()
#   predictions, actuals = [], []
#   for x, y in unshuffled_dataloader:
#     with torch.no_grad():
#       p = model(x)
#       predictions.append(p)
#       actuals.append(y.squeeze())
#   predictions = torch.cat(predictions).numpy()
#   actuals = torch.cat(actuals).numpy()
#   return predictions.squeeze(), actuals

# make_predictions_from_dataloader(lstm1, test_loader)

# predict(test_loader, lstm1)

def one_step_forecast(model, history):
      '''
      model: PyTorch model object
      history: a sequence of values representing the latest values of the time 
      series, requirement -> len(history.shape) == 2
    
      outputs a single value which is the prediction of the next value in the
      sequence.
      '''
      print("begin one step")
      print(history)
      print((history.shape))
      prek = DataLoader(history, batch_size=1, shuffle=False)
      print(prek)
      model.cpu()
      model.eval()
      with torch.no_grad():
        
        pre = torch.Tensor(history).unsqueeze(0)
        print (pre)
        print("that was pre")
        # for X, _ in prek:
        pred = model(pre)
        print(pred)
        print("that was pred")
      return pred.detach().numpy().reshape(-1)

def n_step_forecast(data: pd.DataFrame, target: str, target2:str, tw: int, n: int, forecast_from: int=None, plot=False):
      '''
      n: integer defining how many steps to forecast
      forecast_from: integer defining which index to forecast from. None if
      you want to forecast from the end.
      plot: True if you want to output a plot of the forecast, False if not.
      '''
      print("data")
      history = data.copy()
      
      print(history)
      print("this is history")
      # Create initial sequence input based on where in the series to forecast 
      # from.
      if forecast_from:
        pre = list(history[forecast_from - tw : forecast_from][target].values)
        print(pre)
      else:
        pre = list(history[target])[-tw:]

      # Call one_step_forecast n times and append prediction to history
      for i, step in enumerate(range(n)):
        pre_ = np.array(pre[-tw:]).reshape(-1, 1)
        forecast = one_step_forecast(lstm1, pre_).squeeze()
        pre.append(forecast)
      
      # The rest of this is just to add the forecast to the correct time of 
      # the history series
      res = history.copy()
      ls = [np.nan for i in range(len(history))]

      # Note: I have not handled the edge case where the start index + n is 
      # before the end of the dataset and crosses past it.
      if forecast_from:
        ls[forecast_from : forecast_from + n] = list(np.array(pre[-n:]))
        res['forecast'] = ls
        res.columns = ['actual', 'forecast']
      else:
        fc = ls + list(np.array(pre[-n:]))
        ls = ls + [np.nan for i in range(len(pre[-n:]))]
        ls[:len(history)] = history[target].values
        res = pd.DataFrame([ls, fc], index=['actual', 'forecast']).T
      return res

train_eval_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
# n_step_forecast(dataset_test, 'close', 'close_lead24', 24, 2)
ystar_col = "close forecast"
forecast_train[ystar_col] =predict(forecast_loader, lstm1)
def forecast(model, history_data_loader):
    return null
# print(make_predictions_from_dataloader(lstm1, test_loader))
dataset_train[ystar_col] = predict(train_eval_loader, lstm1).numpy()
dataset_test[ystar_col] = predict(test_loader, lstm1).numpy()
print(predict_step(test_loader, lstm1).numpy()* target_stdev + target_mean)

df_out = pd.concat((dataset_train, dataset_test))[[target, ystar_col]]

tf_out = forecast_train[[target, ystar_col]]

for c in df_out.columns:
    df_out[c] = df_out[c] * target_stdev + target_mean

print(df_out)

for c in tf_out.columns:
     tf_out[c] = tf_out[c] * target_stdev + target_mean
print(tf_out)

print("completed prediciton")

PATH = f'saved_models/hourlylstm2.pt'
# torch.save(lstm1.state_dict(), PATH)
print("model state saved")

def get_volatility_forecast():
        i = 0
        forecast_data =predict_step(forecast_loader,lstm1)
        volatility_data = predict_step(test_loader, lstm1).numpy()* target_stdev + target_mean
        
        
        print(volatility_data)
        size_of_data = len(volatility_data)
        mean = sum(forecast_data)/len(forecast_data)
        variance = sum([((x - mean) ** 2) for x in volatility_data]) / len(volatility_data)
        standard_deviation = variance ** 0.5
        
        # volatility_prediction = standard_deviation/
        return  str((standard_deviation)*size_of_data/(mean))


def get_value():
    predict_step(forecast_loader, lstm1)

return_value = ''


app = Flask(__name__)
app.add_url_rule('/', 'index', (lambda: return_value))
app.add_url_rule('/predict', 'predict', (lambda: return_value +  get_volatility_forecast()))
@app.route('/')
def hello_world():
   return "Hello, World!"

if __name__ == "__main__":
   app.run(host='0.0.0.0', port=5000)