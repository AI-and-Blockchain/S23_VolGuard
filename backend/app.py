from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from math import sqrt
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from flask import Flask, request, url_for, redirect, render_template, jsonify, send_from_directory
from datetime import datetime, timedelta, date
from flask_cors import CORS
from web3 import Web3, HTTPProvider
import web3
import pandas as pd
import time
import requests
import csv
import math
import deploy2
import solcx
from solcx import compile_source, compile_files
import json
from threading import Timer

print("begging of file test")
#getting data from amberdata files
#Data provided by Amberdata.io

def setup():
    dataset = pd.read_csv("./mysite/hourlyETHUSDCdata.csv", index_col='timestamp', parse_dates=True)
    print("Dataset loaded")


    # def get_new_data():


    #     url = "https://web3api.io/api/v2/market/defi/ohlcv/0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640/historical/?exchange=uniswapv3&startDate=2023-03-17&endDate=2023-04-17&timeInterval=hours&format=csv&fields=timestamp%2Copen%2Chigh%2Clow%2Cvolume%2Cclose&timeFormat=human_readable"

    #     headers = {
    #         "accept": "application/json",
    #         "x-api-key": "UAK7ed69235426c360be22bfc2bde1809b6"
    #     }

    #     response = requests.get(url, headers=headers)
    #     f = open('./mysite/hourlyETHUSDCdata.csv', "w")
    #     f.write(response.text)
    #     f.close()
    #     #getting data from amberdata files
    #     #Data provided by Amberdata.io
    #     dataset = pd.read_csv('./mysite/hourlyETHUSDCdata.csv', index_col = 'timestamp', parse_dates=True)
    #     print("Dataset loaded")
    #     return("loaded new dataset")

    def daterange(date1, date2):
        for n in range(int ((date2 - date1).days)+1):
            yield date1 + timedelta(n)

    def datetime_range(start=datetime(2023, 3, 6, 0, 0, 0), end=datetime(2023, 4, 9, 23, 0, 0)):
        span = end - start
        for i in range((span.days)*24 + 1):
            yield start + timedelta(hours=i)

    def to_date_time(timestamp):
        datetime_str = time.mktime(timestamp)
        format = '%Y-%m-%d %H:%M:%S' # The format
        print(timestamp)
        dateTime = time.strftime(format, time.gmtime(datetime_str))
        return dateTime


    # index_dates = []
    # for date in datetime_range():
    #     dataset = dataset.append(pd.DataFrame([['timestamp', date], ['open', 0], ['high', None], ['low', 0], ['volume', 0], ['close', 0]]))
    # print(datetime_range())
    # dataset['datetime'] = index_dates

    # dataset['timestamp'].loc[len(index_dates)] = index_dates
    # dataset_i2 = pd.Index(index_dates)

    # dataset.index.append(dataset_i2)
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


    # for index in dataset:
    #     print(index)
    #     if index in feature_list:
    #         continue
    #     if index == 'close':
    #         continue
    #     dataset.index = to_date_time(index)

    print(dataset)
    forecast_amount= 12
    seq_length= 12
    torch.manual_seed(101)
    target = f"{target_feature}_lead{forecast_amount}"

    dataset[target] = dataset[target_feature].shift(0)
    # first_date = dataset['timestamp'].min()
    today = datetime.today()

    # dataset.set_index('timestamp', inplace=True)
    # idx = pd.date_range(first_date, today, freq='D')
    # df = dataset.reindex(idx)
    # dataset = dataset.iloc[:-10]


    #Basic data labeling
    test_start_date= "2023-04-01 00:00:00 000"
    forecast_start_date= dataset.index[-seq_length]
    dataset_train= dataset.loc[:test_start_date].copy()
    dataset_test= dataset.loc[test_start_date:forecast_start_date].copy()
    dataset_forecast= dataset.loc[forecast_start_date:].copy()
    print("dataset_train")
    # print(dataset_train)
    print("dataset_test")
    # print(dataset_test)
    print("dataset_forecast")
    # dataset_forecast[target] = dataset_forecast[target_feature].shift(-4)
    # dataset_forecast[target_feature]=dataset_forecast[target_feature].shift(-4)
    print(dataset_forecast)
    # dataset_forecast[target] = dataset_forecast[target_feature].shift(20)
    print("Test set fraction:", len(dataset_test) / len(dataset_train))



    target_mean1 = dataset_train[target].mean()
    target_stdev1 = dataset_train[target].std()

    for c in dataset_train.columns:
        mean = dataset_train[c].mean()
        stdev = dataset_train[c].std()

        dataset_train[c] = (dataset_train[c] - mean) / stdev
        dataset_test[c] = (dataset_test[c] - mean) / stdev
        dataset_forecast[c] = (dataset_forecast[c] - mean) / stdev

    # setting up dataset to work with pytorch dataloader
    class SequenceDataset(Dataset):
        def __init__(self, dataframe, target, features, sequence_length=seq_length):
            self.features = features
            self.target = target
            self.sequence_length = sequence_length
            self.y = torch.tensor(dataframe[target].values).float()
            self.X = torch.tensor(dataframe[features].values).float()


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
    forecast_dataset =SequenceDataset(dataset_forecast, target=target, features=feature_list, sequence_length= seq_length)

    # Setup the dataloader for the trainer
    train_loader= DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader= DataLoader(test_dataset, batch_size=4, shuffle=False)
    forecast_loader= DataLoader(forecast_dataset, batch_size=6, shuffle=False)
    X, y = next(iter(train_loader))
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)
    X, y = next(iter(test_loader))
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)
    X, y = next(iter(forecast_loader))
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

    if os.path.isfile('./mysite/hourlylstm2.pt'):
        print("model was found")
        lstm1 = LSTM1(num_sensors=len(feature_list), hidden_units=16)#our lstm class
        model_state_dict = torch.load('./mysite/hourlylstm2.pt') # loading the dictionary object
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


    # print("Untrained test\n--------")
    # test_model(test_loader, lstm1, criterion)
    # print()

    for ix_epoch in range(1):
        # print(f"Epoch {ix_epoch}\n---------")
        # train_model(train_loader, lstm1, criterion, optimizer=optimizer)
        # test_model(test_loader, lstm1, criterion)
        print()
    return("setup done")

setup()

def predict(data_loader, model):
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
            for X, y in data_loader:

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
    with torch.no_grad():
            for X, y in data_loader:
                # print("INPUT this is the y value INPUT")
                # print(str((y * target_stdev + target_mean)*1000))
                # print("INPUT")
                # print("-------------------------------")
                # print((X * target_stdev + target_mean)*1000)
                # print("-------------------------------")
                # print(step)
                # print(X.shape)
                # print(data_loader)
                # print("this is x in predict")
                # print(X)
                y_star = model(X)
                # print("OUTPUT         OUTPUT")
                # print("this is what model(X) gives")
                # print(str((y_star * target_stdev + target_mean)*1000))
                # print("OUTPUT         OUTPUT")
                output = torch.cat((output, y_star), 0)
                step +=1
    return output
def shift_df(new_value_list, data_set, dataloader, datasequence):
    return True

forecast_values = []

def forecast(model, data_set, data_sequence, data_loader, forecast_epochs, target_mean_val, target_stdev_val):
    predicted_values =[]
    last_val = data_set.iloc[-1,-1]
    if forecast_epochs >=1:
        predicted_values = (predict_step(data_loader, model).numpy()* target_stdev_val + target_mean_val)

        for row in predicted_values:
            # data_set.loc[len(data_set.index)]
            new_frame=pd.DataFrame ([["2023-04-20 00::00::00", last_val, max(row, last_val), min(row,last_val), 2192817.561562, row ] ])
            last_val = row
            data_set.append(new_frame)
            data_set.drop(index=data_set.index[0], axis=0, inplace=True)
        # new_datasequence =SequenceDataset(data_set, target=target, features=feature_list, sequence_length= seq_length)
        new_loader = DataLoader(data_sequence, batch_size=12, shuffle=False)
        return predicted_values;
    # concat(forecast(model, data_set.copy(), data_sequence, new_loader))
        forecast_epochs -=1;
    return predicted_values





def predict_step2(model):
    step = 0

    model.eval()

    with torch.no_grad():
        predictions, _ = model(dataset_forecast[-forecast_amount:])
        print(predictions)
    return predictions

# fc_dataset= SequenceDataset(dataset_forecast, target=target, features=feature_list, sequence_length= 1 )
# forecast_loader = DataLoader(fc_dataset, batch_size=4, shuffle=False)

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
def test():

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

def prediction_setup():
    train_eval_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    # n_step_forecast(dataset_test, 'close', 'close_lead24', 24, 2)
    ystar_col = "close forecast"

    # def forecast(model, history_data_loader):
    #     return null
    # print(make_predictions_from_dataloader(lstm1, test_loader))
    forecast_values= forecast(lstm1, dataset_forecast, forecast_dataset, forecast_loader, 1)
    dataset_forecast[ystar_col]= forecast_values
    print("this is dataset_forecast[ysar_col]")
    # print(dataset_forecast[ystar_col]* target_stdev + target_mean)
    dataset_train[ystar_col] = predict(train_eval_loader, lstm1).numpy()
    print("this is dataset_train[ysar_col]")
    # print(dataset_train[ystar_col]* target_stdev + target_mean)
    dataset_test[ystar_col] = predict(test_loader, lstm1).numpy()
    print("this is dataset_test[ysar_col]")
    # print(dataset_test[ystar_col]* target_stdev + target_mean)
    print(predict_step(forecast_loader, lstm1).numpy()* target_stdev + target_mean)

    df_out = pd.concat((dataset_train, dataset_test))[[target, ystar_col]]

    tf_out = dataset_forecast[[target, ystar_col]]

    for c in df_out.columns:
        df_out[c] = df_out[c] * target_stdev + target_mean

    # print(df_out)

    for c in tf_out.columns:
        tf_out[c] = tf_out[c] * target_stdev + target_mean
    # print(tf_out)

    print("completed prediciton")

    PATH = f'hourlylstm2.pt'
    # torch.save(lstm1.state_dict(), PATH)
    print("model state saved")

def get_volatility_forecast():
        i = 0
        forecast_data = forecast_values
        print(forecast_data)
        # volatility_data = predict_step(test_loader, lstm1).numpy()* target_stdev + target_mean


        print(forecast_data)
        size_of_data = len(forecast_data)
        mean = sum(forecast_data)/len(forecast_data)
        variance = sum([((x - mean) ** 2) for x in forecast_data]) / len(forecast_data)
        standard_deviation = variance ** 0.5
        return  str((standard_deviation)*(math.sqrt(24))/(mean))
        # volatility_prediction = standard_deviation/



def get_volatility_historical():
        real_data = df_out[target]
        print(real_data)
        # volatility_data = predict_step(test_loader, lstm1).numpy()* target_stdev + target_mean


        print(real_data)
        size_of_data = len(real_data)
        mean = sum(real_data)/len(real_data)
        variance = sum([((x - mean) ** 2) for x in real_data]) / len(real_data)
        standard_deviation = variance ** 0.5

        # volatility_prediction = standard_deviation/
        return  str((standard_deviation)*(math.sqrt(24))/(mean))
        # volatility_prediction = standard_deviation/
        # return  str((standard_deviation)*size_of_data/(mean))


def get_value():
    predict_step(forecast_loader, lstm1)

return_value = ''

def confidence_interval(num,num_l):
    SE = np.std(num_l)/(len(num_l)**2)
    lower_bound = num - 1.96*SE
    upper_bound = num + 1.96*SE

    return lower_bound,upper_bound

def create_app():
    app = Flask(__name__, static_folder="./frontend/build", static_url_path="/")
    CORS(app)
    dataset = pd.read_csv("./mysite/hourlyETHUSDCdata.csv", index_col='timestamp', parse_dates=True)
    print("Dataset loaded")

    # def daterange(date1, date2):
    #     for n in range(int ((date2 - date1).days)+1):
    #         yield date1 + timedelta(n)

    # def datetime_range(start=datetime(2023, 3, 6, 0, 0, 0), end=datetime(2023, 4, 9, 23, 0, 0)):
    #     span = end - start
    #     for i in range((span.days)*24 + 1):
    #         yield start + timedelta(hours=i)

    # def to_date_time(timestamp):
    #     datetime_str = time.mktime(timestamp)
    #     format = '%Y-%m-%d %H:%M:%S' # The format
    #     print(timestamp)
    #     dateTime = time.strftime(format, time.gmtime(datetime_str))
    #     return dateTime

    num_epochs = 100 #1000 epochs
    learning_rate = 0.001 #0.001 lr
    input_size = 4 #number of features
    hidden_size = 2 #number of features in hidden state
    num_layers = 1 #number of stacked lstm layers
    num_classes = 1 #number of output classes
    device =  'cpu'

    target_feature="close"
    feature_list=list(dataset.columns.difference([target_feature]))

    print(dataset)
    forecast_amount= 12
    seq_length= 12
    test_start_length = seq_length*2
    torch.manual_seed(101)
    target = f"{target_feature}_lead{forecast_amount}"

    dataset[target] = dataset[target_feature].shift(0)
    # first_date = dataset['timestamp'].min()
    today = datetime.today()

    #Basic data labeling
    test_start_date= "2023-04-01 00:00:00 000"
    # forecast_start_date= dataset.index[-seq_length]
    # dataset_train= dataset.loc[:test_start_date].copy()
    # dataset_test= dataset.loc[test_start_date:forecast_start_date].copy()
    # dataset_forecast= dataset.loc[forecast_start_date:].copy()
    print("dataset_train")
    # print(dataset_train)
    print("dataset_test")
    # print(dataset_test)
    print("dataset_forecast")
    # dataset_forecast[target] = dataset_forecast[target_feature].shift(-4)
    # dataset_forecast[target_feature]=dataset_forecast[target_feature].shift(-4)
    # print(dataset_forecast)
    # dataset_forecast[target] = dataset_forecast[target_feature].shift(20)
    # print("Test set fraction:", len(dataset_test) / len(dataset_train))



    # target_mean = dataset_train[target].mean()
    # target_stdev = dataset_train[target].std()

    # for c in dataset_train.columns:
    #     mean = dataset_train[c].mean()
    #     stdev = dataset_train[c].std()

    #     dataset_train[c] = (dataset_train[c] - mean) / stdev
    #     dataset_test[c] = (dataset_test[c] - mean) / stdev
    #     dataset_forecast[c] = (dataset_forecast[c] - mean) / stdev

    # setting up dataset to work with pytorch dataloader
    class SequenceDataset(Dataset):
        def __init__(self, dataframe, target, features, sequence_length=seq_length):
            self.features = features
            self.target = target
            self.sequence_length = sequence_length
            self.y = torch.tensor(dataframe[target].values).float()
            self.X = torch.tensor(dataframe[features].values).float()


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



    # train_dataset= SequenceDataset(dataset_train, target=target, features=feature_list, sequence_length= seq_length )
    # test_dataset= SequenceDataset(dataset_test, target=target, features=feature_list, sequence_length= seq_length)
    # forecast_dataset =SequenceDataset(dataset_forecast, target=target, features=feature_list, sequence_length= seq_length)

    # Setup the dataloader for the trainer
    # train_loader= DataLoader(train_dataset, batch_size=4, shuffle=True)
    # test_loader= DataLoader(test_dataset, batch_size=4, shuffle=False)
    # forecast_loader= DataLoader(forecast_dataset, batch_size=6, shuffle=False)
    # X, y = next(iter(train_loader))
    # print("Features shape:", X.shape)
    # print("Target shape:", y.shape)
    # X, y = next(iter(test_loader))
    # print("Features shape:", X.shape)
    # print("Target shape:", y.shape)
    # X, y = next(iter(forecast_loader))
    # print("Features shape:", X.shape)
    # print("Target shape:", y.shape)


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

    if os.path.isfile('./mysite/hourlylstm2.pt'):
        print("model was found")
        lstm1 = LSTM1(num_sensors=len(feature_list), hidden_units=16)#our lstm class
        model_state_dict = torch.load('./mysite/hourlylstm2.pt') # loading the dictionary object
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
    # def train_model(data_loader, model, loss_function, optimizer):
    #     num_batches = len(data_loader)
    #     total_loss = 0
    #     model.train()

    #     for X, y in data_loader:
    #         output = model(X)
    #         loss = loss_function(output, y)

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         total_loss += loss.item()

    #     avg_loss = total_loss / num_batches
    #     print(f"Train loss: {avg_loss}")

    # def test_model(data_loader, model, loss_function):

    #     num_batches = len(data_loader)
    #     total_loss = 0

    #     model.eval()
    #     with torch.no_grad():
    #         for X, y in data_loader:
    #             output = model(X)
    #             total_loss += loss_function(output, y).item()

    #     avg_loss = total_loss / num_batches
    #     print(f"Test loss: {avg_loss}")


    # print("Untrained test\n--------")
    # test_model(test_loader, lstm1, criterion)
    # print()

    # for ix_epoch in range(1):
    #     # print(f"Epoch {ix_epoch}\n---------")
    #     # train_model(train_loader, lstm1, criterion, optimizer=optimizer)
    #     # test_model(test_loader, lstm1, criterion)
    #     print()

#    setup csvfiles
#

    @app.route('/')
    def index():
        return "hello this is volguard"


    @app.route("/predict")
    def predict():
        # setup functions

        dataset = pd.read_csv("./mysite/hourlyETHUSDCdata.csv", index_col='timestamp', parse_dates=True)

        target_feature="close"
        feature_list=list(dataset.columns.difference([target_feature]))
        target = f"{target_feature}_lead{forecast_amount}"

        dataset[target] = dataset[target_feature].shift(0)
        forecast_start_date= dataset.index[-seq_length]
        test_start_date= "2023-04-01 00:00:00 000"
        dataset_train= dataset.loc[:test_start_date].copy()
        dataset_test= dataset.loc[test_start_date:forecast_start_date].copy()
        dataset_forecast= dataset.loc[forecast_start_date:].copy()
        # set forecast_start_date forecast_start_date= dataset.index[-seq_length]
        # setup dataset_forecast  dataset_forecast= dataset.loc[forecast_start_date:].copy()
        #setup mean and average values

        target_mean1 = dataset_train[target].mean()
        target_stdev1 = dataset_train[target].std()
        for c in dataset_train.columns:
            mean = dataset_train[c].mean()
            stdev = dataset_train[c].std()

            dataset_train[c] = (dataset_train[c] - mean) / stdev
            dataset_test[c] = (dataset_test[c] - mean) / stdev
            dataset_forecast[c] = (dataset_forecast[c] - mean) / stdev


        #setup sequenced datasets and dataloaders
        forecast_dataset =SequenceDataset(dataset_forecast, target=target, features=feature_list, sequence_length= seq_length)
        forecast_loader= DataLoader(forecast_dataset, batch_size=6, shuffle=False)
        forecast_values2= forecast(lstm1, dataset_forecast, forecast_dataset, forecast_loader, 1, target_mean1, target_stdev1)
        forecast_data = forecast_values2
        print(forecast_data)
        # volatility_data = predict_step(test_loader, lstm1).numpy()* target_stdev + target_mean
        print(forecast_data)
        mean = sum(forecast_data)/len(forecast_data)
        variance = sum([((x - mean) ** 2) for x in forecast_data]) / len(forecast_data)
        standard_deviation = variance ** 0.5
        return  str((standard_deviation)*(math.sqrt(24))/(mean))


    @app.route("/predictval")
    def predictval():
        # setup functions

        dataset = pd.read_csv("./mysite/hourlyETHUSDCdata.csv", index_col='timestamp', parse_dates=True)

        target_feature="close"
        feature_list=list(dataset.columns.difference([target_feature]))
        target = f"{target_feature}_lead{forecast_amount}"

        dataset[target] = dataset[target_feature].shift(0)
        forecast_start_date= dataset.index[-seq_length]
        test_start_date= "2023-04-01 00:00:00 000"
        dataset_train= dataset.loc[:test_start_date].copy()
        dataset_test= dataset.loc[test_start_date:forecast_start_date].copy()
        dataset_forecast= dataset.loc[forecast_start_date:].copy()
        # set forecast_start_date forecast_start_date= dataset.index[-seq_length]
        # setup dataset_forecast  dataset_forecast= dataset.loc[forecast_start_date:].copy()
        #setup mean and average values

        target_mean1 = dataset_train[target].mean()
        target_stdev1 = dataset_train[target].std()
        for c in dataset_train.columns:
            mean = dataset_train[c].mean()
            stdev = dataset_train[c].std()

            dataset_train[c] = (dataset_train[c] - mean) / stdev
            dataset_test[c] = (dataset_test[c] - mean) / stdev
            dataset_forecast[c] = (dataset_forecast[c] - mean) / stdev


        #setup sequenced datasets and dataloaders
        forecast_dataset =SequenceDataset(dataset_forecast, target=target, features=feature_list, sequence_length= seq_length)
        forecast_loader= DataLoader(forecast_dataset, batch_size=6, shuffle=False)
        forecast_values2= forecast(lstm1, dataset_forecast, forecast_dataset, forecast_loader, 1, target_mean1, target_stdev1)
        forecast_data = forecast_values2.tolist()
        forecast_readable = ', '.join([str(x) for x in forecast_data])
        return forecast_readable

    @app.route("/model1/predict")
    def predictv2():
         # setup functions

        dataset = pd.read_csv("./mysite/hourlyETHUSDCdata.csv", index_col='timestamp', parse_dates=True)

        target_feature="close"
        feature_list=list(dataset.columns.difference([target_feature]))
        target = f"{target_feature}_lead{forecast_amount}"

        dataset[target] = dataset[target_feature].shift(0)
        forecast_start_date= dataset.index[-seq_length]
        test_start_date= "2023-04-10 00:00:00 000"
        dataset_train= dataset.loc[:test_start_date].copy()
        dataset_test= dataset.loc[test_start_date:forecast_start_date].copy()
        dataset_forecast= dataset.loc[forecast_start_date:].copy()
        # set forecast_start_date forecast_start_date= dataset.index[-seq_length]
        # setup dataset_forecast  dataset_forecast= dataset.loc[forecast_start_date:].copy()
        #setup mean and average values
        close_values =[]
        for i in range(len(dataset_test["close"])-1):
            close_values.append(dataset_test["close"].tolist()[i])
        for k in range(len(dataset_forecast["close"])-1):
            close_values.append(dataset_forecast["close"].tolist()[k])

        target_mean1 = dataset_train[target].mean()
        target_stdev1 = dataset_train[target].std()
        for c in dataset_train.columns:
            mean = dataset_train[c].mean()
            stdev = dataset_train[c].std()

            dataset_train[c] = (dataset_train[c] - mean) / stdev
            dataset_test[c] = (dataset_test[c] - mean) / stdev
            dataset_forecast[c] = (dataset_forecast[c] - mean) / stdev


        #setup sequenced datasets and dataloaders
        forecast_dataset =SequenceDataset(dataset_forecast, target=target, features=feature_list, sequence_length= seq_length)
        forecast_loader= DataLoader(forecast_dataset, batch_size=6, shuffle=False)
        forecast_values2= forecast(lstm1, dataset_forecast, forecast_dataset, forecast_loader, 1, target_mean1, target_stdev1)
        forecast_data = forecast_values2.tolist()

        for fv in forecast_data:
            close_values.append(fv)
        forecast_readable = ', '.join([str(x) for x in close_values])
        datapoint = len(close_values)

        difference = []
        str_output = "this is the starting date: " +test_start_date
        for d in range(len(close_values)-1):
            diff = abs(close_values[d+1] - close_values[d])

            if len(difference) > datapoint :
                difference.pop()
                lower_bound,upper_bound = confidence_interval(np.mean(difference),difference)

                # if diff > upper_bound:
                    #print(dataset["close"].tolist()[d+1],dataset["close"].tolist()[d])
            str_out0= "------------------------------\ln This is :" + str(d)+ " timeframes after :   "
            str_out1 = "Volatility: " + str(diff) +" Average Volatility: " +str(np.mean(difference))
            str_out2 = "------------------------------\ln"
            str_output= str_output+str_out0+ str_out1+str_out2
                    # print("TIME:",dataset["timestamp"].tolist()[d+1])
                    # print("Volatility",diff,"Average Volatility",np.mean(difference))
                    # print("							\n")

            difference.append(diff)
        strout= str(difference[-seq_length+1]/np.mean(difference))
        return strout

    @app.route("/model2/predict")
    def predictv3():
         # setup functions
        seq_length = 3
        dataset = pd.read_csv("./mysite/hourlyETHUSDCdata.csv", index_col='timestamp', parse_dates=True)

        target_feature="close"
        feature_list=list(dataset.columns.difference([target_feature]))
        target = f"{target_feature}_lead{forecast_amount}"

        dataset[target] = dataset[target_feature].shift(0)
        forecast_start_date= dataset.index[-seq_length]
        test_start_date= "2023-04-10 00:00:00 000"
        dataset_train= dataset.loc[:test_start_date].copy()
        dataset_test= dataset.loc[test_start_date:forecast_start_date].copy()
        dataset_forecast= dataset.loc[forecast_start_date:].copy()
        # set forecast_start_date forecast_start_date= dataset.index[-seq_length]
        # setup dataset_forecast  dataset_forecast= dataset.loc[forecast_start_date:].copy()
        #setup mean and average values
        close_values =[]
        for i in range(len(dataset_test["close"])-1):
            close_values.append(dataset_test["close"].tolist()[i])
        for k in range(len(dataset_forecast["close"])-1):
            close_values.append(dataset_forecast["close"].tolist()[k])

        target_mean1 = dataset_train[target].mean()
        target_stdev1 = dataset_train[target].std()
        for c in dataset_train.columns:
            mean = dataset_train[c].mean()
            stdev = dataset_train[c].std()

            dataset_train[c] = (dataset_train[c] - mean) / stdev
            dataset_test[c] = (dataset_test[c] - mean) / stdev
            dataset_forecast[c] = (dataset_forecast[c] - mean) / stdev


        #setup sequenced datasets and dataloaders
        forecast_dataset =SequenceDataset(dataset_forecast, target=target, features=feature_list, sequence_length= seq_length)
        forecast_loader= DataLoader(forecast_dataset, batch_size=6, shuffle=False)
        forecast_values2= forecast(lstm1, dataset_forecast, forecast_dataset, forecast_loader, 1, target_mean1, target_stdev1)
        forecast_data = forecast_values2.tolist()

        for fv in forecast_data:
            close_values.append(fv)
        forecast_readable = ', '.join([str(x) for x in close_values])
        datapoint = len(close_values)

        difference = []
        str_output = "this is the starting date: " +test_start_date
        for d in range(len(close_values)-1):
            diff = abs(close_values[d+1] - close_values[d])

            if len(difference) > datapoint :
                difference.pop()
                lower_bound,upper_bound = confidence_interval(np.mean(difference),difference)

                # if diff > upper_bound:
                    #print(dataset["close"].tolist()[d+1],dataset["close"].tolist()[d])
            str_out0= "------------------------------\ln This is :" + str(d)+ " timeframes after :   "
            str_out1 = "Volatility: " + str(diff) +" Average Volatility: " +str(np.mean(difference))
            str_out2 = "------------------------------\ln"
            str_output= str_output+str_out0+ str_out1+str_out2
                    # print("TIME:",dataset["timestamp"].tolist()[d+1])
                    # print("Volatility",diff,"Average Volatility",np.mean(difference))
                    # print("							\n")

            difference.append(diff)
        strout= str(difference[-seq_length+1]/np.mean(difference))
        return strout


    @app.route("/model3/predict")
    def predictv4():
         # setup functions
        seq_length = 6
        dataset = pd.read_csv("./mysite/hourlyETHUSDCdata.csv", index_col='timestamp', parse_dates=True)

        target_feature="close"
        feature_list=list(dataset.columns.difference([target_feature]))
        target = f"{target_feature}_lead{forecast_amount}"

        dataset[target] = dataset[target_feature].shift(0)
        forecast_start_date= dataset.index[-seq_length]
        test_start_date= "2023-04-10 00:00:00 000"
        dataset_train= dataset.loc[:test_start_date].copy()
        dataset_test= dataset.loc[test_start_date:forecast_start_date].copy()
        dataset_forecast= dataset.loc[forecast_start_date:].copy()
        # set forecast_start_date forecast_start_date= dataset.index[-seq_length]
        # setup dataset_forecast  dataset_forecast= dataset.loc[forecast_start_date:].copy()
        #setup mean and average values
        close_values =[]
        for i in range(len(dataset_test["close"])-1):
            close_values.append(dataset_test["close"].tolist()[i])
        for k in range(len(dataset_forecast["close"])-1):
            close_values.append(dataset_forecast["close"].tolist()[k])

        target_mean1 = dataset_train[target].mean()
        target_stdev1 = dataset_train[target].std()
        for c in dataset_train.columns:
            mean = dataset_train[c].mean()
            stdev = dataset_train[c].std()

            dataset_train[c] = (dataset_train[c] - mean) / stdev
            dataset_test[c] = (dataset_test[c] - mean) / stdev
            dataset_forecast[c] = (dataset_forecast[c] - mean) / stdev


        #setup sequenced datasets and dataloaders
        forecast_dataset =SequenceDataset(dataset_forecast, target=target, features=feature_list, sequence_length= seq_length)
        forecast_loader= DataLoader(forecast_dataset, batch_size=6, shuffle=False)
        forecast_values2= forecast(lstm1, dataset_forecast, forecast_dataset, forecast_loader, 1, target_mean1, target_stdev1)
        forecast_data = forecast_values2.tolist()

        for fv in forecast_data:
            close_values.append(fv)
        forecast_readable = ', '.join([str(x) for x in close_values])
        datapoint = len(close_values)

        difference = []
        str_output = "this is the starting date: " +test_start_date
        for d in range(len(close_values)-1):
            diff = abs(close_values[d+1] - close_values[d])

            if len(difference) > datapoint :
                difference.pop()
                lower_bound,upper_bound = confidence_interval(np.mean(difference),difference)

                # if diff > upper_bound:
                    #print(dataset["close"].tolist()[d+1],dataset["close"].tolist()[d])
            str_out0= "------------------------------\ln This is :" + str(d)+ " timeframes after :   "
            str_out1 = "Volatility: " + str(diff) +" Average Volatility: " +str(np.mean(difference))
            str_out2 = "------------------------------\ln"
            str_output= str_output+str_out0+ str_out1+str_out2
                    # print("TIME:",dataset["timestamp"].tolist()[d+1])
                    # print("Volatility",diff,"Average Volatility",np.mean(difference))
                    # print("							\n")

            difference.append(diff)
        strout= str(difference[-seq_length+1]/np.mean(difference))
        return strout

    @app.route("/model4/predict")
    def predictv5():
         # setup functions
        seq_length = 24
        dataset = pd.read_csv("./mysite/hourlyETHUSDCdata.csv", index_col='timestamp', parse_dates=True)

        target_feature="close"
        feature_list=list(dataset.columns.difference([target_feature]))
        target = f"{target_feature}_lead{forecast_amount}"

        dataset[target] = dataset[target_feature].shift(0)
        forecast_start_date= dataset.index[-seq_length]
        test_start_date= "2023-04-10 00:00:00 000"
        dataset_train= dataset.loc[:test_start_date].copy()
        dataset_test= dataset.loc[test_start_date:forecast_start_date].copy()
        dataset_forecast= dataset.loc[forecast_start_date:].copy()
        # set forecast_start_date forecast_start_date= dataset.index[-seq_length]
        # setup dataset_forecast  dataset_forecast= dataset.loc[forecast_start_date:].copy()
        #setup mean and average values
        close_values =[]
        for i in range(len(dataset_test["close"])-1):
            close_values.append(dataset_test["close"].tolist()[i])
        for k in range(len(dataset_forecast["close"])-1):
            close_values.append(dataset_forecast["close"].tolist()[k])

        target_mean1 = dataset_train[target].mean()
        target_stdev1 = dataset_train[target].std()
        for c in dataset_train.columns:
            mean = dataset_train[c].mean()
            stdev = dataset_train[c].std()

            dataset_train[c] = (dataset_train[c] - mean) / stdev
            dataset_test[c] = (dataset_test[c] - mean) / stdev
            dataset_forecast[c] = (dataset_forecast[c] - mean) / stdev


        #setup sequenced datasets and dataloaders
        forecast_dataset =SequenceDataset(dataset_forecast, target=target, features=feature_list, sequence_length= seq_length)
        forecast_loader= DataLoader(forecast_dataset, batch_size=6, shuffle=False)
        forecast_values2= forecast(lstm1, dataset_forecast, forecast_dataset, forecast_loader, 1, target_mean1, target_stdev1)
        forecast_data = forecast_values2.tolist()
        print(forecast_data)
        print(len(forecast_data))
        for fv in forecast_data:
            close_values.append(fv)
        print(len(close_values))
        print("that was len close data")
        forecast_readable = ', '.join([str(x) for x in close_values])
        datapoint = len(close_values)

        difference = []
        str_output = "this is the starting date: " +test_start_date
        for d in range(len(close_values)-1):
            diff = abs(close_values[d+1] - close_values[d])

            if len(difference) > datapoint :
                difference.pop()
                lower_bound,upper_bound = confidence_interval(np.mean(difference),difference)

                # if diff > upper_bound:
                    #print(dataset["close"].tolist()[d+1],dataset["close"].tolist()[d])
            str_out0= "------------------------------\ln This is :" + str(d)+ " timeframes after :   "
            str_out1 = "Volatility: " + str(diff) +" Average Volatility: " +str(np.mean(difference))
            str_out2 = "------------------------------\ln"
            str_output= str_output+str_out0+ str_out1+str_out2
                    # print("TIME:",dataset["timestamp"].tolist()[d+1])
                    # print("Volatility",diff,"Average Volatility",np.mean(difference))
                    # print("							\n")

            difference.append(diff)
        strout= str(difference[(-seq_length)+1]/np.mean(difference))
        return strout

    @app.route("/data")
    def data():
        url = "https://web3api.io/api/v2/market/defi/ohlcv/0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640/historical/?exchange=uniswapv3&timeInterval=hours&format=csv&fields=timestamp%2Copen%2Chigh%2Clow%2Cvolume%2Cclose&timeFormat=human_readable"

        headers = {
            "accept": "application/json",
            "x-api-key": "UAK7ed69235426c360be22bfc2bde1809b6"
        }

        response = requests.get(url, headers=headers)
        f = open('./mysite/hourlyETHUSDCdata.csv', "w")
        f.write(response.text)
        f.close()
        #getting data from amberdata files
        #Data provided by Amberdata.io
        dataset = pd.read_csv('./mysite/hourlyETHUSDCdata.csv', index_col = 'timestamp', parse_dates=True)
        print("Dataset loaded")
        return("loaded new dataset")

    @app.route("/deploy", methods=['GET', 'POST'])
    def deploy_contract():
        print("-------------------------------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------------------------------")
        # data = request.get_json()
        print(data)
        INFURA_URL = "https://goerli.infura.io/v3/8e16cb87ad3f4ee19a3c24a3582daf8c"
        w3 = Web3(Web3.HTTPProvider(INFURA_URL))
        assert w3.is_connected()
        # PRIVATE_KEY = data.get('privatekey')
        # ACCOUNT_ADDRESS = data.get('accountaddr')
        # print(PRIVATE_KEY)
        # print(ACCOUNT_ADDRESS)
        PRIVATE_KEY = '761f664e6eeed8c353ec52c2ecef75905aafa8e5421a57282f7c3da5112fb95d'
        ACCOUNT_ADDRESS = '0x2088c6c71c7e2609a98bFaf89AC5Ed618518Da74'
        selected_model_id = 1
        print(selected_model_id)
        if (selected_model_id== 1):
            compiled_sol = compile_files("./mysite/Oracle/contract/simple-oracle.sol", output_values=["abi", "bin-runtime", "bin"],)
            print(list(compiled_sol.keys()))
            contract_interface = compiled_sol.get("./mysite/Oracle/contract/simple-oracle.sol:SimpleOracle")
        elif (selected_model_id== 2):
            compiled_sol = compile_files("./mysite/Oracle2/contract/oracle.sol", output_values=["abi", "bin-runtime", "bin"],)
            print(list(compiled_sol.keys()))
            contract_interface = compiled_sol.get("./mysite/Oracle2/contract/oracle.sol:CentralizedOracle")
        elif (selected_model_id== 3):
            compiled_sol = compile_files("./mysite/Oracle3/contract/oracle.sol", output_values=["abi", "bin-runtime", "bin"],)
            print(list(compiled_sol.keys()))
            contract_interface = compiled_sol.get("./mysite/Oracle3/contract/oracle.sol:CentralizedOracle")
        elif (selected_model_id== 4):
            compiled_sol = compile_files("./mysite/Oracle4/contract/oracle.sol", output_values=["abi", "bin-runtime", "bin"],)
            print(list(compiled_sol.keys()))
            contract_interface = compiled_sol.get("./mysite/Oracle4/contract/oracle.sol:CentralizedOracle")
        contract_bin = contract_interface.get('bin')
        model_api_endpoints = {
            1: "https://johnbartleydev.pythonanywhere.com/model1/predict",
            2: "https://johnbartleydev.pythonanywhere.com/model2/predict",
            3: "https://johnbartleydev.pythonanywhere.com/model3/predict",
            4: "https://johnbartleydev.pythonanywhere.com/model4/predict"
        }

        model_api_endpoint = model_api_endpoints[selected_model_id]

        # Deploy the contract
        nonce = w3.eth.get_transaction_count(ACCOUNT_ADDRESS)
        gas_estimate = w3.eth.estimate_gas({"from": ACCOUNT_ADDRESS, "data": contract_bin})
        transaction = {
          'from': ACCOUNT_ADDRESS,
          'data': contract_bin,
          'gas': gas_estimate,
          'gasPrice': w3.eth.gas_price,
          'nonce': nonce,
          'chainId': 5 # Goerli testnet chain ID
        }
        signed_txn = w3.eth.account.sign_transaction(transaction, PRIVATE_KEY)
        txn_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        # Wait for the transaction receipt
        transaction_receipt = w3.eth.wait_for_transaction_receipt(txn_hash)
        contract_address = transaction_receipt['contractAddress']
        print(f"Contract deployed at address: {contract_address}")
        return (contract_address)

    @app.route("/deploy2", methods=['GET', 'POST'])
    def deploy_contract2():
        print("-------------------------------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------------------------------")
        # data = request.get_json()
        print(data)
        INFURA_URL = "https://goerli.infura.io/v3/8e16cb87ad3f4ee19a3c24a3582daf8c"
        w3 = Web3(Web3.HTTPProvider(INFURA_URL))
        assert w3.is_connected()
        # PRIVATE_KEY = data.get('privatekey')
        # ACCOUNT_ADDRESS = data.get('accountaddr')
        # print(PRIVATE_KEY)
        # print(ACCOUNT_ADDRESS)
        PRIVATE_KEY = '761f664e6eeed8c353ec52c2ecef75905aafa8e5421a57282f7c3da5112fb95d'
        ACCOUNT_ADDRESS = '0x2088c6c71c7e2609a98bFaf89AC5Ed618518Da74'
        selected_model_id = 2
        print(selected_model_id)
        if (selected_model_id== 1):
            compiled_sol = compile_files("./mysite/Oracle/contract/simple-oracle.sol", output_values=["abi", "bin-runtime", "bin"],)
            print(list(compiled_sol.keys()))
            contract_interface = compiled_sol.get("./mysite/Oracle/contract/simple-oracle.sol:SimpleOracle")
        elif (selected_model_id== 2):
            compiled_sol = compile_files("./mysite/Oracle/contract/oracle2.sol", output_values=["abi", "bin-runtime", "bin"],)
            print(list(compiled_sol.keys()))
            contract_interface = compiled_sol.get("./mysite/Oracle/contract/oracle2.sol:CentralizedOracle")
        elif (selected_model_id== 3):
            compiled_sol = compile_files("./mysite/Oracle3/contract/oracle.sol", output_values=["abi", "bin-runtime", "bin"],)
            print(list(compiled_sol.keys()))
            contract_interface = compiled_sol.get("./mysite/Oracle3/contract/oracle.sol:CentralizedOracle")
        elif (selected_model_id== 4):
            compiled_sol = compile_files("./mysite/Oracle4/contract/oracle.sol", output_values=["abi", "bin-runtime", "bin"],)
            print(list(compiled_sol.keys()))
            contract_interface = compiled_sol.get("./mysite/Oracle4/contract/oracle.sol:CentralizedOracle")
        contract_bin = contract_interface.get('bin')
        model_api_endpoints = {
            1: "https://johnbartleydev.pythonanywhere.com/model1/predict",
            2: "https://johnbartleydev.pythonanywhere.com/model2/predict",
            3: "https://johnbartleydev.pythonanywhere.com/model3/predict",
            4: "https://johnbartleydev.pythonanywhere.com/model4/predict"
        }

        model_api_endpoint = model_api_endpoints[selected_model_id]

        # Deploy the contract
        nonce = w3.eth.get_transaction_count(ACCOUNT_ADDRESS)
        gas_estimate = w3.eth.estimate_gas({"from": ACCOUNT_ADDRESS, "data": contract_bin})
        transaction = {
          'from': ACCOUNT_ADDRESS,
          'data': contract_bin,
          'gas': gas_estimate,
          'gasPrice': w3.eth.gas_price,
          'nonce': nonce,
          'chainId': 5 # Goerli testnet chain ID
        }
        signed_txn = w3.eth.account.sign_transaction(transaction, PRIVATE_KEY)
        txn_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        # Wait for the transaction receipt
        transaction_receipt = w3.eth.wait_for_transaction_receipt(txn_hash)
        contract_address = transaction_receipt['contractAddress']
        print(f"Contract deployed at address: {contract_address}")
        return (contract_address)

    @app.route("/deploy3", methods=['GET', 'POST'])
    def deploy_contract3():
        print("-------------------------------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------------------------------")
        # data = request.get_json()
        print(data)
        INFURA_URL = "https://goerli.infura.io/v3/8e16cb87ad3f4ee19a3c24a3582daf8c"
        w3 = Web3(Web3.HTTPProvider(INFURA_URL))
        assert w3.is_connected()
        # PRIVATE_KEY = data.get('privatekey')
        # ACCOUNT_ADDRESS = data.get('accountaddr')
        # print(PRIVATE_KEY)
        # print(ACCOUNT_ADDRESS)
        PRIVATE_KEY = '761f664e6eeed8c353ec52c2ecef75905aafa8e5421a57282f7c3da5112fb95d'
        ACCOUNT_ADDRESS = '0x2088c6c71c7e2609a98bFaf89AC5Ed618518Da74'
        selected_model_id = 3
        print(selected_model_id)
        if (selected_model_id== 1):
            compiled_sol = compile_files("./mysite/Oracle/contract/oracle.sol", output_values=["abi", "bin-runtime", "bin"],)
            print(list(compiled_sol.keys()))
            contract_interface = compiled_sol.get("./mysite/Oracle/contract/oracle.sol:CentralizedOracle")
        elif (selected_model_id== 2):
            compiled_sol = compile_files("./mysite/Oracle2/contract/oracle2.sol", output_values=["abi", "bin-runtime", "bin"],)
            print(list(compiled_sol.keys()))
            contract_interface = compiled_sol.get("./mysite/Oracle2/contract/oracle2.sol:CentralizedOracle")
        elif (selected_model_id== 3):
            compiled_sol = compile_files("./mysite/Oracle/contract/oracle3.sol", output_values=["abi", "bin-runtime", "bin"],)
            print(list(compiled_sol.keys()))
            contract_interface = compiled_sol.get("./mysite/Oracle/contract/oracle3.sol:CentralizedOracle")
        elif (selected_model_id== 4):
            compiled_sol = compile_files("./mysite/Oracle4/contract/oracle.sol", output_values=["abi", "bin-runtime", "bin"],)
            print(list(compiled_sol.keys()))
            contract_interface = compiled_sol.get("./mysite/Oracle4/contract/oracle.sol:simpleOracle")
        contract_bin = contract_interface.get('bin')
        model_api_endpoints = {
            1: "https://johnbartleydev.pythonanywhere.com/model1/predict",
            2: "https://johnbartleydev.pythonanywhere.com/model2/predict",
            3: "https://johnbartleydev.pythonanywhere.com/model3/predict",
            4: "https://johnbartleydev.pythonanywhere.com/model4/predict"
        }

        model_api_endpoint = model_api_endpoints[selected_model_id]

        # Deploy the contract
        nonce = w3.eth.get_transaction_count(ACCOUNT_ADDRESS)
        gas_estimate = w3.eth.estimate_gas({"from": ACCOUNT_ADDRESS, "data": contract_bin})
        transaction = {
          'from': ACCOUNT_ADDRESS,
          'data': contract_bin,
          'gas': gas_estimate,
          'gasPrice': w3.eth.gas_price,
          'nonce': nonce,
          'chainId': 5 # Goerli testnet chain ID
        }
        signed_txn = w3.eth.account.sign_transaction(transaction, PRIVATE_KEY)
        txn_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        # Wait for the transaction receipt
        transaction_receipt = w3.eth.wait_for_transaction_receipt(txn_hash)
        contract_address = transaction_receipt['contractAddress']
        print(f"Contract deployed at address: {contract_address}")
        return (contract_address)

    @app.route("/deploy4", methods=['GET', 'POST'])
    def deploy_contract4():
        print("-------------------------------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------------------------------")
        # data = request.get_json()
        print(data)
        INFURA_URL = "https://goerli.infura.io/v3/8e16cb87ad3f4ee19a3c24a3582daf8c"
        w3 = Web3(Web3.HTTPProvider(INFURA_URL))
        assert w3.is_connected()
        # PRIVATE_KEY = data.get('privatekey')
        # ACCOUNT_ADDRESS = data.get('accountaddr')
        # print(PRIVATE_KEY)
        # print(ACCOUNT_ADDRESS)
        PRIVATE_KEY = '761f664e6eeed8c353ec52c2ecef75905aafa8e5421a57282f7c3da5112fb95d'
        ACCOUNT_ADDRESS = '0x2088c6c71c7e2609a98bFaf89AC5Ed618518Da74'
        selected_model_id = 4
        print(selected_model_id)
        if (selected_model_id== 1):
            compiled_sol = compile_files("./mysite/Oracle/contract/oracle.sol", output_values=["abi", "bin-runtime", "bin"],)
            print(list(compiled_sol.keys()))
            contract_interface = compiled_sol.get("./mysite/Oracle/contract/oracle.sol:CentralizedOracle")
        elif (selected_model_id== 2):
            compiled_sol = compile_files("./mysite/Oracle2/contract/oracle.sol", output_values=["abi", "bin-runtime", "bin"],)
            print(list(compiled_sol.keys()))
            contract_interface = compiled_sol.get("./mysite/Oracle2/contract/oracle.sol:CentralizedOracle")
        elif (selected_model_id== 3):
            compiled_sol = compile_files("./mysite/Oracle3/contract/oracle.sol", output_values=["abi", "bin-runtime", "bin"],)
            print(list(compiled_sol.keys()))
            contract_interface = compiled_sol.get("./mysite/Oracle3/contract/oracle.sol:CentralizedOracle")
        elif (selected_model_id== 4):
            compiled_sol = compile_files("./mysite/Oracle/contract/oracle4.sol", output_values=["abi", "bin-runtime", "bin"],)
            print(list(compiled_sol.keys()))
            contract_interface = compiled_sol.get("./mysite/Oracle/contract/oracle4.sol:CentralizedOracle")
        contract_bin = contract_interface.get('bin')
        model_api_endpoints = {
            1: "https://johnbartleydev.pythonanywhere.com/model1/predict",
            2: "https://johnbartleydev.pythonanywhere.com/model2/predict",
            3: "https://johnbartleydev.pythonanywhere.com/model3/predict",
            4: "https://johnbartleydev.pythonanywhere.com/model4/predict"
        }

        model_api_endpoint = model_api_endpoints[selected_model_id]

        # Deploy the contract
        nonce = w3.eth.get_transaction_count(ACCOUNT_ADDRESS)
        gas_estimate = w3.eth.estimate_gas({"from": ACCOUNT_ADDRESS, "data": contract_bin})
        transaction = {
          'from': ACCOUNT_ADDRESS,
          'data': contract_bin,
          'gas': gas_estimate,
          'gasPrice': w3.eth.gas_price,
          'nonce': nonce,
          'chainId': 5 # Goerli testnet chain ID
        }
        signed_txn = w3.eth.account.sign_transaction(transaction, PRIVATE_KEY)
        txn_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        # Wait for the transaction receipt
        transaction_receipt = w3.eth.wait_for_transaction_receipt(txn_hash)
        contract_address = transaction_receipt['contractAddress']
        print(f"Contract deployed at address: {contract_address}")
        return (contract_address)

    @app.route("/deployupdater", methods=['GET', 'POST'])
    def contract_update():
        print("-------------------------------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------------------------------")
        contract_abi = [...]
        contract_address = '...'

        # Replace with your Ethereum private key and Infura API key
        # =========================================================

        private_key = 'YOUR_PRIVATE_KEY'
        infura_api_key = 'YOUR_INFURA_API_KEY'

        # Connect to Ethereum
        # ===================

        w3 = Web3(HTTPProvider(f"https://mainnet.infura.io/v3/{infura_api_key}"))
        contract = w3.eth.contract(address=Web3.toChecksumAddress(contract_address), abi=contract_abi)

        # Replace with your API URL
        # =========================

        api_url = 'https://johnbartleydev.pythonanywhere.com/model1/predict'

        # Your Ethereum address
        # =====================

        my_address = '...'

        # Function to fetch data from API and update stored number in the contract
        # ========================================================================

        def update_stored_number():
            try:
                response = requests.get(api_url)
                new_number = int(float(response.text))

                gas_price = w3.eth.gas_price
                gas_estimate = contract.functions.updateStoredNumber(new_number).estimate_gas({'from': my_address})

                transaction = contract.functions.updateStoredNumber(new_number).build_transaction({
                  'from': my_address,
                  'gas': gas_estimate,
                  'gasPrice': gas_price,
                  'nonce': w3.eth.get_transaction_count(my_address),
                })

                signed_tx = w3.eth.account.sign_transaction(transaction, private_key)
                transaction_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                print(f"Transaction hash: {transaction_hash.hex()}")

                receipt = w3.eth.wait_for_transaction_receipt(transaction_hash)
                if receipt['status'] == 0:
                  print("Transaction failed.")

            except ValueError as ve:
                print(f"Error updating stored number: {ve}")
            except requests.exceptions.RequestException as re:
                print(f"Error fetching data from API: {re}")

        Timer(6 * 60 * 60, update_stored_number).start()






        # model_api_endpoints = {
        #     1: "https://johnbartleydev.pythonanywhere.com/model1/predict",
        #     2: "https://johnbartleydev.pythonanywhere.com/model2/predict",
        #     3: "https://johnbartleydev.pythonanywhere.com/model3/predict",
        #     4: "https://johnbartleydev.pythonanywhere.com/model4/predict"
        # }

        # model_api_endpoint = model_api_endpoints[selected_model_id]

        # # Deploy the contract
        # nonce = w3.eth.get_transaction_count(ACCOUNT_ADDRESS)
        # gas_estimate = w3.eth.estimate_gas({"from": ACCOUNT_ADDRESS, "data": contract_bin})
        # transaction = {
        #   'from': ACCOUNT_ADDRESS,
        #   'data': contract_bin,
        #   'gas': gas_estimate,
        #   'gasPrice': w3.eth.gas_price,
        #   'nonce': nonce,
        #   'chainId': 5 # Goerli testnet chain ID
        # }
        # signed_txn = w3.eth.account.sign_transaction(transaction, PRIVATE_KEY)
        # txn_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        # # Wait for the transaction receipt
        # transaction_receipt = w3.eth.wait_for_transaction_receipt(txn_hash)
        # contract_address = transaction_receipt['contractAddress']
        # print(f"Contract deployed at address: {contract_address}")
        # return (contract_address)
        return(update_stored_number())
    return app



app =create_app()


