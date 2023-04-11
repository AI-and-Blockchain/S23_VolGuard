from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.autograd import Variable 
from math import sqrt
from flask import Flask, jsonify
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import os
import parquet

print("begging of file test")
#getting data from amberdata files
#Data provided by Amberdata.io
dataset = pd.read_csv("../../historical_files/hourlyETHUSDCdata.csv", index_col = 'timestamp', parse_dates=True)
print("Dataset loaded")
#model input values
num_epochs = 60000 #1000 epochs
learning_rate = 0.0005 #0.001 lr
input_size = 4 #number of features
hidden_size = 2 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers
num_classes = 1 #number of output classes
avg_volatility =6.77
 
#Basic data labeling
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4:5] 

#basic data proccessing
mm = MinMaxScaler()
ss = StandardScaler()
X_ss = ss.fit_transform(X)
y_mm = mm.fit_transform(y)

#Creation of training and test sets
splitsize = 600
X_train = X_ss[:splitsize, :]
X_test = X_ss[splitsize:, :]
y_train = y_mm[:splitsize, :]
y_test = y_mm[splitsize:, :] 

#turn the numpy arrays into tensors
X_train_tensors = Variable(torch.Tensor(X_train))
X_test_tensors = Variable(torch.Tensor(X_test))

y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test)) 
X_tensor_train = torch.reshape(X_train_tensors,   (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
X_tensor_test = torch.reshape(X_test_tensors,  (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])) 

##LSTM model class
class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out
    
# if model is already defined load that model instead of creating a new one
if os.path.isfile('./saved_models/hourlylstm.pt'):
    print("model was found")
    lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_tensor_train.shape[1])
    model_state_dict = torch.load('./saved_models/hourlylstm.pt') # loading the dictionary object
    lstm1.load_state_dict(model_state_dict) # load_state_dict() function takes a dictionary object, NOT a path to a saved object
    lstm1.eval() # since we need to use the model for inference
else:
    print("no saved model found")
lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_tensor_train.shape[1]) #our lstm class 

# Defining loss function, and optimizing parameters
criterion = torch.nn.MSELoss()    
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate) 
loss_list=[]

# Trains model for set number of epochs, and then returns the last gotten RSE loss value
print("reached training")
def train():
    for epoch in range(num_epochs):
        outputs = lstm1.forward(X_tensor_train) #forward pass
        optimizer.zero_grad() #caluclate the gradient, manually setting to 0
    
    # obtain the loss function
        loss = criterion(outputs, y_train_tensors)
        loss_list.append(loss)
        loss.backward() #calculates the loss of the loss function
    
        optimizer.step() #improve from loss, i.e backprop
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
    return loss_list[-1]

# shapes dataset into usable values, then predicts eth-usd values for given timeframe, and returns the predicted volatility of the first 24 points of prediction aka the next day of prediciton
def predict_plot():
    df_X_ss = ss.transform(dataset.iloc[:, :-1]) #old transformers
    df_y_mm = mm.transform(dataset.iloc[:, -1:]) #old transformers

    df_X_ss = Variable(torch.Tensor(df_X_ss)) #converting to Tensors
    df_y_mm = Variable(torch.Tensor(df_y_mm))
    #reshaping the dataset
    df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1])) 
    train_predict = lstm1(df_X_ss)#forward pass
    data_predict = train_predict.data.numpy() #numpy conversion
    dataY_plot = df_y_mm.data.numpy()

    data_predict = mm.inverse_transform(data_predict) #reverse transformation
    dataY_plot = mm.inverse_transform(dataY_plot)
    plt.figure(figsize=(10,6)) #plotting
    plt.axvline(x=splitsize, c='r', linestyle='--') #size of the training set

    plt.plot(dataY_plot, label='Actuall Data') #actual plot
    plt.plot(data_predict, label='Predicted Data') #predicted plot
    plt.title('Time-Series Prediction')
    plt.legend()
    plt.show() 
    volatility_data = data_predict[:splitsize]
    mean = sum(volatility_data)/len(volatility_data)
    variance = sum([((x - mean) ** 2) for x in volatility_data]) / len(volatility_data)
    standard_deviation = variance ** 0.5
    volatility_prediction = standard_deviation/dataY_plot[600]*100
    return volatility_prediction

# Plots loss values given during training
def mse_plot():
    plt.plot(loss_list)
    plt.ylabel('RMSE loss value')
    plt.show()
    
    
    # sets up flask wep application
def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    
    # Call to get next day predicted volatility from api
    @app.route('/predict', methods=['POST'])
    def predict():
    if request.method == 'POST':
        # we will get the file from the request
        file = request.files['file']
        # convert that to bytes
        dataset = pd.read_csv(file, index_col = 'timestamp', parse_dates=True)
        volatility_prediction = predict_plot()
        return jsonify({'predicted_daily_volatility': volatility_prediction})
    
    # call to train the model from rest API
    @app.route('/train', methods=['POST'])
    def predict():
    if request.method == 'POST':
        # we will get the file from the request
        file = request.files['file']
        # convert that to bytes
        dataset = pd.read_csv(file, index_col = 'timestamp', parse_dates=True)
        lossvalue = train()
        return jsonify({'final loss': lossvalue})


        
# train()
# print("Completed training")
# predict_plot()
# mse_plot()
# print("completed prediciton")

PATH = f'saved_models/hourlylstm.pt'
torch.save(lstm1.state_dict(), PATH)
print("model state saved")