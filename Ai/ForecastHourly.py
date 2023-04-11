from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

print("begging of file test")
#getting data from amberdata files
#Data provided by Amberdata.io
dataset = pd.read_csv("../../historical_files/hourlyETHUSDCdata.csv", index_col = 'timestamp', parse_dates=True)
print("Dataset loaded")
#model input values
num_epochs = 1000 #1000 epochs
learning_rate = 0.0005 #0.001 lr
input_size = 4 #number of features
hidden_size = 2 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers
num_classes = 1 #number of output classes
 
 
target_feature="close"
feature_list=list(dataset.columns.difference([target_feature]))

forecast_amount= 100
seq_length= 24
torch.manual_seed(101)
target = f"{target_feature}_lead{forecast_amount}"

dataset[target] = dataset[target_feature].shift(-forecast_amount)



#Basic data labeling
test_start_date= "2023-03-26 00:00:00 000"
dataset_train= dataset.loc[:test_start_date].copy()
dataset_test= dataset.loc[test_start_date:].copy()
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
    
# if os.path.isfile('./saved_models/hourlylstm.pt'):
#     print("model was found")
#     lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_tensor_train.shape[1])
#     model_state_dict = torch.load('./saved_models/hourlylstm.pt') # loading the dictionary object
#     lstm1.load_state_dict(model_state_dict) # load_state_dict() function takes a dictionary object, NOT a path to a saved object
#     lstm1.eval() # since we need to use the model for inference
# else:
#     print("no saved model found")
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
test_model(test_loader, lstm1, criterion)
print()

for ix_epoch in range(100):
    print(f"Epoch {ix_epoch}\n---------")
    train_model(train_loader, lstm1, criterion, optimizer=optimizer)
    test_model(test_loader, lstm1, criterion)
    print()
    
    
    
# def train():
#     for epoch in range(num_epochs):
#         outputs = lstm1.forward(X_tensor_train) #forward pass
#         optimizer.zero_grad() #caluclate the gradient, manually setting to 0
    
#     # obtain the loss function
#         loss = criterion(outputs, y_train_tensors)
    
#         loss.backward() #calculates the loss of the loss function
    
#         optimizer.step() #improve from loss, i.e backprop
#         if epoch % 100 == 0:
#             print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 



# def predict_plot():
#     df_X_ss = ss.transform(dataset.iloc[:, :-1]) #old transformers
#     df_y_mm = mm.transform(dataset.iloc[:, -1:]) #old transformers

#     df_X_ss = Variable(torch.Tensor(df_X_ss)) #converting to Tensors
#     df_y_mm = Variable(torch.Tensor(df_y_mm))
#     #reshaping the dataset
#     df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1])) 
#     train_predict = lstm1(df_X_ss)#forward pass
#     data_predict = train_predict.data.numpy() #numpy conversion
#     dataY_plot = df_y_mm.data.numpy()

#     data_predict = mm.inverse_transform(data_predict) #reverse transformation
#     dataY_plot = mm.inverse_transform(dataY_plot)
#     plt.figure(figsize=(10,6)) #plotting
#     plt.axvline(x=splitsize, c='r', linestyle='--') #size of the training set

#     plt.plot(dataY_plot, label='Actuall Data') #actual plot
#     plt.plot(data_predict, label='Predicted Data') #predicted plot
#     plt.title('Time-Series Prediction')
#     plt.legend()
#     plt.show() 
def predict(data_loader, model):

    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)
    
    return output


train_eval_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)

ystar_col = "close forecast"
dataset_train[ystar_col] = predict(train_eval_loader, lstm1).numpy()
dataset_test[ystar_col] = predict(test_loader, lstm1).numpy()

df_out = pd.concat((dataset_train, dataset_test))[[target, ystar_col]]

for c in df_out.columns:
    df_out[c] = df_out[c] * target_stdev + target_mean

print(df_out)

plot_template = dict(
    layout=go.Layout({
        "font_size": 18,
        "xaxis_title_font_size": 24,
        "yaxis_title_font_size": 24})
)

fig = px.line(df_out, labels=dict(created_at="Date", value="dolar-to-eth"))
fig.add_vline(x=test_start_date, line_width=4, line_dash="dash")
fig.add_annotation(xref="paper", x=0.75, yref="paper", y=0.8, text="Test set start", showarrow=False)
fig.update_layout(
    template=plot_template, legend=dict(orientation='h', y=1.02, title_text="")
)
fig.show()

# train()
# print("Completed training")
# predict_plot()
print("completed prediciton")

# PATH = f'saved_models/hourlylstm.pt'
# torch.save(lstm1.state_dict(), PATH)
# print("model state saved")