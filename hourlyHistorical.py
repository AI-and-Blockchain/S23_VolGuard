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
from datetime import datetime, timedelta, date
import pandas as pd
import time
import requests
import csv
import math



def get_new_data():
    

    url = "https://web3api.io/api/v2/market/defi/ohlcv/0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640/historical/?exchange=uniswapv3&startDate=2023-03-15&endDate=2023-04-13&timeInterval=hours&format=csv&fields=timestamp%2Copen%2Chigh%2Clow%2Cvolume%2Cclose&timeFormat=human_readable"

    headers = {
        "accept": "application/json",
        "x-api-key": ""
    }

    response = requests.get(url, headers=headers)
    f = open('./historical_files/hourlyETHUSDCdata.csv', "w")
    f.write(response.text)
    f.close()
    #getting data from amberdata files
    #Data provided by Amberdata.io
    # ?dataset = pd.read_csv('./historical_files/hourlyETHUSDCdata.csv', index_col = 'timestamp', parse_dates=True)
    print("Dataset loaded")
    
get_new_data()