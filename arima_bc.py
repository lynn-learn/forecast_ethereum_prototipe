# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 15:11:03 2021

@author: Derly garzon
"""

from sklearn.metrics import mean_absolute_error
import os
import pandas as pd
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt



#def arimaModel(diasApredecir,p,d,q,path):
def arimaModel(diasApredecir,pathdirectory,p,d,q,var_):    
    coin_dataframes = {}
    #df = pd.read_csv(os.path.join("input/", 'ethereum_price.csv'), parse_dates=["Date"])
    df = pd.read_csv(pathdirectory, parse_dates=["Date"])
    #print(df)
    
    df = df[["Date", var_]]
    size = int(len(df) * 0.9)
    coin_dataframes['ethereum_price'] = df.sort_values('Date');
    coin_ordenado = df.sort_values('Date');
    print('datesort')
    print(type(coin_dataframes['ethereum_price']))
    
    coin_dataframes['ethereum_price'] = coin_dataframes['ethereum_price'][[var_]]
    print('dddd')
    
    X = coin_dataframes['ethereum_price'].values
    #print(X)
    
    print('xlen')
    print(len(X))
    train, test = X[0:size], X[size:len(X)]
    train2, test2, = coin_ordenado[0:size], coin_ordenado[size:len(X)] 
    print('train2')
    #print(train2)
    print()
    history = [x for x in train]
    '''print('test')
    print(test)
    print('train')
    print(train)
    print('history')
    print(history)'''
    
    
    predictions = list()
    print('history')
    #print(history)
    for t in range(diasApredecir):
        model = ARIMA(history, order=(p,d,q))
        model_fit = model.fit()
        output = model_fit.forecast()
        
        yhat = output[0]
        predictions.append(yhat)
        pd.to_numeric(yhat, errors='coerce')
        obs = ([yhat])
        history.append(obs)    
    print(predictions)
    return predictions
        
#def arimaModel(diasApredecir,pathdirectory,p,d,q,var_) 
#arimaModel(5,'C:/Users/Derly garzon/ethereum_price.csv',1,1,1,"Low")