# -*- coding: utf-8 -*-
"""
Created on Sat May 22 06:50:23 2021
da
@author: Derly garzon
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot

from keras.layers import Input, Dense, Bidirectional, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential

import numpy as np
os.listdir("input/")

print(1)
#coin_dataframes = {}
#1,2,4,8,12,20
#days = 5

#feature_cols = ['Open'] #variable de entrada
inputs_values=[1,10,20,]


'''def convert_comma_int(field):
    try:
        return int(field.replace(',', ''))
    except ValueError:
        return None
for fn in os.listdir("input/"):
    if "bitcoin_cache" in fn:
        continue
    if fn.endswith("ethereum_price.csv"):
        coin_name = fn.split("_")[0]
        print('moneda')
        print(fn)
        df = pd.read_csv(os.path.join("input/", fn), parse_dates=["Date"])
        #df['Market Cap'] = df['Market Cap'].map(convert_comma_int)
        coin_dataframes[coin_name] = df.sort_values('Date')
        #print(coin_dataframes['ethereum'])


coin_dataframes.keys()
coin_dataframes['ethereum'].head()'''

#ethereum value growth
'''plt.figure(figsize=(20,8))
coin_dataframes['ethereum'].plot(x='Date', y='Close')
plt.show()'''

#Compute relative growth and other relative values
def add_relative_columns(df):
    day_diff = df['Close'] - df['Open']
    df['rel_close'] = day_diff / df['Open']
    df['high_low_ratio'] = df['High'] / df['Low']
    df['rel_high'] = df['High'] / df['Close']
    df['rel_low'] = df['Low'] / df['Close']
    
    
'''for df in coin_dataframes.values():
    add_relative_columns(df)'''
    
#coin_dataframes["ethereum"].head()

#Create historical training data

def create_history_frames(coin_dataframes,feature_cols,days):
    history_frames = {}
    for coin_name, df in coin_dataframes.items():
        history_frames[coin_name], x_cols = create_history_frame(df,feature_cols,days)
    return history_frames, x_cols    


def create_history_frame(df,feature_cols,days):
    y_col = ['Close']
    x_cols = []
    history = df[['Date'] + y_col].copy()
    '''days= ventana de tiempo define en el inicio'''
    for n in range(1, days+1):
        for feat_col in feature_cols:
            colname = '{}_{}'.format(feat_col, n)
            history[colname] = df[feat_col].shift(n)
            x_cols.append(colname)
    history = history[days:]
    return history, x_cols

y_col = 'Close'
#print(y_col)

#Define model

def create_model2(x_cols):
    input_layer = Input(batch_shape=(None, len(x_cols), 1))
    layer = Bidirectional(LSTM(64, return_sequences=True))(input_layer)
    layer = Bidirectional(LSTM(64, return_sequences=True))(layer)
    #layer = Bidirectional(LSTM(32, return_sequences=True))(layer)
    #layer = Bidirectional(LSTM(16, return_sequences=True))(layer)
    out = Dense(1)(layer)
    m = Model(inputs=input_layer, outputs=out)
    m.compile("rmsprop", loss='')
    return m

#no_inputs = 100
activation__ = 'relu'
dense__ = 1
optimizer__ = 'adam'
#loss__ = 'mae'

def create_model(no_inputs,loss__,x_cols,days):
    print('shape_cr:::')
    print(str(days),',',str(len(x_cols)))
    
    model = Sequential()
    inp = (days, len(x_cols))
    
    '''if days>1:
        inp=(None,None,days) ''' 
    #model.add(LSTM(no_inputs, activation=activation__, input_shape=(days, len(x_cols))))
     
    #La capa recurrente LSTM compuesta por unidades de memoria
    if days>1:
        model.add(LSTM(no_inputs, activation=activation__,input_shape=(days, 1)))
       
    else:
        model.add(LSTM(no_inputs, activation=activation__, input_shape=(days, 1))) 
    #Una capa completamente conectada que a menudo sigue 
    #capas LSTM y se utiliza para retornar la predicción se llama dense
    model.add(Dense(dense__))
    model.compile(optimizer=optimizer__, loss=loss__) 
    print(model.summary())
    return model


def create_train_test_mtx(history,x_cols):
    X = history[x_cols].values
    print('x shape')
    print(X.shape)
    y = history[y_col].values
    X = X.reshape(X.shape[0], X.shape[1], 1)
    rand_mtx = np.random.permutation(X.shape[0])
    print('totallen')
    print(len(X))
    train_split = int(X.shape[0] * 0.9)
    '''print('train split')
    print(train_split)'''#para imprimir
    #train_indices = rand_mtx[:train_split]
    #test_indices = rand_mtx[train_split:]

    X_train = X[0:train_split]
    X_test = X[train_split+1:X.shape[0]]
    y_train = y[0:train_split]
    y_test =  y[train_split+1:X.shape[0]]
    return X_train, X_test, y_train, y_test

def train_model(model, X, y,epoch_,batch_,validate_):
    ea = EarlyStopping(monitor='val_loss', patience=2)
    val_loss = model.fit(X, y, epochs=epoch_, batch_size=batch_, callbacks=[ea], verbose=0, validation_split=validate_)
    #val_loss = model.fit(X, y, epochs=epoch_, batch_size=batch_, verbose=0)
    #val_loss = model.fit(X, y, epochs=600, batch_size=64, verbose=1, validation_split=.1
    return val_loss

#Train a model for each currency




def prediccion(no_inputs,loss__,epoch_,feature_cols,file_,batch_,validate_,days,pathdirectory):
   
    coin_dataframes = {}
    feature_cols = feature_cols #variable de entrada
    df = pd.read_csv(pathdirectory, parse_dates=["Date"])
    coin_dataframes['ethereum'] = df.sort_values('Date')
    print(coin_dataframes['ethereum'])
    coin_dataframes.keys()
    coin_dataframes['ethereum'].head()
    plt.figure(figsize=(20,8))
    coin_dataframes['ethereum'].plot(x='Date', y='Close')
    plt.show()
    y_col = 'Close' 
    rmse = {}
    pred = {}
    test = {}
    mae = {}
    mae__=[]
    rmse_ =[]
    pred___=[]
    predicciones__=[]
    coin_history, x_cols = create_history_frames(coin_dataframes,feature_cols,days)
    #print(coin_history['ethereum'])

    #print(coin_history['ethereum'])
    #print(x_cols)


    for i in range(1, 2):
        
        print('rango__',i)
        print(i)
        for coin_name, history in coin_history.items():
            #print(model.summary())
            #print(model.loss)#Diagrama del modelo para imprimir
            X_train, X_test, y_train, y_test = create_train_test_mtx(history,x_cols)
            #print(X_train)
            print('shape__')
            print(X_train.shape)
            model = create_model(no_inputs,loss__,x_cols,days)
           
            train_model(model, X_train, y_train,epoch_,batch_,validate_)
            test[coin_name] = y_test
            '''print('X_test')
            print(X_test)
            print('y_test')
            print(y_test)'''
            # run prediction on test set
            print('test__')
            print(X_test)
            print(len(X_test))
            pred[coin_name] = model.predict(X_test)
            
            for t in range(len(pred[coin_name])):
                '''print(t)
                print(np.mean(sum(pred[coin_name][t])))'''
                pred[coin_name][t] = [np.mean(sum(pred[coin_name][t]))]
                pred___.append([np.mean(sum(pred[coin_name][t]))])
            
            
                   
            pred___ = np.array(pred___)
            '''print('y_result')
            print(pred___)'''
            y_pred_col= pred___.reshape(pred___.shape[0])
            
            
            
            #y_pred_col= pred[coin_name].reshape(pred[coin_name].shape[0])
            
            if i == 1:
                pyplot.plot(y_test)
                pyplot.plot(y_pred_col, color='red')
                pyplot.xlabel('Grafica y_test vs y_prediction  __' +activation__ + ' ' + str(no_inputs) + ' ' + feature_cols[0] + ' ' +loss__ +' dense: ' + str(dense__) + 'epochs:' + str(epoch_))
                #pyplot.xlabel('Grafia y_test vs y_prediction  __' +activation__ + ' ' + str(no_inputs) + ' ' + feature_cols[0] + '  msle' )
                pyplot.savefig('C:/Users/Derly garzon/Documents/trabajo_de_grado_2/'+ file_)
            
                pyplot.show()
                
           # compute test loss
            #mae[coin_name] =  mean_absolute_error(y_test, pred[coin_name])
            '''print('test')
            print(len(y_test))
            print(y_test)''' #para imprimir
            y_pred = pred___
            '''print('y prediccion')
            print(len(pred[coin_name]))
            print(y_pred.shape)
            print(pred[coin_name])'''#para imprimir
            if days == 1:
                y_pred_col =y_pred
            else:
                y_pred_col = np.delete(y_pred,np.s_[days-(days-1):days], axis=1)
    
            '''print('pred:y_pred_col')
            print(y_pred_col.shape);
            print(y_pred_col.shape[0])'''#para imprimir
    
            y_pred_col_resh = y_pred_col.reshape(y_pred_col.shape[0])#se puede hacer un fit para mejorar resultados en alguna funcion
            '''print('y_pred_col')
            print(y_pred_col_resh)'''
            y_absoute_error_sum = abs(sum(y_pred_col_resh - y_test))#para imprimir
            mae = y_absoute_error_sum / len(pred[coin_name])
            mae__.append(mae)
            rmse[coin_name] = np.sqrt(np.mean((y_pred_col_resh  - y_test)**2))
            rmse_.append(rmse['ethereum'])
            
            print(' ')
            print('prueba ' + str(i) +' ' + feature_cols[0] + ' ' + str(days) + ' dias') 
            print('Mae:',mae)
            print('rmse:',rmse['ethereum'])
            
            print(' ')
            print('parametros')
            '''print('activation:' , activation__,',',)
            print('dense:' , dense__,',',)
            print('optimizer:' , optimizer__,',',)
            print('loss:' , loss__,',',)
            print('timesteps:', days,',',)
            print('variable:', feature_cols[0],',',)'''
           
        predicciones__.append(pred___);
        
        pred___=[]
        
            
        #mae[coin_name] = np.sqrt(sum(pred[coin_name] - y_test)) np.abs(out_arr/len(y_test))
        #print(out_arr)
        #mae[coin_name] =  np.abs(y_test - newarray)
    print('no_inputs:',no_inputs,',','activation:' , activation__,',','dense:' , dense__,',','optimizer:' , optimizer__,',','timesteps:', days,',','variable:', feature_cols[0],',',)
    print('mae')
    print(sorted(mae__))
    prome_m = sum(mae__)/len(mae__)
    print('promedio',prome_m)
        
    print('rmse')
    print(sorted(rmse_))
    prome_r = sum(rmse_)/len(rmse_)
    print('promedio',prome_r)
    print('loss:' , loss__ , ' ', prome_r)
    #return {'timesteps':days, 'mae:':prome_m, 'units:' : no_inputs, 'rmse':prome_r, 'mae:':prome_m, 'validttion':validate_, 'epochs': epoch_, 'feature:':feature_cols ,'loss:' : loss__}
    '''print('predicciones')
    print(predicciones__)
    print(type(predicciones__))
    print(predicciones__[0])
    print(predicciones__[0].shape)'''
    res__Pred = predicciones__[0].reshape(predicciones__[0].shape[0])
    print(len(res__Pred))
    return res__Pred;




def prediccion__(no_inputs,loss__,epoch_,feature_cols,file_,batch_,validate_,days,pathdirectory):
   
    coin_dataframes = {}
    feature_cols = feature_cols #variable de entrada
    df = pd.read_csv(pathdirectory, parse_dates=["Date"])
    coin_dataframes['ethereum'] = df.sort_values('Date')
    print(coin_dataframes['ethereum'])
    coin_dataframes.keys()
    coin_dataframes['ethereum'].head()
    plt.figure(figsize=(20,8))
    coin_dataframes['ethereum'].plot(x='Date', y='Close')
    plt.show()
    y_col = 'Close' 
    rmse = {}
    pred = {}
    test = {}
    mae = {}
    mae__=[]
    rmse_ =[]
    pred___=[]
    predicciones__=[]
    coin_history, x_cols = create_history_frames(coin_dataframes,feature_cols,days)
    #print(coin_history['ethereum'])

    #print(coin_history['ethereum'])
    #print(x_cols)


    for i in range(1, 2):
        
        print('rango__',i)
        print(i)
        for coin_name, history in coin_history.items():
            #print(model.summary())
            #print(model.loss)#Diagrama del modelo para imprimir
            X_train, X_test, y_train, y_test = create_train_test_mtx(history,x_cols)
            #print(X_train)
            print('shape__')
            print(X_train.shape)
            model = create_model(no_inputs,loss__,x_cols,days)
           
            train_model(model, X_train, y_train,epoch_,batch_,validate_)
            test[coin_name] = y_test
            '''print('X_test')
            print(X_test)
            print('y_test')
            print(y_test)'''
            # run prediction on test set
            print('test__')
            print(X_test.shape)
            X_test__ = X_test.reshape(187)
            X_test___ = X_test__[0:1]
            predV = []
            diasapredecir = 200
            
            for h in range(diasapredecir):
                print(h,'Pr---')
                X_test___ = X_test___.reshape(X_test___.shape[0],1,1)
                print('X_test___')
                print(X_test___)
                predi_un = model.predict(X_test___)
                predi_un = predi_un.reshape(predi_un.shape[0])
                print('pred')
                print(predi_un)
                '''for t in range(len(predi_un)):
                    print('pred',t)
                    print(predi_un[t])
                print(predi_un[len(predi_un)-1]);'''
                
                
                
                X_test___ = X_test___.reshape(X_test___.shape[0])
                
                
                print(X_test___)
                
                #X_test___ = np.append(predi_un[len(predi_un)-1], X_test___)
                X_test___ = predi_un
                predV.append(X_test___)
                print(X_test___)
                
                
            print('prediccion1()')
            print(predV)
            print('------')
            print(len(predV)-1)
                
                
            X_test___final = ();
            print(type(X_test___))
            
            print(X_test___)
            print(X_test)
            print(len(X_test))
            pred[coin_name] = model.predict(X_test)
            
            for t in range(len(pred[coin_name])):
                '''print(t)
                print(np.mean(sum(pred[coin_name][t])))'''
                pred[coin_name][t] = [np.mean(sum(pred[coin_name][t]))]
                pred___.append([np.mean(sum(pred[coin_name][t]))])
            
            '''yhat = output[0]
            predictions.append(yhat)'''
        
                   
            pred___ = np.array(pred___)
            '''print('y_result')
            print(pred___)'''
            y_pred_col= pred___.reshape(pred___.shape[0])
            
            
            
            #y_pred_col= pred[coin_name].reshape(pred[coin_name].shape[0])
            
            if i == 1:
                pyplot.plot(y_test)
                pyplot.plot(predV, color='red')
                pyplot.xlabel('Grafica y_test vs y_prediction  __' +activation__ + ' ' + str(no_inputs) + ' ' + feature_cols[0] + ' ' +loss__ +' dense: ' + str(dense__) + 'epochs:' + str(epoch_))
                #pyplot.xlabel('Grafia y_test vs y_prediction  __' +activation__ + ' ' + str(no_inputs) + ' ' + feature_cols[0] + '  msle' )
                pyplot.savefig('C:/Users/Derly garzon/Documents/trabajo_de_grado_2/'+ file_)
            
                pyplot.show()
                
           # compute test loss
            #mae[coin_name] =  mean_absolute_error(y_test, pred[coin_name])
            '''print('test')
            print(len(y_test))
            print(y_test)''' #para imprimir
            y_pred = pred___
            '''print('y prediccion')
            print(len(pred[coin_name]))
            print(y_pred.shape)
            print(pred[coin_name])'''#para imprimir
            if days == 1:
                y_pred_col =y_pred
            else:
                y_pred_col = np.delete(y_pred,np.s_[days-(days-1):days], axis=1)
    
            '''print('pred:y_pred_col')
            print(y_pred_col.shape);
            print(y_pred_col.shape[0])'''#para imprimir
    
            y_pred_col_resh = y_pred_col.reshape(y_pred_col.shape[0])#se puede hacer un fit para mejorar resultados en alguna funcion
            '''print('y_pred_col')
            print(y_pred_col_resh)'''
            y_absoute_error_sum = abs(sum(y_pred_col_resh - y_test))#para imprimir
            mae = y_absoute_error_sum / len(pred[coin_name])
            mae__.append(mae)
            rmse[coin_name] = np.sqrt(np.mean((y_pred_col_resh  - y_test)**2))
            rmse_.append(rmse['ethereum'])
            
            print(' ')
            print('prueba ' + str(i) +' ' + feature_cols[0] + ' ' + str(days) + ' dias') 
            print('Mae:',mae)
            print('rmse:',rmse['ethereum'])
            
            print(' ')
            print('parametros')
            '''print('activation:' , activation__,',',)
            print('dense:' , dense__,',',)
            print('optimizer:' , optimizer__,',',)
            print('loss:' , loss__,',',)
            print('timesteps:', days,',',)
            print('variable:', feature_cols[0],',',)'''
           
        predicciones__.append(pred___);
        
        pred___=[]
        
            
        #mae[coin_name] = np.sqrt(sum(pred[coin_name] - y_test)) np.abs(out_arr/len(y_test))
        #print(out_arr)
        #mae[coin_name] =  np.abs(y_test - newarray)
    print('no_inputs:',no_inputs,',','activation:' , activation__,',','dense:' , dense__,',','optimizer:' , optimizer__,',','timesteps:', days,',','variable:', feature_cols[0],',',)
    print('mae')
    print(sorted(mae__))
    prome_m = sum(mae__)/len(mae__)
    print('promedio',prome_m)
        
    print('rmse')
    print(sorted(rmse_))
    prome_r = sum(rmse_)/len(rmse_)
    print('promedio',prome_r)
    print('loss:' , loss__ , ' ', prome_r)
    #return {'timesteps':days, 'mae:':prome_m, 'units:' : no_inputs, 'rmse':prome_r, 'mae:':prome_m, 'validttion':validate_, 'epochs': epoch_, 'feature:':feature_cols ,'loss:' : loss__}
    '''print('predicciones')
    print(predicciones__)
    print(type(predicciones__))
    print(predicciones__[0])
    print(predicciones__[0].shape)'''
    res__Pred = predicciones__[0].reshape(predicciones__[0].shape[0])
    print(len(res__Pred))
    return res__Pred;





def loos_p():    
    mae_def=[100, 200, 500]
    res_L_mae = []
    res_L_mse = []
    #mae_def=[]
    
    if len(mae_def)==0:
        print(213)
        prediccion(100,'mae',600)
    else:
        
        for i in range(0, len(mae_def)):
            res_L_mae.append(prediccion(mae_def[i],'mae',600))
            res_L_mse.append(prediccion(mae_def[i],'mse',600))
    
    print('res_L_mae_loss')
    print(res_L_mae)
    print(res_L_mse)
        
#loos_p()

    
def epochs_p_():  
    epochs_der=[1,50,100,200,300,600,900,1200]
    
    res_L_ep=[]
   
    if len(epochs_der)==0:
        print(213)
        prediccion(100,'mae',1200,['Low'],'plt.png',64,.1,1)
    else:
        for i in range(0, len(epochs_der)):
            
            res_L_ep.append(prediccion(100,'mae',epochs_der[i],['Low'],'Low'+ '_epoch_'+str(epochs_der[i]) +'.png',64,.1,1))
            res_L_ep.append(prediccion(100,'mae',epochs_der[i],['High'],'High'+ '_epoch_'+str(epochs_der[i]) +'.png',64,.1,1))

    print('res_L_ep')    
    print(res_L_ep)
    return ''
    
#epochs_p_()      
def batch_p():
    batchs_der=[1,16,32,64,128,256,512,1024]
    res_L_bat = []
    if len(batchs_der)==0:
        prediccion(100,'mae',100,['Low'],'plt.png',64,.1,1)
    else:
        for i in range(0, len(batchs_der)):
            res_L_bat.append(prediccion(100,
                                        'mae',
                                        100,
                                        ['Low'],
                                        'Low_batch_'+str(batchs_der[i]) +'.png',
                                        batchs_der[i],.1,1))
    print('res_L_bat')
    print(res_L_bat)
    
        
#batch_p()  

    
    
def validation_sp_p():
    val_spli=[.1,.2,.3,.4,.5,.6,.7,.8,.9]
    #val_spli=[]
    res_L_val = []
    if len(val_spli)==0:
         print(prediccion(100,'mae',100,['Low'],'plt.png',128,.1,1))
    else:
        for i in range(0, len(val_spli)):
            res_L_val.append(prediccion(100,'mae',100,['Low'],'Low_validation_split_'+str(val_spli[i]) +'.png',128,val_spli[i],1))
    print('res_L_val')   
    print(res_L_val)
    
#validation_sp_p()  


def timesteps():
    val_time=[1,2,4,8,12,24]
    res_L_time=[]
    if len(val_time)==0:
        print(prediccion(100,'mae',100,['Low'],'plt.png',128,.1,1))
    else:
        for i in range(0, len(val_time)):
            res_L_time.append(prediccion(100,'mae',100,['Low'],'Low_time_step_'+str(val_time[i]) +'.png',128,.1,val_time[i]))
            
    print('res_L_time')   
    print(res_L_time)

#timesteps()

def pruebas_resultados():
    #units_p=[1,10,20,40,50,70, 100, 110, 120,130, 170,200,500]
    units_p=[1000,1500]
    res_p=[]
    if len(units_p)==0:
        print(prediccion(100,'mae',100,['Low'],'Low_units_' + '.png',128,.1))
    else:
        for i in range(0, len(units_p)):
            res_p.append(prediccion(units_p[i],'mae',100,['Low'],'Low_units_' + str(units_p[i]) + '.png',128,.1))
    print(res_p)  
        
#prediccion(100,'mae',100,['Low'],'Low_time_step_' +'.png',128,.1,1,'C:/Users/ethereum_price.csv')   
#pruebas_resultados()
#print(prediccion(100,'mae',100,['Low'],'Low_timesteps_' + str(days) + '.png',128,.1))
'''pred_sign = {coin_name: np.sign(pred[coin_name]) * np.sign(test[coin_name]) for coin_name in pred.keys()}
for coin, val in sorted(pred_sign.items()):
    cnt = np.unique(pred_sign[coin], return_counts=True)[1]
    print("[{}] pos/neg change guessed correctly: {}, incorrectly: {}, correct%: {}".format(
        coin, cnt[0], cnt[1], cnt[0]/ (cnt[0]+cnt[1]) * 100))'''
    
        