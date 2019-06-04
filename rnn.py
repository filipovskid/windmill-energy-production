# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import keras
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


def normalize(data, data_min, data_max):
    data_nrm = (data - data_min) / (data_max - data_min)
    return data_nrm


def preprocess_data(train_df, test_df):
    train_data = train_df.values
    train_data = train_data.astype('float32')
    
    test_data = test_df.values
    test_data = test_data.astype('float32')
    
    cols = train_data.shape[1]
    
#    data_min = []
#    data_max = []
#    
#    for atr in range(0, cols):
#        data_min.append(np.amin(train_data[:, atr], axis=0))
#        data_max.append(np.amax(train_data[:, atr], axis=0))
#        train_data[:, atr] = normalize(train_data[:, atr], data_min[atr], data_max[atr])
#        
#    for atr in range(0, cols):
#        test_data[:, atr] = normalize(test_data[:, atr], data_min[atr], data_max[atr])
    scaler = preprocessing.MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
        
    return train_data[:, 0:cols-1], test_data[:, 0:cols-1], train_data[:, cols-1], test_data[:, cols-1], scaler #, data_min[cols-1], data_max[cols-1] 
        

def build_model():
    model = Sequential()
#                        11              (None, 11)
    model.add(LSTM(units=7, input_shape=(None, 7), return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=64, activation='tanh'))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=1, activation='relu'))
    
    opt = keras.optimizers.Adam()
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=[keras.metrics.MAPE])
    
    return model

def lstm_reshape(x_train, x_test, y_train, y_test):
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    y_train = np.reshape(y_train, (y_train.shape[0], 1, 1))
    y_test = np.reshape(y_test, (y_test.shape[0], 1, 1))
    
    return x_train, x_test, y_train, y_test

def revert_reshape(x_train, x_test, y_train, y_test):
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[2]))
    y_train = np.reshape(y_train, (y_train.shape[0]))
    y_test = np.reshape(y_test, (y_test.shape[0]))
    
    return x_train, x_test, y_train, y_test
    

def mape(y_predicted, y_observed):
    a = []
    f = []
    
    for i in range(0, y_observed.shape[0]):
        if y_observed[i] != 0:
            a.append(y_observed[i])
            f.append(y_predicted[i])
            
    a = np.array(a)
    f = np.array(f)
    
    return np.sum(np.abs((a - f)/a))/f.shape[0]
#    return np.sum(np.abs((y_observed - y_predicted)/y_observed))/y_predicted.shape[0]
    

def main():
    train_df = pd.read_csv('data/wpph_c_train.csv', header=None)
    test_df = pd.read_csv('data/wpph_c_test.csv', header=None)
#    pd.read_csv()
    
#    x_train, x_test, y_train, y_test, data_min, data_max = preprocess_data(train_df, test_df)
    x_train, x_test, y_train, y_test, scaler = preprocess_data(train_df, test_df)
    x_train, x_test, y_train, y_test = lstm_reshape(x_train, x_test, y_train, y_test)
    
    model = build_model()
    
    epochs_no = 100
    model.fit(x_train, y_train, batch_size=10, epochs=epochs_no)
    model.save('models/lstm_dense_c.h5') 
    
    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    
    train_predictions = np.reshape(train_predictions, (train_predictions.shape[0]))
    test_predictions = np.reshape(test_predictions, (test_predictions.shape[0]))

    x_train, x_test, y_train, y_test = revert_reshape(x_train, x_test, y_train, y_test)

    
    print('Train MAPE:', mape(train_predictions, y_train))
    print('Test MAPE:', mape(test_predictions, y_test))
    
    
if __name__ == '__main__':
    main()