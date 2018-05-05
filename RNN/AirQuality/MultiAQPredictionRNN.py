"""
.. module:: WindPrediction

WindPrediction
*************

:Description: WindPrediction

:Authors: bejar


:Version:

:Created on: 06/09/2017 9:47

"""

from __future__ import print_function
import os
import sys
import json
import pandas as pd
import numpy as np
import math
import keras.backend as K
import pprint
import tensorflow as tf
from keras_sequential_ascii import keras2ascii
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.optimizers import RMSprop
from tensor_board import TrainValTensorBoard
from time import time
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def lagged_matrix(data, lag=0, ahead=0):
    """
    Returns a matrix with columns that are the steps of the lagged time series
    Last column is the value to predict
    :param data:
    :param lag:
    :return:
    """
    lvect = []
    for i in range(lag):
        lvect.append(data[i: -lag - ahead + i])
    lvect.append(data[lag + ahead:])

    return np.stack(lvect, axis=1)


def architecture(neurons, drop, nlayers, activation, activation_r, rnntype, impl=1):
    """
    RNN architecture

    :return:
    """
    RNN = LSTM if rnntype == 'LSTM' else GRU
    model = Sequential()
    if nlayers == 1:
        model.add(RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2]), implementation=impl,
                      recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r))
    else:
        model.add(RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2]), implementation=impl,
                      recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                      return_sequences=True))
        for i in range(1, nlayers - 1):
            model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl,
                          activation=activation, recurrent_activation=activation_r, return_sequences=True))
        model.add(RNN(neurons, recurrent_dropout=drop, activation=activation,
                      recurrent_activation=activation_r, implementation=impl))

    model.add(Dense(1))

    return model


def load_dataset(config):
    raw_data = np.load('LondonAQ.npz')
    df = pd.DataFrame()

    for site in config['data_names']:
        for var in config['vars']:
            column = site + '_' + var
            var_data = raw_data[site][:, config['vars'].index(var)]
            df[column] = var_data

    return df[config['data_slice']]


def r2_score(y_true, y_pred):
    SS_res = K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


if __name__ == '__main__':
    config = json.load(open('config.json'))
    df = load_dataset(config['data'])
    data = StandardScaler().fit_transform(df.values)

    test_size = math.ceil(data.shape[0] * config['data']['test_fraction'])
    val_size = test_size
    train_size = data.shape[0] - test_size - val_size
    train_data = data[:train_size, :]
    test_data = data[train_size:train_size + test_size, :]
    val_data = data[train_size + test_size:, :]


    train = lagged_matrix(train_data, lag=lag, ahead=ahead)
    test = lagged_matrix(test_data, lag=lag, ahead=ahead)
    val = lagged_matrix(val_data, lag=lag, ahead=ahead)
    train_x, train_y = train[:, :lag], train[:, -1:, -1]
    test_x, test_y = test[:, :lag], test[:, -1:, -1]
    val_x, val_y = val[:, :lag], val[:, -1:, -1]

    # LOGGING
    n_sites = len(np.unique([site.split('_')[0] for site in config['data']['data_slice']]))
    n_vars = len(config['data']['data_slice'])
    dir_name = "vars=%d,nn=%d,nl=%d,lag=%d,drop= %3.2f,ahead=%d" % (
        n_vars,
        config['arch']['neurons'],
        config['arch']['nlayers'],
        lag,
        config['arch']['drop'],
        ahead)

    log_dir = "logs/" + dir_name
    os.makedirs(log_dir)
    tensor_board = TrainValTensorBoard(log_dir=log_dir.format(time()))
    log_file = open('./multi_step_results.txt', 'a')
    sys.stdout = log_file
    # MODEL
    model = architecture(neurons=config['arch']['neurons'],
                         drop=config['arch']['drop'],
                         nlayers=config['arch']['nlayers'],
                         activation=config['arch']['activation'],
                         activation_r=config['arch']['activation_r'],
                         rnntype=config['arch']['rnn'],
                         impl=1)

    optimizer = config['training']['optimizer']

    if optimizer == 'rmsprop':
        optimizer = RMSprop(lr=config['training']['lrate'])

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[r2_score])

    # TRAINING
    callbacks = [tensor_board, EarlyStopping(monitor='val_r2_score', min_delta=0, patience=0, verbose=0, mode='max')]

    history = model.fit(train_x, train_y, batch_size=config['training']['batch'],
                        epochs=config['training']['epochs'],
                        validation_data=(test_x, test_y),
                        verbose=0, callbacks=callbacks)

    # RESULTS
    score = model.evaluate(val_x, val_y, batch_size=config['training']['batch'], verbose=0)
    # pprint.pprint(config)
    # keras2ascii(model)
    print('Lag=', lag, 'Steps ahead=', ahead, '; MSE test= ', [float("{0:.3f}".format(s)) for s in score])
