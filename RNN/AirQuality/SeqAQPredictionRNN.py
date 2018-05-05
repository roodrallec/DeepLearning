"""An implementation of sequence to sequence learning for performing addition
Input: "535+61"
Output: "596"
Padding is handled by using a repeated sentinel character (space)
Input may optionally be inverted, shown to increase performance in many tasks in:
"Learning to Execute"
http://arxiv.org/abs/1410.4615
and
"Sequence to Sequence Learning with Neural Networks"
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
Theoretically it introduces shorter term dependencies between source and target.
Two digits inverted:
+ One layer LSTM (128 HN), 5k training examples = 99% train/test accuracy in 55 epochs
Three digits inverted:
+ One layer LSTM (128 HN), 50k training examples = 99% train/test accuracy in 100 epochs
Four digits inverted:
+ One layer LSTM (128 HN), 400k training examples = 99% train/test accuracy in 20 epochs
Five digits inverted:
+ One layer LSTM (128 HN), 550k training examples = 99% train/test accuracy in 30 epochs
"""
from keras import layers
from six.moves import range
import argparse
import json
import pandas as pd
import numpy as np
import math
import keras.backend as K
from keras.models import Sequential
from time import time
from sklearn.preprocessing import StandardScaler


class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self, chars):
        """Initialize character table.
        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One hot encode given string C.
        # Arguments
            num_rows: Number of rows in the returned one hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)


def r2_score(y_true, y_pred):
    SS_res = K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def load_dataset(config):
    raw_data = np.load('LondonAQ.npz')
    df = pd.DataFrame()

    for site in config['data_names']:
        for var in config['vars']:
            column = site + '_' + var
            var_data = raw_data[site][:, config['vars'].index(var)]
            df[column] = var_data

    return df[config['data_slice']]


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)", action='store_true', default=False)
    parser.add_argument('--gpu', help="Use LSTM/GRU gopu implementation", action='store_true', default=False)
    args = parser.parse_args()

    verbose = 1 if args.verbose else 0
    impl = 2 if args.gpu else 0

    config = json.load(open('config.json'))
    df = load_dataset(config['data'])
    print(df.head())
    data = StandardScaler().fit_transform(df.values)

    test_size = math.ceil(data.shape[0] * config['data']['test_fraction'])
    val_size = test_size
    train_size = data.shape[0] - test_size - val_size
    train_data = data[:train_size, :]
    test_data = data[train_size:train_size + test_size, :]
    val_data = data[train_size + test_size:, :]

    lag = config['data']['lag']
    ahead = config['data']['ahead']
    train = lagged_matrix(train_data, lag=lag, ahead=ahead)
    test = lagged_matrix(test_data, lag=lag, ahead=ahead)
    val = lagged_matrix(val_data, lag=lag, ahead=ahead)

    x_train, y_train = train[:, :lag], train[:, -1:, -1]
    x_test, y_test = test[:, :lag], test[:, -1:, -1]
    x_val, y_val = val[:, :lag], val[:, -1:, -1]

    n_vars = len(config['data']['data_slice'])

    print('Training Data:')
    print(x_train.shape)
    print(y_train.shape)

    print('Validation Data:')
    print(x_val.shape)
    print(y_val.shape)

    ############################################
    # Model

    # Try replacing GRU, or SimpleRNN.
    RNN = layers.LSTM
    HIDDEN_SIZE = 128
    BATCH_SIZE = 128
    LAYERS = 1
    Dropout = 0.0

    print('Build model...')
    model = Sequential()
    # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
    # Note: In a situation where your input sequences have a variable length,
    # use input_shape=(None, num_feature).
    model.add(RNN(HIDDEN_SIZE, input_shape=(lag, n_vars), recurrent_dropout=Dropout, implementation=impl))
    # As the decoder RNN's input, repeatedly provide with the last hidden state of
    # RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
    # length of output, e.g., when DIGITS=3, max output is 999+999=1998.
    DIGITS = 3
    model.add(layers.RepeatVector(DIGITS + 1))
    # The decoder RNN could be multiple layers stacked or a single layer.
    for _ in range(LAYERS):
        # By setting return_sequences to True, return not only the last output but
        # all the outputs so far in the form of (num_samples, timesteps,
        # output_dim). This is necessary as TimeDistributed in the below expects
        # the first dimension to be the timesteps.
        model.add(RNN(HIDDEN_SIZE, return_sequences=True, recurrent_dropout=Dropout, implementation=impl))

    # Apply a dense layer to the every temporal slice of an input. For each of step
    # of the output sequence, decide which character should be chosen.
    model.add(layers.TimeDistributed(layers.Dense(n_vars)))
    model.add(layers.Activation('softmax'))

    ############################################
    # Training

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=[r2_score])
    model.summary()

    # Train the model each generation and show predictions against the validation
    # dataset.

    iterations = 50
    for iteration in range(1, iterations + 1):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(x_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=1,
                  validation_data=(x_val, y_val), verbose=verbose)
        # Select 10 samples from the validation set at random so we can visualize
        # errors.
        for i in range(10):
            ind = np.random.randint(0, len(x_val))
            rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
            preds = model.predict_classes(rowx, verbose=0)
            # q = ctable.decode(rowx[0])
            # correct = ctable.decode(rowy[0])
            # guess = ctable.decode(preds[0], calc_argmax=False)
            # print('Q', q[::-1] if INVERT else q)
            # print('T', correct)
            # if correct == guess:
            #     print('+', end=" ")
            # else:
            #     print('-', end=" ")
            # print(guess)
        print('---')
    print()
    print("Ending:", time.ctime())