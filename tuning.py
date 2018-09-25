import numpy as np
from scipy import misc
import cv2
import csv
import pickle
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils import np_utils
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers import GRU, LSTM
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import losses

from ModisUtils.misc import *

def argParse():
    parser = argparse.ArgumentParser(description='ConvLstm \
                                                  fine tuning')
    parser.add_argument('-t', '--time-steps', type=int, 
                        help='length of LSTM time steps')
    parser.add_argument('-f', '--filters', type=int,
                        help='number of convolutional filters')
    parser.add_argument('-k', '--kernel-size', type=int,
                        help='size of convolutional kernel')
    parser.add_argument('-n', '--n-hidden-layers', type=int,
                        help='number of hidden layers')
    parser.add_argument('-e', '--epochs', type=int,
                        help='number of epochs when training')
    return parser.parse_args()


# main function
def main():
    # Load argument
    parser = argParse()
    timeSteps = parser.time_steps
    filters = parser.filters
    kernel_size = parser.kernel_size
    n_hidden_layers = parser.n_hidden_layers
    epochs = parser.epochs

    # Create data file if not created yet
    createFileData(dataDir='MOD13Q1', reservoirsUse=[0], 
                   bandsUse=['NIR'], timeSteps=timeSteps)

    # Load data
    ##reduceSize = None
    reduceSize = (40,40)
    if not os.path.isdir(os.path.join('data', str(timeSteps))):
        os.makedirs(os.path.join('data', str(timeSteps)))
    (train_data, train_target) = get_data('train', timeSteps, 
                                          reduceSize=reduceSize)
    (val_data, val_target) = get_data('val', timeSteps, 
                                      reduceSize=reduceSize)
    (test_data, test_target) = get_data('test', timeSteps, 
                                        reduceSize=reduceSize)

    # Scale data
    scaler = MinMaxScaler()
    scaler.fit(train_data.reshape(train_data.shape[0]*\
                              train_data.shape[1]*train_data.shape[2],
                              train_data.shape[3]*train_data.shape[4]))
    train_data = scaleAsTrain(train_data, scaler)
    train_target = scaleAsTrain(train_target, scaler)
    val_data = scaleAsTrain(val_data, scaler)
    val_target = scaleAsTrain(val_target, scaler)
    test_data = scaleAsTrain(test_data, scaler)
    test_target = scaleAsTrain(test_target, scaler)

    # Create model and fit
    # Or load model if already train and save
    dir_prefix = createDirPrefix(timeSteps, filters, 
                                 kernel_size, n_hidden_layers)
    if not os.path.isdir(os.path.join('cache', dir_prefix)):
        os.makedirs(os.path.join('cache', dir_prefix))

    trained = False
    weight_best_path = os.path.join('cache', dir_prefix, 
                                        'weights_best.h5')
    if os.path.isfile(weight_best_path):
        trained = True
        seq = load_model(weight_best_path)
    else:
        input_shape = (None, train_data.shape[2], 
                       train_data.shape[3], train_data.shape[4])
        seq = createModel(filters, kernel_size, 
                          input_shape, n_hidden_layers)
        seq.compile(optimizer = "sgd", 
                    loss = losses.mean_squared_error,
                    metrics =["mse"])

    # Set callback functions to early stop training 
    # and save the best model so far
    early_stopping_monitor = EarlyStopping(monitor='val_loss', 
                                           patience=15)
    callbacks = [early_stopping_monitor,
                 ModelCheckpoint(filepath=weight_best_path, 
                         monitor='val_loss', save_best_only=True)]
    
    if trained:
        curEpochs = epochs - loadLatestModel(timeSteps, filters, 
                                        kernel_size, n_hidden_layers)
    else:
        curEpochs = epochs

    history = seq.fit(train_data, train_target, batch_size=10,
                  epochs=epochs, 
                  callbacks=callbacks,
                  validation_data=(val_data, val_target))

    if early_stopping_monitor.stopped_epoch > 0:
        curEpochs = epochs - curEpochs \
                           + early_stopping_monitor.stopped_epoch
    
    dir_prefix = createDirPrefix(timeSteps, filters, kernel_size, 
                                 n_hidden_layers, curEpochs)

    save_model(seq, timeSteps, filters, kernel_size, 
               n_hidden_layers, curEpochs)

    ## Plot loss
    if not os.path.isdir(os.path.join('results', dir_prefix)):
        os.makedirs(os.path.join('results', dir_prefix))

    plt.plot(history.history['loss'], color='r', label='loss')
    plt.plot(history.history['val_loss'], color='b', label='val_loss')
    plt.title('Loss and Val_loss')
    plt.legend()
    plt.savefig(os.path.join('results', dir_prefix, 'loss.png'))
    
    # Visualize predicted result
    ## Remove old log
    seq = load_model(weight_best_path)
    log_file_path = os.path.join('results', dir_prefix, 'log.txt')
    if os.path.isfile(log_file_path):
        os.remove(log_file_path)
    
    for i in range(test_data.shape[0]):
        predictAndVisualize(i, test_data, test_target, timeSteps, 
                            filters, kernel_size, n_hidden_layers, 
                            curEpochs, seq)



if __name__ == '__main__':
    main()