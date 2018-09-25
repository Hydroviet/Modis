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

# Utils functions
def createFileData_1(dataDir, reservoirsUse, bandsUse, timeSteps, 
                     yearRange, inputFile, targetFile):
    inputF = open(inputFile, "w")
    targetF = open(targetFile, "w")
    writerInput = csv.writer(inputF)
    writertarget = csv.writer(targetF)
    
    timeSteps += 1
    for reservoir in reservoirsUse:
        for year in yearRange:
            listFilesInWindow = []
            yearDir = dataDir + '/' + str(reservoir) + '/' + str(year)
            listFolders = os.listdir(yearDir)
            listFolders = sorted(listFolders, key=lambda x: int(x))
            
            for i in np.arange(timeSteps):
                folder = listFolders[i]
                dayDir = yearDir + '/' + folder
                listFiles = os.listdir(dayDir)
                for band in bandsUse:
                    for file in listFiles:
                        if band in file:
                            listFilesInWindow.append(dayDir + '/' 
                                                            + file)
            writerInput.writerow(listFilesInWindow[:-1])
            writertarget.writerow(listFilesInWindow[-1:])
            
            for i in np.arange(timeSteps, len(listFolders)):
                folder = listFolders[i]
                listFilesInWindow = listFilesInWindow[1:]
                dayDir = yearDir + '/' + folder
                listFiles = os.listdir(dayDir)
                for band in bandsUse:
                    for file in listFiles:
                        if band in file:
                            listFilesInWindow.append(dayDir + '/' 
                                                            + file)
                writerInput.writerow(listFilesInWindow[:-1])
                writertarget.writerow(listFilesInWindow[-1:])

    inputF.close()
    targetF.close()
    
    return listFilesInWindow


def createFileData(dataDir, reservoirsUse, bandsUse, timeSteps, 
                   startYear=2001, endYear=2017, 
                   valPercent=0.2, testPercent=0.2):
    train_year = [2002, 2007, 2009, 2010, 2014, 2005, 2003, 2015, 2011]
    val_year = [2001, 2006, 2016, 2013]
    test_year = [2008, 2017, 2012, 2004]
    
    # Single reservoirUse supported
    if not os.path.isdir('data_file/{}/{}'.format(reservoirsUse[0], timeSteps)):
        os.makedirs('data_file/{}/{}'.format(reservoirsUse[0], timeSteps))
        # train
        createFileData_1(dataDir, reservoirsUse, bandsUse, 
                         timeSteps, train_year, 
                         'data_file/{}/{}/train_data.csv'\
                            .format(reservoirsUse[0], timeSteps),
                         'data_file/{}/{}/train_target.csv'\
                            .format(reservoirsUse[0], timeSteps))
        # val
        createFileData_1(dataDir, reservoirsUse, bandsUse, 
                         timeSteps, val_year, 
                         'data_file/{}/{}/val_data.csv'\
                            .format(reservoirsUse[0], timeSteps),
                         'data_file/{}/{}/val_target.csv'\
                            .format(reservoirsUse[0], timeSteps))
    # test
    return createFileData_1(dataDir, reservoirsUse, bandsUse, 
                     timeSteps, test_year, 
                     'data_file/{}/{}/test_data.csv'\
                        .format(reservoirsUse[0], timeSteps),
                     'data_file/{}/{}/test_target.csv'\
                        .format(reservoirsUse[0], timeSteps))


def get_data_and_target_path(data_file, target_file):
    with open(data_file, "r") as dataF:
        reader = csv.reader(dataF)
        data_paths = [row for row in reader]
    with open(target_file, "r") as targetF:
        reader = csv.reader(targetF)
        target_paths = [row[0] for row in reader]
    return data_paths, target_paths


def get_im(path, reduceSize=None):
    # reduceSize can be a tuple, example: (128, 96)
    img = misc.imread(path)
    # Reduce size
    if reduceSize is not None:
        img = misc.imresize(img, reduceSize)
    return img


def load_data_target(data_file, target_file, reduceSize=None):
    data_paths, target_paths = get_data_and_target_path(data_file,
                                                        target_file)
    X_ = []
    y_ = []
    for data_path_list in data_paths:
        currentX_ = []
        for fl in data_path_list:
            img = get_im(fl, reduceSize)
            currentX_.append(img)
        X_.append(currentX_)
    
    for target_path in target_paths:
        img = get_im(target_path, reduceSize)
        y_.append(img)
        
    return X_, y_


# Data
def cache_data(data, path):
    token = path.split('/')
    dir_prefix = ''
    for t in token[:-1]:
        dir_prefix = os.path.join(dir_prefix, t)
    try:
        os.makedirs(dir_prefix)
    except:
        pass
    file = open(path, 'wb')
    pickle.dump(data, file)
    file.close()


def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
        file.close()
    return data


def createDirPrefix(timeSteps, filters, kernel_size, 
                    n_hidden_layers, epochs=None):
    dir_prefix = os.path.join('timeSteps_{}'.format(str(timeSteps)), 
                              'filters_{}'.format(str(filters)), 
                              'kernel_size_{}'\
                                .format(str(kernel_size)),
                              'n_hidden_layers_{}'\
                                .format(str(n_hidden_layers)))
    if epochs is not None:
        dir_prefix = os.path.join(dir_prefix, 
                                  'epochs_{}'.format(str(epochs)))
    return dir_prefix


# Model
def save_model(model, timeSteps, filters, kernel_size, 
               n_hidden_layers, epochs):
    
    dir_prefix = createDirPrefix(timeSteps, filters, kernel_size, 
                                 n_hidden_layers, epochs)
    json_string = model.to_json()
    if not os.path.isdir(os.path.join('cache', dir_prefix)):
        os.makedirs(os.path.join('cache', dir_prefix))
    open(os.path.join('cache', dir_prefix, 'architecture.json'), 'w')\
        .write(json_string)
    weight_path = os.path.join('cache', dir_prefix,
        'model_weights.h5')
    model.save(weight_path, overwrite=True)


def loadModel(timeSteps, filters, kernel_size, 
               n_hidden_layers, epochs):
    dir_prefix = createDirPrefix(timeSteps, filters, kernel_size, 
                                 n_hidden_layers, epochs)
    model = model_from_json(open(os.path.join('cache', dir_prefix,
                                    'architecture.json')).read())
    model.load_weights(os.path.join('cache', dir_prefix,
                                    'model_weights.h5'))
    return model


def loadLatestModel(timeSteps, filters, kernel_size, 
                    n_hidden_layers):
    dir_prefix = createDirPrefix(timeSteps, filters, kernel_size, 
                                 n_hidden_layers)
    listDir = os.listdir(os.path.join('cache', dir_prefix))
    listDir = filter(lambda x: '.h5' not in x, listDir)
    print(listDir)
    epochs = [int(dir.split('_')[-1]) for dir in listDir]
    print(epochs)
    max_epochs = max(epochs)
    return max_epochs


# Get train, validation and test data
def get_data(data_type, reservoirIndex, timeSteps, reduceSize=None):
    cache_path = os.path.join('data', str(reservoirIndex), str(timeSteps), 
                              '{}.dat'.format(data_type))
    if not os.path.isfile(cache_path):
        print('Read {} images.'.format(data_type))
        
        data, target = load_data_target(
            os.path.join('data_file', str(reservoirIndex), str(timeSteps), 
                         '{}_data.csv'.format(data_type)), 
            os.path.join('data_file', str(reservoirIndex), str(timeSteps), 
                         '{}_target.csv'.format(data_type)),
            reduceSize=reduceSize)
        
        data = np.array(data, dtype=np.float32)
        data = np.expand_dims(data, axis=-1)
        target = np.array(target, dtype=np.float32)
        target = np.expand_dims(target, axis=-1)
        cache_data((data, target), cache_path)
    
    else:
        print('Restore {} from cache!'.format(data_type))
        (data, target) = restore_data(cache_path)
    
    return (data, target)


# Scale data to [0, 1]
def scaleAsTrain(data, scaler):
    if (data.ndim == 5):
        x = data.reshape(data.shape[0]*data.shape[1]*data.shape[2], 
                         data.shape[3]*data.shape[4])
    else:
        x = data.reshape(data.shape[0]*data.shape[1], 
                         data.shape[2]*data.shape[3])
    scale_x = scaler.transform(x)
    return scale_x.reshape(data.shape)


# Create model
def createModel(filters, kernel_size, input_shape, n_hidden_layers):
    seq = Sequential()
    kernel_size_tuple = (kernel_size, kernel_size)
    seq.add(ConvLSTM2D(filters=filters, kernel_size=kernel_size_tuple,
                       input_shape=input_shape,
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    for _ in range(n_hidden_layers):
        seq.add(ConvLSTM2D(filters=filters, 
                       kernel_size=kernel_size_tuple,
                       padding='same', return_sequences=True))
        seq.add(BatchNormalization())        

    seq.add(ConvLSTM2D(filters=filters, kernel_size=kernel_size_tuple,
                       padding='same', return_sequences=False))
    seq.add(BatchNormalization())

    seq.add(Conv2D(filters=1, kernel_size=kernel_size_tuple,
                   activation='sigmoid',
                   padding='same', data_format='channels_last'))

    return seq


def predictAndVisualize(which, data, target, timeSteps, filters, 
                        kernel_size, n_hidden_layers, epochs, seq):
    import matplotlib.gridspec as gridspec
    plt.figure(figsize=(10, 10))
    G = gridspec.GridSpec(2, timeSteps)
    
    example = data[which, :, :, :,0]
    for i, img in enumerate(example):
        axe = plt.subplot(G[0, i])
        axe.imshow(img)

    target_example = target[which, :, :, 0]
    pred = seq.predict(data[which][np.newaxis, :, :, :, :])
    
    ax_groundtruth = plt.subplot(G[1, :timeSteps//2])
    ax_groundtruth.imshow(target_example)
    ax_groundtruth.set_title('groundtruth')
    
    ax_pred = plt.subplot(G[1, timeSteps//2:2*(timeSteps//2)])
    ax_pred.imshow(pred[0, :, :, 0])
    ax_pred.set_title('predict')

    eval = seq.evaluate(np.expand_dims(data[which], axis=0), 
                        np.expand_dims(target[which], axis=0))
    dir_prefix = createDirPrefix(timeSteps, filters, kernel_size, 
                                 n_hidden_layers, epochs)
    if not os.path.isdir(os.path.join('results', dir_prefix)):
        os.makedirs(os.path.join('results', dir_prefix))

    with open(os.path.join('results', dir_prefix, 
                           'log.txt'), 'a') as w:
        w.write('{},{}'.format(eval[0], eval[1]))
        w.write('\n')

    plt.savefig(os.path.join('results', dir_prefix, 
                             '{}.png'.format(which)))


def predictAndVisualize_RandomCrop(which, reservoir_index, test_data, test_target, model, crop_size):
    timeSteps = test_data.shape[1]
    input_seq = test_data[which]
    ground_truth = test_target[which]
    offset_x = test_data.shape[2] % crop_size
    offset_y = test_data.shape[3] % crop_size
    input_seq = input_seq[:, offset_x//2:-(offset_x - offset_x//2), offset_y//2:-(offset_y - offset_y//2), :]
    predict = np.zeros_like(ground_truth)

    for i in range(input_seq.shape[1] // crop_size):
        for j in range(input_seq.shape[2] // crop_size):
            pred = model.predict(input_seq[np.newaxis, :, i*crop_size:(i+1)*crop_size, j*crop_size:(j+1)*crop_size, :])
            predict[i*crop_size:(i+1)*crop_size, j*crop_size:(j+1)*crop_size, :] = pred[0]

    import matplotlib.gridspec as gridspec
    plt.figure(figsize=(10, 10))
    G = gridspec.GridSpec(2, timeSteps)

    for i, img in enumerate(input_seq[:, :, :, 0]):
        axe = plt.subplot(G[0, i])
        axe.imshow(img)

    ax_groundtruth = plt.subplot(G[1, :timeSteps//2])
    ax_groundtruth.imshow(ground_truth[:, :, 0])
    ax_groundtruth.set_title('groundtruth')

    ax_pred = plt.subplot(G[1, timeSteps//2:2*(timeSteps//2)])
    ax_pred.imshow(predict[:, :, 0])
    ax_pred.set_title('predict') 

    dir_prefix = os.path.join('results', 'random_crop', str(reservoir_index))
    try:
        os.makedirs(dir_prefix)
    except:
        pass
    plt.savefig(os.path.join(dir_prefix, '{}.png'.format(which)))
