import numpy as np
import rasterio as rio
from .generators.generators import SimpleImageGenerator
from .misc import *

def mergeDataAndTarget(data, target):
    target_1 = target.reshape(target.shape[0], 1, target.shape[1], 
                              target.shape[2], target.shape[3])
    data_1 = np.concatenate([data, target_1], axis=1)
    data_1 = data_1.squeeze(axis=-1)
    return data_1


def splitDataAndTarget(dataMergeTarget):
    data = np.expand_dims(dataMergeTarget[:, :-1, :, :], axis=-1)
    target = dataMergeTarget[:, -1, :, :]
    target = np.expand_dims(target, axis=-1)
    return (data, target)


def generate(dataMergeTarget, datagen):
    data_iterator = datagen.flow_from_list(x=dataMergeTarget, nframes=dataMergeTarget.shape[1])
    
    res = []
    nSamples = 100
    while nSamples > 0:
        #print(nSamples)
        res1 = data_iterator._get_batches_of_transformed_samples(np.arange(dataMergeTarget.shape[0]))
        res.append(res1)
        nSamples -= 1
    return np.vstack(res)


def checkImageSize(imgPath, crop_size):
    with rio.open(imgPath) as src:
        img = src.read(1)
        return (img.shape[0] >= 2*crop_size and img.shape[1] >= 2*crop_size)
    return False


def augmentationOneReservoir(reservoirIndex, dataDir='MOD13Q1', 
                             bandsUse=['NIR'], timeSteps=7,
                             crop_size=20, random_crop=True):
    # Create data file if not created yet
    listTestFiles = createFileData(dataDir=dataDir, reservoirsUse=[reservoirIndex], 
                                   bandsUse=bandsUse, timeSteps=timeSteps)
    if not checkImageSize(listTestFiles[0], crop_size):
        return False

    # Load data
    reduceSize = None
    #reduceSize = (40,40)
    if not os.path.isdir(os.path.join('data', str(reservoirIndex), str(timeSteps))):
        os.makedirs(os.path.join('data', str(reservoirIndex), str(timeSteps)))
    (train_data, train_target) = get_data('train', reservoirIndex, timeSteps, 
                                          reduceSize=reduceSize)
    (val_data, val_target) = get_data('val', reservoirIndex, timeSteps, 
                                      reduceSize=reduceSize)
    (test_data, test_target) = get_data('test', reservoirIndex, timeSteps, 
                                        reduceSize=reduceSize)

    train_merged = mergeDataAndTarget(train_data, train_target)
    val_merged = mergeDataAndTarget(val_data, val_target)
    test_merged = mergeDataAndTarget(test_data, test_target)
    
    datagen = SimpleImageGenerator(crop_size=(crop_size,crop_size), random_crop=random_crop)
    train = generate(train_merged, datagen)
    val = generate(val_merged, datagen)
    test = generate(test_merged, datagen)

    train_data_augment, train_target_augment = splitDataAndTarget(train)
    val_data_augment, val_target_augment = splitDataAndTarget(val)
    test_data_augment, test_target_augment = splitDataAndTarget(test)

    data_augment_path_prefix = os.path.join('data_augment', str(reservoirIndex), str(timeSteps))
    if not os.path.isdir(data_augment_path_prefix):
        os.makedirs(data_augment_path_prefix)
    train_path = os.path.join(data_augment_path_prefix, 'train.dat')
    val_path = os.path.join(data_augment_path_prefix, 'val.dat')
    test_path = os.path.join(data_augment_path_prefix, 'test.dat')

    cache_data((train_data_augment, train_target_augment), train_path)
    cache_data((val_data_augment, val_target_augment), val_path)
    cache_data((test_data_augment, test_target_augment), test_path)

    return True
