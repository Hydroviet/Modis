import sys
import numpy as np
import os
import pickle
# Data
def cache_data(data, path):
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

data = [[], []]
path = [sys.argv[1], sys.argv[2]]

for i in range(2):
    restoreData = restore_data(path[i])
    data[0].append(restoreData[0])
    data[1].append(restoreData[1])
data = (np.vstack(data[0]), np.vstack(data[1]))

if (len(sys.argv) == 3):
    cache_path = 'data_augment_all/tmp.dat'
else:
    cache_path = sys.argv[3]
    
cache_data(data, cache_path)
  

