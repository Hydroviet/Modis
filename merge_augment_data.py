import subprocess

# for time in range(6):
#     for data_type in range(1):
#         subprocess.call(['python', 'augment_data.py', '--type', str(data_type), '--iter', str(time)])

import os
import numpy as np
from ModisUtils.misc import *

timeSteps = 8
data = [[], []]
for i in range(2):
    path = os.path.join('data_augment_{}'.format(i), 'timeSteps_{}'.format(timeSteps), 'val.dat')
    restoreData = restore_data(path)
    data[0].append(restoreData[0])
    data[1].append(restoreData[1])
data = (np.vstack(data[0]), np.vstack(data[1]))

if not os.path.isdir('data_augment_all'):
    os.makedirs('data_augment_all')
cache_data(data, 'data_augment_all/val_0.dat')
