#import subprocess
#
# for i in range(2, 50):
#     print('Reservoir', i)
#     subprocess.call(['python', 'augment_one_reservoir.py', '-i', str(i)])
#     print()

import os
import sys
import argparse
import numpy as np
from ModisUtils.misc import *

def argParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type=int, 
                        help='train - 0 | val - 1 | test - 2')
    parser.add_argument('-i', '--iter', type=int,
                        help='time of processing, 0 | 1 | 2')
    return parser.parse_args()


timeSteps = 8
list_data_type = ['train', 'val', 'test']
parser = argParse()
type_data = list_data_type[parser.type]

cache_path_prefix = os.path.join('data_augment_{}'.format(parser.iter),  'timeSteps_{}'.format(timeSteps))
if not os.path.isdir(cache_path_prefix):
    os.makedirs(cache_path_prefix)

cache_path = {}
data = {}
cache_path[type_data] = os.path.join(cache_path_prefix, type_data + '.dat')
data[type_data] = ([], [])

print('type =', type_data, '-- iter =', parser.iter)
if not os.path.isfile(cache_path[type_data]):
    list_dir = sorted(os.listdir('data_augment'), key = lambda x: int(x))
    
    n = len(list_dir)
    i = parser.iter
    if i < 5:
        list_dir_use = list_dir[i*(n//6): (i + 1)*(n//6)]
    else:
        list_dir_use = list_dir[i*(n//6):]
    print(list_dir_use)

    for reservoirIndex in list_dir_use:
        reservoirPath = os.path.join('data_augment', reservoirIndex, str(timeSteps), type_data + '.dat')
        reservoirData = restore_data(reservoirPath)
        print(reservoirIndex, reservoirData[0].shape, reservoirData[1].shape)
        data[type_data][0].append(reservoirData[0])
        data[type_data][1].append(reservoirData[1])
    _cache_data = (np.vstack(data[type_data][0]), np.vstack(data[type_data][1]))
    cache_data(_cache_data, cache_path[type_data])
