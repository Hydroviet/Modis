import argparse
from ModisUtils.modis_generator import augmentationOneReservoir


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--index', type=int, 
                    help='reservoir index', default=0)
parser.add_argument('-t', '--timeSteps', type=int, 
                    help='timeSteps', default=8)
parser = parser.parse_args()

augmentationOneReservoir(reservoirIndex=parser.index, timeSteps=parser.timeSteps)