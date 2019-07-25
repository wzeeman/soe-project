#!/usr/bin/python

import numpy as np
import time
import sys, os
import matplotlib.pyplot as plt
from pathlib import Path

import pyximport
pyximport.install(setup_args={
    "include_dirs":np.get_include()}, reload_support=True)
import SOE_functions_swp as utils

#------------------------------------------------------------------
# FUNCTION/CLASS DEFINITIONS
#------------------------------------------------------------------

# Read in scatter data from a csv file and convert into numpy array
def read_scatter_data(data_dir, data_filename):
    '''
    Takes in an CSV file (format: edep1,x1,y1,z1,edep2,x2,y2,z2) and reads the
    information into a python numpy array.

    @params:
        data_dir            - Required : Path to 2x scatter data .csv file (Str)
        data_filename       - Required : Name of 2x scatter data .csv file to read data from (Str)

    @returns:
        allScatters_data    - numpy array of the 2x scatter CSV data
    '''

    print('Reading in data from', data_filename + '.', '\nThis may take several minutes...')

    allScatters_data = np.genfromtxt(data_dir + data_filename, delimiter = ',')
    print('{:30} {:d}'.format('Total number of events:', len(allScatters_data)))
    #print ('  -- allScatters_data[0]: {}'.format(allScatters_data[0]))

    return allScatters_data



#------------------------------------------------------------------
# MAIN PROGRAM
#------------------------------------------------------------------

def main():
    print ('\n--- Running SOE_main_test.py ---')

    # reads in csv and returns numpy array of double scatters
    data_dir = Path('190516/')
    data_files = list(data_dir.glob('*.csv'))
    allDoubles = read_scatter_data(data_dir.name+'/', data_files[1].name)

    # set recon parameters
    nX = 50;  nZ = 50  # pixel size in the X and Z direction of the reconstruction / note, it's assumed the pixel pitch is 1 mm

    # generates the sparse matrix for use in the SOE routines; returns a csr matrix
    #  parameters: genSparseMat(allDoubles, nX, nZ, totalPts, numBlocks)
    #   - nX, nZ: pixel size in the X and Z direction of the reconstruction / note, it's assumed the pixel pitch is 1 mm
    #   - totalPts: how many datapoints from allDoubles to use
    #   - numBlocks: number of blocks of data used when computing the sparse matrix; this was tested to reduce memory footprint during backprojection
    sparseMat = utils.genSparseMat(allDoubles, nX, nZ, 100000, 50)

    print('Shape of sparse matrix: {}'.format(sparseMat.shape))
    #print ('  -- sparseMat[0]: {}'.format(sparseMat[0]))
    #print ('  -- sparseMat: {}'.format(sparseMat))
    #print ('  -- sparseMat.shape: {}'.format(sparseMat.shape))

    # performs a simplistic SOE recontruction on the sparse matrix
    #  parameters: normSOEFast(sparseMat, nX, nZ, loopTot, itrsWR)
    #   - nX, nZ: pixel size in the X and Z direction of the reconstruction / note, it's assumed the pixel pitch is 1 mm
    #   - loopTot: total number of SOE loops to perform; each loop is over all data points
    #   - itrsWR: ITeRationS Worth of Rands; number of random numbers generated at a time to save memory space
    rcon = utils.normSOEFast(sparseMat, nX, nZ, 10, 100)
    #print ('  -- rcon: {}'.format(rcon))

    # view final image
    plt.imshow(rcon)
    plt.show()


if __name__ == "__main__":

    main()
