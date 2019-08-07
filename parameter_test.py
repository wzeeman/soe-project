"""Play with the parameters of normSOEFastG.""" 

from SOE_main_test import read_scatter_data
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

import pyximport
pyximport.install(setup_args={
    "include_dirs":np.get_include()}, reload_support=True)
import SOE_functions_swp as utils

def plot_recon(ax, data):
    """Plot reconstructed image.
    
    Parameters
    ----------
    ax : matplotlib.Axes
        The axes to draw to

    data : np.ndarray
        ndarray output from utils.normSOEFastG() to plot

    Returns
    -------
    out : list
        list of artists
    """
    ax.imshow(data)

def main():
    print ('--- Running parameter_test.py ---')

    # Read in csv and returns numpy array of double scatters.
    data_dir = Path('190516/')
    data_files = list(data_dir.glob('*.csv'))
    allDoubles = read_scatter_data(data_dir.name+'/', data_files[1].name)

    # Set recon parameters.
    nX = 50;  nZ = 50 # pixel size in the X and Z direction of the
    # reconstruction. Note: it's assumed the pixel pitch is 1 mm.

    sparseMat = utils.genSparseMat(allDoubles, nX, nZ, 100000)

    for beta in range(1, 1001):
        rcon = utils.normSOEFastG(sparseMat, nX, nZ, 10, 100, beta)
        fig, ax = plt.subplots(1, 1)
        plot_recon(ax, rcon)
        plt.savefig('rcon%d' % beta, bbox_inches='tight')

if __name__ == "__main__":

    main()
