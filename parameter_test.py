"""Play with the parameters of normSOEFastG.""" 

from SOE_main_test import read_scatter_data
import matplotlib.pyplot as plt
from pathlib import Path

import pyximport
pyximport.install(setup_args={
    "include_dirs":np.get_include()}, reload_support=True)
import SOE_functions_swp as utils

def main():
    print ('--- Running parameter_test.py ---')

    # Read in csv and returns numpy array of double scatters.
    data_dir = Path('190516/')
    data_files = list(data_dir.glob('*.csv'))
    allDoubles = read_scatter_data(data_dir.name+'/', data_files[1].name)

    # Set recon parameters.
    nX = 50;  nZ = 50 # pixel size in the X and Z direction of the
    # reconstructionÄnote: it's assumed the pixel pitch is 1 mm.

    sparseMat = utils.genSparseMat(allDoubles, nX, nZ, 100000)

    rcon = utils.normSOEFastG(sparseMat, nX, nZ, 10, 100)

    # view final image
    plt.imshow(rcon)
    plt.show()

if __name__ == "__main__":

    main()
