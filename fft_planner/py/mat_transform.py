import scipy.io
import numpy as np
import sys
import os

if __name__=="__main__":

    file = sys.argv[1]
    dir_name, file_name = os.path.split(file)
    name, ext = os.path.splitext(file_name)
    mat = scipy.io.loadmat(file)['HD']
    np.savetxt(name + '.csv', mat, delimiter=',')
    