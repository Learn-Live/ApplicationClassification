#!/usr/bin/python
from __future__ import division

import os
import sys
import socket
import copy
import numpy
from datetime import *
import time
from numpy import random
import time

if __name__ == "__main__":

    script = sys.argv

    randArray = random.random(size=(5, 5))

    print(randArray)

    max_value = 0
    max_row_index = None
    max_col_index = None
    for row_index in range(0, randArray.shape[0] - 1):
        for col_index in range(row_index + 1, randArray.shape[0]):
            if randArray[row_index][col_index] > max_value:
                max_value = randArray[row_index][col_index]
                max_row_index = row_index
                max_col_index = col_index

    print(max_row_index, max_col_index, randArray[max_row_index][max_col_index])

    mask = numpy.ones([randArray.shape[0] + 1, randArray.shape[1] + 1], dtype=bool)
    il1 = numpy.tril_indices(randArray.shape[0] + 1)
    mask[il1] = False

    new_mask = numpy.delete(mask, mask.shape[0] - 1, 0)
    new_mask = numpy.delete(new_mask, mask.shape[1] - 1, 1)
    mask = new_mask

    result = numpy.argwhere(randArray == randArray[mask].max())
    print(result[0][0], result[0][1], randArray[mask].max())

    import numpy as np

    a = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # diag = np.diagonal(a, 3)
    #
    # # It's not writable. MAKE it writable.
    # diag.setflags(write=True)
    #
    # diag.fill
    np.fill_diagonal(a, 5)
    print(a)

    import numpy as np

    N = 5
    b = np.random.random_integers(-2000, 2000, size=(N, N))
    b_symm = (b + b.T) / 2
    np.fill_diagonal(b_symm, 0)
    print(b_symm)

    tri_upper_no_diag = np.triu(a, k=1)
    print(tri_upper_no_diag)
