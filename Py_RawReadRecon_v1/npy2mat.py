# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 10:43:55 2021

@author: yongquan.ye
"""

from scipy.io import savemat
import os
import numpy as np


for (dirpath, dirnames, filenames) in os.walk('C:\\DataTemp\\MTP\\5T\\20210610_T1map_test'):
    for fn in filenames:
        if fn.endswith('.npy'):
            print(fn)
            npy_fn = dirpath + '\\' + fn
            raw = np.load(npy_fn)
            mat_fn = dirpath + '\\' + fn[:-4] + '.mat'
            savemat(mat_fn,raw)
