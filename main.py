#!/usr/bin/python

import numpy as np
import pandas as pd

from Utils import load_data
from MatrixFactorization import DNNMF

if __name__ == '__main__':
    # input_m = load_data('./playtimes.csv',
    #                     leading_row=1,
    #                     leading_column=1)
    input_m = np.array([[5, 4, 2, 0, 0, 1, 2, 0, 0, 3],
                        [0, 3, 1, 2, 3, 3, 2, 0, 5, 1],
                        [1, 3, 1, 0, 3, 0, 2, 4, 5, 2],
                        [5, 5, 1, 3, 1, 2, 0, 2, 0, 0],
                        [2, 2, 2, 4, 0, 0, 1, 0, 5, 2]],
                       dtype=np.float32)
    #input_m = np.array([[0, 0, 0, 4, 2, 0, 0, 0, 0, 0],
    #                    [4, 0, 0, 0, 4, 0, 0, 0, 0, 5],
    #                    [0, 3, 0, 0, 0, 0, 5, 0, 0, 0],
    #                    [0, 0, 1, 0, 0, 2, 0, 0, 0, 0],
    #                    [0, 0, 0, 0, 0, 0, 0, 3, 0, 4],
    #                    [3, 2, 0, 0, 0, 0, 0, 0, 5, 0]],
    #                   dtype=np.float32)
    shape = input_m.shape
    print 'input width: ', shape[0]
    print 'input height: ', shape[1]

    input_df = pd.DataFrame(input_m)

    mf = DNNMF(input_df,
               2,
               rank=3,
               lr=0.01,
               steps=50000)

    W, H, bw, bh, WH = mf.train()

    print '=' * 50
    print 'W: \n', W
    print '=' * 50
    print 'H: \n', H
    print '=' * 50
    print 'bw: \n', bw
    print '=' * 50
    print 'bh: \n', bh

    prediction = WH.round()
    print '=' * 50
    print 'prediction: \n', prediction
    print '=' * 50
    print 'original: \n', input_m
