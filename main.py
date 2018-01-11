#!/usr/bin/python

import numpy as np
import pandas as pd

from Utils import load_data
from MatrixFactorization import MatrixFactorization

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
    # input_m = np.divide(input_m, input_m.max())
    shape = input_m.shape
    print 'input width: ', shape[0]
    print 'input height: ', shape[1]

    input_df = pd.DataFrame(input_m)

    mf = MatrixFactorization(input_df,
                             rank=2,
                             lr=0.001,
                             steps=100000)

    W, H, b = mf.train(enable_b=True)

    print '=' * 50
    print 'W: \n', W
    print '=' * 50
    print 'H: \n', H
    print '=' * 50
    print 'b: \n', b

    prediction = np.add(np.dot(W, H), b).round()
    # prediction = np.dot(W, H).round()
    print '=' * 50
    print 'prediction: \n', prediction
    print '=' * 50
    print 'original: \n', input_m
