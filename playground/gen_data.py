import os
import pandas as pd
import numpy as np

def generate_sample(n):
    X1 = np.random.randn(n, 2)
    Y1 = np.zeros(n)
    X2 = np.random.randn(n, 2) + [10, 5]
    Y2 = np.ones(n)
    X = np.vstack((X1, X2))
    Y = np.hstack((Y1, Y2))

    return X, Y


n = 100
(X_train, Y_train) = generate_sample(int(n/2))
(X_test, Y_test) = generate_sample(int(n/2))
# print(X_train.shape)
# print(Y_train.shape)

# datafile = 'data.csv'

data = pd.DataFrame({'x1': X_train[:, 0], 'x2': X_train[:, 1], 'y': Y_train})
# data.to_csv('X_train.csv', index=False)
# data = pd.DataFrame({'y': Y_train})
data.to_csv('train.csv', index=False)

data = pd.DataFrame({'x1': X_test[:, 0], 'x2': X_test[:, 1], 'y': Y_test})
# data.to_csv('X_test.csv', index=False)
# data = pd.DataFrame({'y': Y_test})
data.to_csv('test.csv', index=False)

