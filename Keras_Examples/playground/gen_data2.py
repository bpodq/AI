# 产生playground2的数据
# 里层是0，外层是1

import pandas as pd
import numpy as np


def generate_sample(m):
    n = int(m/2)
    rho = np.random.uniform(0, 5, (n, 1))
    theta = np.random.uniform(0, 2*np.pi, (n, 1))
    X1 = np.hstack((rho*np.cos(theta), rho*np.sin(theta)))
    Y1 = np.zeros((n, 1))

    rho = np.random.uniform(5, 10, (n, 1))
    theta = np.random.uniform(0, 2*np.pi, (n, 1))
    X2 = np.hstack((rho*np.cos(theta), rho*np.sin(theta)))
    Y2 = np.ones((n, 1))

    X = np.vstack((X1, X2))
    Y = np.vstack((Y1, Y2))

    return X, Y


n = 100
(X_train, Y_train) = generate_sample(n)
(X_test, Y_test) = generate_sample(n)

data = pd.DataFrame({'x1': X_train[:, 0], 'x2': X_train[:, 1], 'y': Y_train[:, 0]})
data.to_csv('train2.csv', index=False)

data = pd.DataFrame({'x1': X_test[:, 0], 'x2': X_test[:, 1], 'y': Y_test[:, 0]})
data.to_csv('test2.csv', index=False)

