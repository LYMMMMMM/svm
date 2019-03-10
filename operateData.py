import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


def load_data(s='./ex6data1.mat'):
    data = sio.loadmat(s)
    X = data['X']
    y = data['y']
    y = y.astype(int)
    return X, y


def plot_data(X, y):
    pos = np.nonzero(y)
    neg = np.nonzero(y == 0)
    plt.plot(X[pos], X[pos[0], 1], 'xk')
    plt.plot(X[neg], X[neg[0], 1], 'ok', markerfacecolor='y')
    # plt.show()
