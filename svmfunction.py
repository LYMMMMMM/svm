from numpy import *
import matplotlib.pyplot as plt


def linearKernel(x1, x2):  # 线性核函数
    x1 = x1.flatten()
    x2 = x2.flatten()
    return dot(x1, x2)


def gaussianKernel(sigma):  # 高斯核,返回一个计算高斯核的函数
    def gaussianKernel(x1, x2):
        if isinstance(x1, int):
            x1 = array(x1)
            x2 = array(x2)
        sim = exp(-1.0 / (2 * sigma ** 2) * sum((x1 - x2) ** 2))
        return sim

    return gaussianKernel


def svmTrain(X, y, C, kernel_funtion, tol=1e-3, max_passes=5):
    # 训练集大小m以及特征向量大小n
    m = X.shape[0]
    n = X.shape[1]

    # 将y中的0变成-1
    y = y.astype(int)
    y[y == 0] = -1

    # 变量
    alphas = zeros([m, 1])  # m*1
    b = 0.
    E = zeros([m, 1])
    passes = 0
    eta = 0.
    L = 0.

    # 计算特征K矩阵  m*m
    if kernel_funtion.__name__ == 'linearKernel':
        K = dot(X, X.T)
    elif kernel_funtion.__name__ == 'gaussianKernel':
        X2 = array([sum(X ** 2, 1)])  # 每个特征向量的模平方m*1
        K = -2 * X.dot(X.T) + X2 + X2.T
        K = kernel_funtion(1, 0) ** K
    else:
        K = zeros([m, m])
        for i in range(0, m):
            for j in range(0, m):
                K[i, j] = kernel_funtion(X[i, :], X[j, :])
                K[j, i] = K[i, j]

    count = 0
    # 训练模型
    print('\nTraining ...', end='')
    dots = 12
    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(0, m):
            E[i, 0] = b + sum(alphas * y * K[:, [i]]) - y[i, 0]

            if (y[i, 0] * E[i, 0] < -tol and alphas[i, 0] < C) or \
                    (y[i, 0] * E[i, 0] > tol and alphas[i, 0] > 0):

                j = int(floor(m * random.rand()))
                while j == i:
                    j = int(floor(m * random.rand()))

                E[j, 0] = b + sum(alphas * y * K[:, [j]]) - y[j, 0]

                alpha_i_old = alphas[i, 0]
                alpha_j_old = alphas[j, 0]

                if y[i, 0] == y[j, 0]:
                    L = max(0, alphas[j, 0] + alphas[i, 0] - C)
                    H = min(C, alphas[j, 0] + alphas[i, 0])
                else:
                    L = max(0, alphas[j, 0] - alphas[i, 0])
                    H = min(C, C + alphas[j, 0] - alphas[i, 0])

                if L == H:
                    continue

                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                alphas[j, 0] = alphas[j, 0] - (y[j, 0] * (E[i, 0] - E[j, 0])) / eta

                alphas[j, 0] = min(H, alphas[j, 0])
                alphas[j, 0] = max(L, alphas[j, 0])

                if abs(alphas[j, 0] - alpha_j_old) < tol:
                    alphas[j, 0] = alpha_j_old
                    continue

                alphas[i, 0] = alphas[i, 0] + y[i, 0] * y[j, 0] * (alpha_j_old - alphas[j, 0])

                b1 = b - E[i, 0] \
                     - y[i, 0] * (alphas[i, 0] - alpha_i_old) * K[i, j] \
                     - y[j, 0] * (alphas[j, 0] - alpha_j_old) * K[i, j]
                b2 = b - E[j, 0] \
                     - y[i, 0] * (alphas[i, 0] - alpha_i_old) * K[i, j] \
                     - y[j, 0] * (alphas[j, 0] - alpha_j_old) * K[j, j]

                if 0 < alphas[i, 0] < C:
                    b = b1
                elif 0 < alphas[j, 0] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.

                num_changed_alphas += 1

        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0

        print('.', end='')
        dots = dots + 1
        if dots > 78:
            dots = 0
            print('\n', end='')
    print('Done!\n')

    idx = nonzero(alphas > 0)[0]  # 或者可以用 idx = (alphas>0).T[0]
    dt = dtype([('X', ndarray), ('y', ndarray), ('b', 'f4'),
                ('alphas', ndarray), ('w', ndarray)])
    X = X[idx]
    y = y[idx]
    alphas = alphas[idx]
    w = dot((alphas * y).T, X).T
    model = array([(X, y, b, alphas, w)], dtype=dt)

    return model


# 可视化决策边界
def visualizeBoundaryLinear(X, y, model):
    w = model[0]['w']
    b = model[0]['b']
    xp = linspace(min(X[:, 0]), max(X[:, 0]), 100)
    yp = -(w[0, 0] * xp + b) / w[1, 0]
    plt.plot(xp, yp, '-b')
    plt.show()
