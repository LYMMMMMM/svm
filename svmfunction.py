from numpy import *
import matplotlib.pyplot as plt


class Model:
    def __init__(self, X, y, b, alphas, w, kernel_func):
        self.X = X
        self.y = y
        self.b = b
        self.alphas = alphas
        self.w = w

        def kernel_function(self, x1, x2):
            return kernel_func(x1, x2)

        Model.kernel_function = kernel_function
        Model.kernel_function.__name__ = kernel_func.__name__


def linearKernel(x1, x2):  # 线性核函数
    x1 = x1.flatten()
    x2 = x2.flatten()
    return sum(x1 * x2)


def gaussianKernel(sigma):  # 高斯核,返回一个计算高斯核的函数
    def gaussianKernel(x1, x2):
        if isinstance(x1, int):
            x1 = array(x1)
            x2 = array(x2)
        sim = exp(-1.0 / (2 * sigma ** 2) * sum((x1 - x2) ** 2))
        return sim

    return gaussianKernel


def svmTrain(X, y, C, kernel_function, tol=1e-3, max_passes=5, screen=True):
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
    if kernel_function.__name__ == 'linearKernel':
        K = dot(X, X.T)
    elif kernel_function.__name__ == 'gaussianKernel':
        X2 = array([sum(X ** 2, 1)])  # 每个特征向量的模平方m*1
        K = -2 * X.dot(X.T) + X2 + X2.T
        K = kernel_function(1, 0) ** K
    else:
        K = zeros([m, m])
        for i in range(0, m):
            for j in range(0, m):
                K[i, j] = kernel_function(X[i, :], X[j, :])
                K[j, i] = K[i, j]

    # 训练模型
    if screen:
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

        if screen:
            print('.', end='')
            dots = dots + 1
            if dots > 78:
                dots = 0
                print('\n', end='')
    if screen:
        print('Done!')

    idx = nonzero(alphas > 0)[0]  # 或者可以用 idx = (alphas>0).T[0]
    X = X[idx]  # m'*n
    y = y[idx]
    alphas = alphas[idx]
    w = dot((alphas * y).T, X).T  # 2*1
    model = Model(X, y, b, alphas, w, kernel_function)

    return model


def svmTrain_one2n(X, y, C, kernel_function, tol=1e-3, max_passes=5, screen=True):
    model_list = []
    for i in range(10):
        y_i = (y == i).astype(int)
        model_list.append(svmTrain(X, y_i, C, kernel_function, tol, max_passes, screen))
    return model_list


def svmPredict(model, X):
    m = X.shape[0]
    pred = zeros((m, 1))
    p = zeros((m, 1))

    if model.kernel_function.__name__ == 'linearKernel':
        p = dot(X, model.w) + model.b
    elif model.kernel_function.__name__ == 'gaussianKernel':
        X1 = array([sum(X ** 2, axis=1)]).T  # m*1
        X2 = array([sum(model.X ** 2, axis=1)])  # 1*m'
        K = X2 - 2 * dot(X, model.X.T) + X1  # X(m*n), model.X(m'*n), K(m*m')
        K = (model.kernel_function(1, 0)) ** K
        K = (model.y.T * K) * model.alphas.T
        p = array([sum(K, axis=1)]).T  # m*1
    else:
        for i in range(m):
            prediction = 0
            for j in range(model.X.shape[0]):
                prediction = prediction + \
                             model.alphas[j, 0] * model.y[j, 0] * \
                             model.kernel_function(X[i], model.X[j])
            p[i] = prediction + model.b

    # Convert predictions into 0 / 1
    pred[p >= 0] = 1
    pred[p < 0] = 0
    return pred  # m*1


def svmPredict_one2n(model, X):
    m = X.shape[0]
    p = zeros((m, 10))

    for k in range(10):
        if model[k].kernel_function.__name__ == 'linearKernel':
            p[:, [k]] = dot(X, model[k].w) + model[k].b
        elif model[k].kernel_function.__name__ == 'gaussianKernel':
            X1 = array([sum(X ** 2, axis=1)]).T  # m*1
            X2 = array([sum(model[k].X ** 2, axis=1)])  # 1*m'
            K = X2 - 2 * dot(X, model[k].X.T) + X1  # X(m*n), model.X(m'*n), K(m*m')
            K = (model[k].kernel_function(1, 0)) ** K
            K = (model[k].y.T * K) * model[k].alphas.T
            p[:, [k]] = array([sum(K, axis=1)]).T  # m*1
        else:
            for i in range(m):
                prediction = 0
                for j in range(model[k].X.shape[0]):
                    prediction = prediction + \
                                 model[k].alphas[j, 0] * model[k].y[j, 0] * \
                                 model[k].kernel_function(X[i], model[k].X[j])
                p[i, [k]] = prediction + model[k].b
    pred = array([argmax(p, axis=1)]).T
    return pred  # m*1

# 针对高斯核函数 测试不同的参数C，sigma,
def optimizeParams(X, y, Xval, yval):
    C = 1
    sigma = 0.3
    error = 1
    # count = 0.
    # print('Training...%s' % str(0) + '%', end='\r')
    for C_temp in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
        for sigma_temp in [0.1, 0.3, 1, 3, 10, 30]:
            model = svmTrain(X, y, C_temp, gaussianKernel(sigma_temp), screen=False)
            predictions = svmPredict(model, Xval)

            error_temp = float(mean(double(predictions != yval)))
            if error_temp < error:
                C = C_temp
                sigma = sigma_temp
                error = error_temp
            print('c:%0.3f ; sigma:%0.3f  ; error:%0.4f ; minimum error:%0.4f'
                  % (C_temp, sigma_temp, error_temp, error))
            # 显示百分制表示的进度
            # count += 1
            # percent = round(count / 48. * 100, 2)
            # print('Training...%s' % str(percent) + '%  ', end='\r')
    print('\nDone!')
    return C, sigma


def optimizeParams_one2n(X, y, Xval, yval):
    C = 1
    sigma = 0.3
    error = 1
    # count = 0.
    # print('Training...%s' % str(0) + '%', end='\r')
    for C_temp in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
        for sigma_temp in [0.1, 0.3, 1, 3, 10, 30]:
            model = svmTrain_one2n(X, y, C_temp, gaussianKernel(sigma_temp), screen=False)
            predictions = svmPredict_one2n(model, Xval)

            error_temp = float(mean(double(predictions != yval)))
            if error_temp < error:
                C = C_temp
                sigma = sigma_temp
                error = error_temp
            print('c:%0.3f ; sigma:%0.3f  ; error:%0.4f ; minimum error:%0.4f'
                  % (C_temp, sigma_temp, error_temp, error))
            # 显示百分制表示的进度
            # count += 1
            # percent = round(count / 48. * 100, 2)
            # print('Training...%s' % str(percent) + '%  ', end='\r')
    print('\nDone!')
    return C, sigma


# 可视化决策边界
def visualizeBoundaryLinear(X, y, model):
    w = model.w
    b = model.b
    xp = linspace(min(X[:, 0]), max(X[:, 0]), 100)
    yp = -(w[0, 0] * xp + b) / w[1, 0]
    plt.plot(xp, yp, '-b')
    plt.show()


def visualizeBoundary(X, y, model):
    x1plot = linspace(min(X[:, 0]), max(X[:, 0]), 100)
    x2plot = linspace(min(X[:, 1]), max(X[:, 1]), 100)
    X1, X2 = meshgrid(x1plot, x2plot)
    vals = zeros(X1.shape)
    for i in range(X1.shape[1]):
        this_x = hstack([X1[:, [i]], X2[:, [i]]])
        vals[:, [i]] = svmPredict(model, this_x)  # 一维

    C = plt.contour(X1, X2, vals, 1, colors='black', linewidths=0.5)
    plt.show()
