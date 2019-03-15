from ex6.svmfunction import *
from ex6.operateData import *

# 载入数据
X, y = load_data('./ex6data1.mat')
plot_data(X, y)

C = 1.
sigma = 0.1
model = svmTrain(X, y, C, gaussianKernel(sigma))
visualizeBoundary(X, y, model)

# 训练线性核
