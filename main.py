from svmfunction import *
from operateData import *
from numpy import *

# 载入数据
X, y = load_data('./ex6data1.mat')
plot_data(X, y)

C = 1.
sigma = 0.1
model = svmTrain(X, y, C, gaussianKernel(sigma))



# 训练线性核


