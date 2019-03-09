from svmfunction import *
from operateData import *

# 载入数据
# X, y = load_data('./ex6data1.mat')
# plot_data(X, y)

# 训练线性核
from svmfunction import gaussianKernel
g = gaussianKernel(1)
print(type(g))
print(g.__name__)