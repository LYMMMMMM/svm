from ex6.svmfunction import *
from ex6.operateData import *

# 载入数据
X, y, Xval, yval = load_data3('./ex6/ex6data3.mat')
plot_data(X, y)

C, sigma = optimizeParams(X, y, Xval, yval)

model = svmTrain(X, y, C, gaussianKernel(sigma), screen=False)
visualizeBoundary(X, y, model)
