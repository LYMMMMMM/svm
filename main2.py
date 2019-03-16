from svmfunction import *
from LoadMnist import *
import pickle

# 训练集文件
train_labels_idx1_ubyte = '.\\MnistData\\train-labels.idx1-ubyte'
train_images_idx3_ubyte = '.\\MnistData\\train-images.idx3-ubyte'
t10k_labels_idx1_ubyte = '.\\MnistData\\t10k-labels.idx1-ubyte'
t10k_images_idx3_ubyte = '.\\MnistData\\t10k-images.idx3-ubyte'

X = load_idx3_ubyte(t10k_images_idx3_ubyte, 100)
y = load_idx1_ubyte(t10k_labels_idx1_ubyte, 100)
# 载入数据
Xval = load_idx3_ubyte2(train_images_idx3_ubyte, 1000, 1100)
yval = load_idx1_ubyte2(train_labels_idx1_ubyte, 1000, 1100)

# C,sigma = optimizeParams_one2n(X,y,Xval,yval)

C = 0.1
sigma = 1
model = svmTrain_one2n(X, y, C, gaussianKernel(sigma))
y_pre = svmPredict_one2n(model, X)

accuracy = mean((y_pre == y).astype('f4'))
print(accuracy)