from svmfunction import *
from LoadMnist import *
import pickle

"""
最优化高斯核模型参数sigma, C
"""

# 训练集文件
train_labels_idx1_ubyte = '.\\MnistData\\train-labels.idx1-ubyte'
train_images_idx3_ubyte = '.\\MnistData\\train-images.idx3-ubyte'
t10k_labels_idx1_ubyte = '.\\MnistData\\t10k-labels.idx1-ubyte'
t10k_images_idx3_ubyte = '.\\MnistData\\t10k-images.idx3-ubyte'

X = load_idx3_ubyte(t10k_images_idx3_ubyte, 10000)
y = load_idx1_ubyte(t10k_labels_idx1_ubyte, 10000)
# 载入数据
Xval = load_idx3_ubyte2(train_images_idx3_ubyte, 10000, 20000)
yval = load_idx1_ubyte2(train_labels_idx1_ubyte, 10000, 20000)

C, sigma = optimizeParams_one2n(X, y, Xval, yval)
print(C)
print(sigma)

