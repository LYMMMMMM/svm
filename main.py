from svmfunction import *
from LoadMnist import *
import pickle
import re

# 训练集文件
train_labels_idx1_ubyte = '.\\MnistData\\train-labels.idx1-ubyte'
train_images_idx3_ubyte = '.\\MnistData\\train-images.idx3-ubyte'
t10k_labels_idx1_ubyte = '.\\MnistData\\t10k-labels.idx1-ubyte'
t10k_images_idx3_ubyte = '.\\MnistData\\t10k-images.idx3-ubyte'

# 载入数据
X = load_idx3_ubyte(train_images_idx3_ubyte, 100)
y = load_idx1_ubyte(train_labels_idx1_ubyte, 100)

C = 0.1
sigma = 1.

model = svmTrain_one2n(X, y, C, gaussianKernel(sigma))
y_pre = svmPredict_one2n(model, X)

accuracy = mean((y_pre == y).astype('f4'))
print(accuracy)

# 是否保存模型
flag = input('要保存此次的模型吗？：（Y/N）')
if re.match(r'^Y|y$', flag):
    model_file = './model_save.pkl'
    with open(model_file, 'wb') as file:
        picklestring = pickle.dump(model, file)
elif re.match(r'^N|n$', flag):
    pass
else:
    flag = input('输入错误，请重新输入：（Y/N）')
