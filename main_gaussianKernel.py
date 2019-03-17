from svmfunction import *
from LoadMnist import *
import pickle
import re
"""
训练高斯核模型
"""

# 训练集文件
train_labels_idx1_ubyte = '.\\MnistData\\train-labels.idx1-ubyte'
train_images_idx3_ubyte = '.\\MnistData\\train-images.idx3-ubyte'
t10k_labels_idx1_ubyte = '.\\MnistData\\t10k-labels.idx1-ubyte'
t10k_images_idx3_ubyte = '.\\MnistData\\t10k-images.idx3-ubyte'

# 载入数据
X = load_idx3_ubyte(train_images_idx3_ubyte, 5000)
y = load_idx1_ubyte(train_labels_idx1_ubyte, 5000)

# 训练高斯核模型
C = 0.02
sigma = 1.
model = svmTrain_one2n(X, y, C, gaussianKernel(sigma))
y_pre = svmPredict_one2n(model, X)

# 计算模型在训练集上的准确度
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
