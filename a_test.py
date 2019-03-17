from svmfunction import *
from LoadMnist import *
import pickle

"""
用训练好的模型进行预测
可以一个一个查看预测结果
"""
# 训练集文件
train_labels_idx1_ubyte = '.\\MnistData\\train-labels.idx1-ubyte'
train_images_idx3_ubyte = '.\\MnistData\\train-images.idx3-ubyte'
t10k_labels_idx1_ubyte = '.\\MnistData\\t10k-labels.idx1-ubyte'
t10k_images_idx3_ubyte = '.\\MnistData\\t10k-images.idx3-ubyte'

# 载入测试集数据
X = load_idx3_ubyte(t10k_images_idx3_ubyte, 2000)
y = load_idx1_ubyte(t10k_labels_idx1_ubyte, 2000)

# 加载模型可选高斯或线性
sigma = 1.
# model_file = './model_save.pkl'
# mmm = Model(0, 0, 0, 0, 0, gaussianKernel(sigma))  # 不加这句话则加载的类对象没有成员函数
mmm = Model(0, 0, 0, 0, 0, linearKernel)
model_file = './linear_model_save.pkl'

with open(model_file, 'rb') as file:
    model = pickle.load(file)

# 一个一个预测
svmPredict_onebyone(model, X, y, 9)

# # 批量预测并计算准确度
# y_pre = svmPredict_one2n(model, X)
#
# # 准确度
# accuracy = mean((y_pre == y).astype('f4'))
# print(accuracy)
