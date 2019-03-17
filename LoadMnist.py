import numpy as np
import struct
import matplotlib.pyplot as plt

# 训练集文件
train_labels_idx1_ubyte = '.\\MnistData\\train-labels.idx1-ubyte'
train_images_idx3_ubyte = '.\\MnistData\\train-images.idx3-ubyte'
t10k_labels_idx1_ubyte = '.\\MnistData\\t10k-labels.idx1-ubyte'
t10k_images_idx3_ubyte = '.\\MnistData\\t10k-images.idx3-ubyte'


def load_idx1_ubyte(path, m):
    """
    :param path: 解析idx1文件的通用函数,即读入标签
    :param m: 读取列数
    :return: 数据集y
    """

    with open(path, 'rb') as label_file:
        head_data = label_file.read(8)  # 读入头部数据magic number 和num of items
        magic_number, num_labels = struct.unpack_from('>ii', head_data, 0)
        print('魔数%d,标签数%d张' % (magic_number, num_labels))

        # 读取labels
        num_labels = m  # 列数指定
        fmt = '>' + str(num_labels) + 'B'  # 格式

        labels_data = label_file.read(num_labels)
        labels = np.array(struct.unpack_from(fmt, labels_data)).reshape((num_labels, 1))
        # 把0~9转化为如[0 1 0 0 0 0 0 0 0 0]形式
        # y = np.zeros((num_labels, 10))
        # y[range(num_labels), labels.T[0]] = 1
    # return y
    return labels


def load_idx3_ubyte(path, m):
    """
    :param path: 解析idx1文件的通用函数
    :param m: 读取列数
    :return: 数据集X ,m*784,每个元素在0~1
    """
    with open(path, 'rb') as image_file:
        # 读入头部数据
        head_data = image_file.read(struct.calcsize('iiii'))
        magic_number, num_images, num_rows, num_columns = \
            struct.unpack_from('>iiii', head_data, 0)
        print('魔数%d,图数%d张,图大小%d*%d' %
              (magic_number, num_images, num_rows, num_columns))
        n = num_rows * num_columns  # 特征数量

        # 读取图
        num_images = m  # 列数指定
        fmt = '>' + str(num_images * n) + 'B'  # 格式

        images_data = image_file.read(num_images * n)
        X = np.array(struct.unpack_from(fmt, images_data)).reshape((num_images, n))
        X = X.astype('f4') / 255.0
    return X


def display_image(X, num_images):
    """
    :param X: 数据集
    :param num_images: 画出多少个图像
    :return:
    """
    # 将num_images组装成矩阵，行数是row,列数是col
    row = int(np.floor(num_images ** 0.5))
    col = int(np.ceil(num_images / row))
    length = int(X.shape[1] ** 0.5)  # 每幅图的边长
    pad = 1  # 图与图之间的间隔

    display_array = np.zeros((row * (length + pad) + pad, col * (length + pad) + pad))
    curr_ex = 0
    for i in range(row):
        for j in range(col):
            if curr_ex >= num_images:
                break
            display_array[i * (pad + length) + pad:(i + 1) * (pad + length),
            j * (pad + length) + pad:(j + 1) * (pad + length)] \
                = X[[curr_ex]].reshape(length, length)
            curr_ex += 1
    # 画图
    plt.close()
    plt.imshow(display_array, 'gray')
    plt.show()

# 测试代码
# X = load_idx3_ubyte(train_images_idx3_ubyte, 100)
# display_image(X, 100)
# y = load_idx1_ubyte(train_labels_idx1_ubyte, 100)
# print(y)
def load_idx1_ubyte2(path, m, m2):
    """
    :param path: 解析idx1文件的通用函数,即读入标签
    :param m: 读取列数
    :return: 数据集y
    """

    with open(path, 'rb') as label_file:
        head_data = label_file.read(8)  # 读入头部数据magic number 和num of items
        magic_number, num_labels = struct.unpack_from('>ii', head_data, 0)
        print('魔数%d,标签数%d张' % (magic_number, num_labels))

        # 读取labels
        num_labels = m2-m  # 列数指定
        fmt = '>' + str(num_labels) + 'B'  # 格式
        label_file.read(m)  # 跳过n个
        labels_data = label_file.read(num_labels)
        labels = np.array(struct.unpack_from(fmt, labels_data)).reshape((num_labels, 1))
        # 把0~9转化为如[0 1 0 0 0 0 0 0 0 0]形式
        # y = np.zeros((num_labels, 10))
        # y[range(num_labels), labels.T[0]] = 1
    # return y
    return labels


def load_idx3_ubyte2(path, m, m2):
    """
    :param path: 解析idx1文件的通用函数
    :param m: 读取列数
    :param m2: 与m一起组成切片
    :return: 数据集X ,m*784,每个元素在0~1
    """
    with open(path, 'rb') as image_file:
        # 读入头部数据
        head_data = image_file.read(struct.calcsize('iiii'))
        magic_number, num_images, num_rows, num_columns = \
            struct.unpack_from('>iiii', head_data, 0)
        print('魔数%d,图数%d张,图大小%d*%d' %
              (magic_number, num_images, num_rows, num_columns))
        n = num_rows * num_columns  # 特征数量
        image_file.read(m*n)  # 跳过
        # 读取图
        num_images = m2-m  # 列数指定
        fmt = '>' + str(num_images * n) + 'B'  # 格式

        images_data = image_file.read(num_images * n)
        X = np.array(struct.unpack_from(fmt, images_data)).reshape((num_images, n))
        X = X.astype('f4') / 255.0
    return X
