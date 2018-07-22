'''
PreW  ork.py
功能：实现对指定大小的生成图片进行sample与label分类制作
获得神经网络输入的get_files文件，同时为了方便网络的训练，输入数据进行batch处理。

2018/7/19完成
-------copyright@GCN-------
'''

import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import *

angry = []
label_angry = []
disgusted = []
label_disgusted = []
fearful = []
label_fearful = []
happy = []
label_happy = []
sadness = []
label_sadness = []
surprised = []
label_surprised = []


def get_file(file_dir):
    # step1：获取路径下所有的图片路径名，存放到
    # 对应的列表中，同时贴上标签，存放到label列表中。
    for file in os.listdir(file_dir + '/angry'):
        angry.append(file_dir + '/angry' + '/' + file)
        label_angry.append(0)
    for file in os.listdir(file_dir + '/disgusted'):
        disgusted.append(file_dir + '/disgusted' + '/' + file)
        label_disgusted.append(1)
    for file in os.listdir(file_dir + '/fearful'):
        fearful.append(file_dir + '/fearful' + '/' + file)
        label_fearful.append(2)
    for file in os.listdir(file_dir + '/happy'):
        happy.append(file_dir + '/happy' + '/' + file)
        label_happy.append(3)
    for file in os.listdir(file_dir + '/sadness'):
        sadness.append(file_dir + '/sadness' + '/' + file)
        label_sadness.append(4)
    for file in os.listdir(file_dir + '/surprised'):
        surprised.append(file_dir + '/surprised' + '/' + file)
        label_surprised.append(5)

    # 打印出提取图片的情况，检测是否正确提取
    print("There are %d angry\nThere are %d disgusted\nThere are %d fearful\n" %(len(angry), len(disgusted), len(fearful)),end="")
    print("There are %d happy\nThere are %d sadness\nThere are %d surprised\n" %(len(happy),len(sadness),len(surprised)))

    # step2：对生成的图片路径和标签List做打乱处理把所有的合起来组成一个list（img和lab）
    # 合并数据numpy.hstack(tup)
    # tup可以是python中的元组（tuple）、列表（list），或者numpy中数组（array），函数作用是将tup在水平方向上（按列顺序）合并
    image_list = np.hstack((angry, disgusted, fearful, happy, sadness, surprised))
    label_list = np.hstack((label_angry, label_disgusted, label_fearful, label_happy, label_sadness, label_surprised))
    # 利用shuffle，转置、随机打乱
    temp = np.array([image_list, label_list])   # 转换成2维矩阵
    temp = temp.transpose()     # 转置
    # numpy.transpose(a, axes=None) 作用：将输入的array转置，并返回转置后的array
    np.random.shuffle(temp)     # 按行随机打乱顺序函数

    # 将所有的img和lab转换成list
    all_image_list = list(temp[:, 0])    # 取出第0列数据，即图片路径
    all_label_list = list(temp[:, 1])    # 取出第1列数据，即图片标签
    label_list = [int(i) for i in label_list]   # 转换成int数据类型

    ''' 
    # 将所得List分为两部分，一部分用来训练tra，一部分用来测试val
    # ratio = 0.8
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample * ratio))  # 测试样本数, ratio是测试集的比例
    n_train = n_sample - n_val  # 训练样本数

    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]   # 转换成int数据类型
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]   # 转换成int数据类型

    return tra_images, tra_labels, val_images, val_labels
    '''
    return image_list, label_list

# 将image和label转为list格式数据，因为后边用到的的一些tensorflow函数接收的是list格式数据
# 为了方便网络的训练，输入数据进行batch处理
# image_W, image_H, ：图像高度和宽度
# batch_size：每个batch要放多少张图片
# capacity：一个队列最大多少
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # step1：将上面生成的List传入get_batch() ，转换类型，产生一个输入队列queue
    # tf.cast()用来做类型转换
    image = tf.cast(image, tf.string)   # 可变长度的字节数组.每一个张量元素都是一个字节数组
    label = tf.cast(label, tf.int32)
    # tf.train.slice_input_producer是一个tensor生成器
    # 作用是按照设定，每次从一个tensor列表中按顺序或者随机抽取出一个tensor放入文件名队列。
    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])   # tf.read_file()从队列中读取图像

    # step2：将图像解码，使用相同类型的图像
    image = tf.image.decode_jpeg(image_contents, channels=3)
    # jpeg或者jpg格式都用decode_jpeg函数，其他格式可以去查看官方文档

    # step3：数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # 对resize后的图片进行标准化处理
    image = tf.image.per_image_standardization(image)

    # step4：生成batch
    # image_batch: 4D tensor [batch_size, width, height, 3], dtype = tf.float32
    # label_batch: 1D tensor [batch_size], dtype = tf.int32
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=16, capacity=capacity)

    # 重新排列label，行数为[batch_size]
    label_batch = tf.reshape(label_batch, [batch_size])
    # image_batch = tf.cast(image_batch, tf.uint8)    # 显示彩色图像
    image_batch = tf.cast(image_batch, tf.float32)    # 显示灰度图
    # print(label_batch) Tensor("Reshape:0", shape=(6,), dtype=int32)
    return image_batch, label_batch
    # 获取两个batch，两个batch即为传入神经网络的数据

"""
def PreWork():
    # 对预处理的数据进行可视化，查看预处理的效果
    IMG_W = 256
    IMG_H = 256
    BATCH_SIZE = 6
    CAPACITY = 64

    train_dir = 'DeepLearning/CNN/CK+_part'

    # image_list, label_list, val_images, val_labels = get_file(train_dir)
    image_list, label_list = get_file(train_dir)
    image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
    print(label_batch.shape)

    lists = ('angry', 'disgusted', 'fearful', 'happy', 'sadness', 'surprised')

    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()  # 创建一个线程协调器，用来管理之后在Session中启动的所有线程
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop() and i < 1:
                # 提取出两个batch的图片并可视化。
                img, label = sess.run([image_batch, label_batch])  # 在会话中取出img和label
                # img = tf.cast(img, tf.uint8)

                '''
                1、range()返回的是range object，而np.arange()返回的是numpy.ndarray()
                range(start, end, step)，返回一个list对象，起始值为start，终止值为end，但不含终止值，步长为step。只能创建int型list。
                arange(start, end, step)，与range()类似，但是返回一个array对象。需要引入import numpy as np，并且arange可以使用float型数据。
                
                2、range()不支持步长为小数，np.arange()支持步长为小数
                
                3、两者都可用于迭代
                range尽可用于迭代，而np.nrange作用远不止于此，它是一个序列，可被当做向量使用。
                '''
                for j in np.arange(BATCH_SIZE):
                    # np.arange()函数返回一个有终点和起点的固定步长的排列
                    print('label: %d' % label[j])
                    plt.imshow(img[j, :, :, :])
                    title = lists[int(label[j])]
                    plt.title(title)
                    plt.show()
                i += 1
        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    PreWork()
"""