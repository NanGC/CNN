'''
Model.py
含3层卷积池化层，2层全连接层，激活函数ReLU，采用dropout和softmax函数做分类器
2018/7/18完成搭建，2018/7/19实现文件调用，2018/7/22修改网络结构
-------copyright@GCN-------
'''
import tensorflow as tf

'''
在TensorFlow的世界里，变量的定义和初始化是分开的，所有关于图变量的赋值和计算都要通过tf.Session的run来进行。

想要将所有图变量进行集体初始化时应该使用tf.global_variables_initializer
tf.placehold与tf.Variable的区别：
    tf.placehold 占位符
        主要为真实输入数据和输出标签的输入， 用于在 feed_dict中的变量，不需要指定初始值，具体值在feed_dict中的变量给出。
    tf.Variable 主要用于定义weights bias等可训练会改变的变量，必须指定初始值。
        通过Variable()构造函数后，此variable的类型和形状固定不能修改了，但值可以用assign方法修改。

tf.get_variable和tf.Variable函数差别
相同点：通过两函数创建变量的过程基本一样，
        tf.variable函数调用时提供的维度(shape)信息以及初始化方法(initializer)的参数和tf.Variable函数调用时提供的初始化过程中的参数基本类似。
不同点：两函数指定变量名称的参数不同，
        对于tf.Variable函数，变量名称是一个可选的参数，通过name="v"的形式给出
        tf.get_variable函数，变量名称是一个必填的参数，它会根据变量名称去创建或者获取变量
        
'''

# 函数申明
def weight_variable(shape, n):
    # tf.truncated_normal(shape, mean, stddev)这个函数产生正态分布，均值和标准差自己设定。
    # shape表示生成张量的维度，mean是均值
    # stddev是标准差,，默认最大为1，最小为-1，均值为0
    initial = tf.truncated_normal(shape, stddev=n, dtype=tf.float32)
    return initial

def bias_variable(shape):
    # 创建一个结构为shape矩阵也可以说是数组shape声明其行列，初始化所有值为0.1
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return initial

def conv2d(x, W):
    # 卷积遍历各方向步数为1，SAME：边缘外自动补0，遍历相乘
    # padding 一般只有两个值
    # 卷积层后输出图像大小为：（W+2P-f）/stride+1并向下取整
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x, name):
    # 池化卷积结果（conv2d）池化层采用kernel大小为3*3，步数也为2，SAME：周围补0，取最大值。数据量缩小了4倍
    # x 是 CNN 第一步卷积的输出量，其shape必须为[batch, height, weight, channels];
    # ksize 是池化窗口的大小， shape为[batch, height, weight, channels]
    # stride 步长，一般是[1，stride， stride，1]
    # 池化层输出图像的大小为(W-f)/stride+1，向上取整
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


# 一个简单的卷积神经网络，卷积+池化层 x2，全连接层x2，最后一个softmax层做分类。
# 64个3x3的卷积核（3通道），padding=’SAME’，表示padding后卷积的图与原图尺寸一致，激活函数relu()
def inference(images, batch_size, n_classes):
    # 搭建网络
    # 第一层卷积
    # 第一二参数值得卷积核尺寸大小，即patch；第三个参数是通道数；第四个是卷积核个数
    with tf.variable_scope('conv1') as scope:
        # 所谓名字的scope，指当绑定了一个名字到一个对象的时候，该名字在程序文本中的可见范围
        w_conv1 = tf.Variable(weight_variable([3, 3, 3, 64], 1.0), name='weights', dtype=tf.float32)
        b_conv1 = tf.Variable(bias_variable([64]), name='biases', dtype=tf.float32)   # 64个偏置值
        # tf.nn.bias_add 是 tf.add 的一个特例:tf.add(tf.matmul(x, w), b) == tf.matmul(x, w) + b
        # h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(images, w_conv1), b_conv1), name=scope.name)
        h_conv1 = tf.nn.relu(conv2d(images, w_conv1)+b_conv1, name='conv1')  # 得到128*128*64(假设原始图像是128*128)
    # 第一层池化
    # 3x3最大池化，步长strides为2，池化后执行lrn()操作，局部响应归一化，增强了模型的泛化能力。
    # tf.nn.lrn(input,depth_radius=None,bias=None,alpha=None,beta=None,name=None)
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = max_pool_2x2(h_conv1, 'pooling1')   # 得到64*64*64
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    # 第二层卷积
    # 32个3x3的卷积核（16通道），padding=’SAME’，表示padding后卷积的图与原图尺寸一致，激活函数relu()
    with tf.variable_scope('conv2') as scope:
        w_conv2 = tf.Variable(weight_variable([3, 3, 64, 32], 0.1), name='weights', dtype=tf.float32)
        b_conv2 = tf.Variable(bias_variable([32]), name='biases', dtype=tf.float32)   # 32个偏置值
        h_conv2 = tf.nn.relu(conv2d(norm1, w_conv2)+b_conv2, name='conv2')  # 得到64*64*32

    # 第二层池化
    # 3x3最大池化，步长strides为2，池化后执行lrn()操作
    with tf.variable_scope('pooling2_lrn') as scope:
        pool2 = max_pool_2x2(h_conv2, 'pooling2')  # 得到32*32*32
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    # 第三层卷积
    # 16个3x3的卷积核（16通道），padding=’SAME’，表示padding后卷积的图与原图尺寸一致，激活函数relu()
    with tf.variable_scope('conv3') as scope:
        w_conv3 = tf.Variable(weight_variable([3, 3, 32, 16], 0.1), name='weights', dtype=tf.float32)
        b_conv3 = tf.Variable(bias_variable([16]), name='biases', dtype=tf.float32)   # 16个偏置值
        h_conv3 = tf.nn.relu(conv2d(norm2, w_conv3)+b_conv3, name='conv3')  # 得到32*32*16

    # 第三层池化
    # 3x3最大池化，步长strides为2，池化后执行lrn()操作
    with tf.variable_scope('pooling3_lrn') as scope:
        pool3 = max_pool_2x2(h_conv3, 'pooling3')  # 得到16*16*16
        norm3 = tf.nn.lrn(pool3, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')

    # 第四层全连接层
    # 128个神经元，将之前pool层的输出reshape成一行，激活函数relu()
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(norm3, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        w_fc1 = tf.Variable(weight_variable([dim, 128], 0.005),  name='weights', dtype=tf.float32)
        b_fc1 = tf.Variable(bias_variable([128]), name='biases', dtype=tf.float32)
        h_fc1 = tf.nn.relu(tf.matmul(reshape, w_fc1) + b_fc1, name=scope.name)

    # 第五层全连接层
    # 128个神经元，激活函数relu()
    with tf.variable_scope('local4') as scope:
        w_fc2 = tf.Variable(weight_variable([128 ,128], 0.005),name='weights', dtype=tf.float32)
        b_fc2 = tf.Variable(bias_variable([128]), name='biases', dtype=tf.float32)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc1, name=scope.name)


    # 对卷积结果执行dropout操作
    # keep_prob = tf.placeholder(tf.float32)
    h_fc2_dropout = tf.nn.dropout(h_fc2, 0.5)
    # tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None)
    # 第二个参数keep_prob: 设置神经元被选中的概率,在初始化时keep_prob是一个占位符

    # Softmax回归层
    # 将前面的FC层输出，做一个线性回归，计算出每一类的得分，在这里是2类，所以这个层输出的是两个得分。
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(weight_variable([128, n_classes], 0.005), name='softmax_linear', dtype=tf.float32)
        biases = tf.Variable(bias_variable([n_classes]), name='biases', dtype=tf.float32)
        softmax_linear = tf.add(tf.matmul(h_fc2_dropout, weights), biases, name='softmax_linear')
        # softmax_linear = tf.nn.softmax(tf.add(tf.matmul(h_fc2_dropout, weights), biases, name='softmax_linear'))
    return softmax_linear
    # 最后返回softmax层的输出


# loss计算
# 传入参数：logits，网络计算输出值。labels，真实值，在这里是0或者1
# 返回参数：loss，损失值
def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss

# loss损失值优化
# 输入参数：loss。learning_rate，学习速率。
# 返回参数：train_op，训练op，这个参数要输入sess.run中让模型去训练。
def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

# 评价/准确率计算
# 输入参数：logits，网络计算值。labels，标签，也就是真实值，在这里是0或者1。
# 返回参数：accuracy，当前step的平均准确率，也就是在这些batch中多少张图片被正确分类了。
def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float16))
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy

