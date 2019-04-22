# CNN
Training a convolutional neural network using its own picture dataset

Run train.py directly

The running parameters of the network are saved in the CK+ folder. If you change the name, retraining the thing.ckpt will be overwritten.

Explain：the test code is supplemented by the follow-up. In order to simplify the training and facilitate the test, three types of pictures are randomly crawled on the network as the training set and the test set.

When actually use your own datasets, just put the datasets together with codes in a folder, modify the relevant path in the code.

More specific can refer to the blog:
Https://blog.csdn.net/jesmine_gu/article/details/81155787


直接运行train.py就可以

网络的运行参数保存在CK+文件夹里面，不改名字重新训练thing.ckpt会被覆盖

说明，测试代码是后续补上的，为了简化训练方便测试，通过在网上随机爬取三类图片作为训练集和测试集。

实际换用自己的数据集时，只要把是数据集跟代码放在一个文件夹里，修改训练以及测试代码中的相关路径就可以了。

更具体的可以参见博客：
完整实现利用tensorflow训练自己的图片数据集
https://blog.csdn.net/jesmine_gu/article/details/81155787
