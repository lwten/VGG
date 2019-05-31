# VGG
利用 Vggnet-16 识别人民币面值，详细说明：https://www.tinymind.cn/articles/4185。
文件主要包括4个：
-vgg16_trainable.py: VGG-16的网络结构定义
-train_VGG16.py：VGG-16的训练代码
-utils.py:其他代码里用到的一些工具函数
-test_dataset.py:利用训练好的VGG-16识别人民币面值。

说明：代码基于ImageNet预训练的参数进行迁移学习，预训练模型vgg16.npy可以网盘下载：https://pan.baidu.com/s/1tbeZgYEbuQYdSAcdmrX-fg，密码：bh96。
代码里面的路径需要自己的实际情况进行修改。
