---
title: squeeze_net的模型优化

date: 2017/7/20 12:04:12

categories:
- 深度学习
tags:
- deeplearning
- 梯度下降法
- 正则化
- 激活函数
- 神经网络
---


<div class="github-widget" data-repo="DragonFive/deep-learning-exercise"></div>

SqueezeNet主要是为了降低CNN模型参数数量而设计的。

<!--more-->

使用的squeezenet的pre-trained model来自[SqueezeNet repo](https://github.com/DeepScale/SqueezeNet)

# 设计原则

（1）替换3x3的卷积kernel为**1x1的卷积kernel**

卷积模板的选择，从12年的AlexNet模型一路发展到2015年底Deep Residual Learning模型，基本上卷积大小都选择在3x3了，因为其有效性，以及设计简洁性。本文替换3x3的卷积kernel为1x1的卷积kernel可以让参数缩小9X。但是为了不影响识别精度，并不是全部替换，而是一部分用3x3，一部分用1x1。具体可以看后面的模块结构图。

（2）减少输入3x3卷积的input feature map数量 
如果是conv1-conv2这样的直连，那么实际上是没有办法**减少conv2的input feature map**数量的。所以作者巧妙地把原本一层conv分解为两层，并且封装为一个**Fire Module**。


（3）**减少pooling **
这个观点在很多其他工作中都已经有体现了，比如GoogleNet以及Deep Residual Learning。


## Fire Module

Fire Module是本文的核心构件，思想非常简单，就是将原来简单的一层conv层变成两层：**squeeze层+expand层**，各自带上Relu激活层。在squeeze层里面全是1x1的卷积kernel，数量记为S11；在expand层里面有1x1和3x3的卷积kernel，数量分别记为E11和E33，**要求S11 < input map number即满足上面的设计原则（2）**。expand层之后将1x1和3x3的卷积output feature maps在**channel维度拼接起来**。

![squeezenet][1]








**持续更新中。。。。。。。。。。。**

# reference
[深度学习（六十二）SqueezeNet网络设计思想笔记](http://blog.csdn.net/hjimce/article/details/72809131)

[ 深度学习方法（七）：最新SqueezeNet 模型详解](http://blog.csdn.net/xbinworld/article/details/50897870)


  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1502707058373.jpg