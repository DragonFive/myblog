---
title: mobile_net的模型优化

date: 2017/7/17 12:04:12

categories:
- 深度学习
tags:
- deeplearning
- 梯度下降法
- 正则化
- 激活函数
- 神经网络
---
[TOC]


论文出自google的 MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications。
源代码和训练好的模型: [tensorflow版本](https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.md)

![enter description here][1]

<!--more-->



在建立小型和有效的神经网络上，已经有了一些工作，比如SqueezeNet，Google Inception，Flattened network等等。大概分为压缩预训练模型和直接训练小型网络两种。MobileNets主要关注优化延迟，同时兼顾模型大小。

# mobileNets模型结构




[tensorflow官网](https://www.tensorflow.org/mobile/)给出了部署方式，支持android,ios,raspberry Pi等。





**持续更新中。。。。。。。。。。。**

# reference

[github源码](https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.md)

[官方的部署方式](https://www.tensorflow.org/mobile/)

[ 深度学习（六十五）移动端网络MobileNets](http://blog.csdn.net/hjimce/article/details/72831171)

[MobileNets 论文笔记](http://blog.csdn.net/Jesse_Mx/article/details/70766871)

[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications 论文理解](http://www.jianshu.com/p/2fd0c007a560)

[tensorflow训练好的模型怎么调用？](https://www.zhihu.com/question/58287577)

[如何用TensorFlow和TF-Slim实现图像分类与分割](https://www.ctolib.com/topics-101544.html)



  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1500434910512.jpg