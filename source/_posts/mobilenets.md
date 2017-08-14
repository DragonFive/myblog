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

## 深度可分解卷积 
MobileNets模型基于**深度可分解的卷积**，它可以**将标准卷积分解成一个深度卷积和一个点卷积（1 × 1卷积核）**。标准卷积核为：a × a × c，其中a是卷积核大小，c是卷积核的通道数，本文将其一分为二，一个卷积核是a × a × 1，一个卷积核是1 ×1 × c。简单说，就是标准卷积同时完成了**2维卷积计算和改变特征数量**两件事，本文把这两件事分开做了。后文证明，这种分解可以有效减少计算量，降低模型大小。

![深度可分解的卷积][2]


首先是标准卷积，假定输入F的维度是 DF×DF×M ，经过标准卷积核K得到输出G的维度 DG×DG×N ，卷积核参数量表示为 DK×DK×M×N 。如果计算代价也用数量表示，应该为 DK×DK×M×N×DF×DF 。

现在将卷积核进行分解，那么按照上述计算公式，可得深度卷积的计算代价为 DK×DK×M×DF×DF ，点卷积的计算代价为 M×N×DF×DF 。

![参数量][3]


## 模型结构和训练 

![模型][4]

![mobilenet架构][5]






















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
  [2]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1502675769608.jpg
  [3]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1502676514289.jpg
  [4]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1502677244854.jpg
  [5]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1502677189961.jpg