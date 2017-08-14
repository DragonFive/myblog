---
title: 神经网络模型演化
date: 2017/7/5 17:38:58

categories:
- 深度学习
tags:
- deeplearning
- alexnet
- googlenet
- caffenet
- caffe
---
[TOC]

1. Lenet，1986年
2. Alexnet，2012年
3. GoogleNet，2014年
4. VGG，2014年
5. Deep Residual Learning，2015年


<!--more-->

<div class="github-widget" data-repo="DragonFive/python_cv_AI_ML"></div>


# 网络结构的基础知识
1. 下采样层的目的是为了**降低网络训练参数**及模型的**过拟合程度**。
2. LRN局部响应归一化有利于模型泛化。（过拟合）
3. alexnet做了重叠池化，与lenet不同。也就是说它的pool kernel的步长比kernel size要小。
4. dropout在每个不同的样本进来的时候用不同的一半的神经元做fc层。但他们共享权重，
5. relu是线性的，非饱和的。收敛速度比sigmoid和tanh快
6. inception的主要思路是用密集成分来近似局部稀疏结构。网络越到后面，特征越来越抽象，3x3、5x5的卷积的比例也在增加，但是5x5的卷积计算量会大，后续层的参数会多，因此需要用1x1的卷积进行降维

## alexNet

它证明了CNN在复杂模型下的有效性，然后GPU实现使得训练在可接受的时间范围内得到结果
Alexnet有一个特殊的计算层，LRN层，做的事是对当前层的输出结果做平滑处理。

前面的结构  conv - relu - pool - LRN

![卷积层][1]

全连接的结构 fc - relu - dropout


# googLeNet
从这里开始pooling层其实变少了。
![inception结构][2]

一分四，然后做一些不同大小的卷积，之后再堆叠feature map。这样提取不同尺度的特征，能够提高网络表达能力。


# reference


  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1502709541017.jpg
  [2]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1502709248613.jpg