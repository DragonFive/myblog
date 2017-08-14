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


## googLeNet
从这里开始pooling层其实变少了。
要想提高CNN的网络能力，比如分类准确率，一般的想法就是增大网络，比如Alexnet确实比以前早期Lenet大了很多，但是纯粹的增大网络——比如把**每一层的channel数量翻倍**——但是这样做有两个缺点——**参数太多容易过拟合，网络计算量也会越来越大**。
### inception v1
![inceptionv1结构][2]

一分四，然后做一些不同大小的卷积，之后再堆叠feature map。这样提取不同尺度的特征，能够提高网络表达能力。

目前很多工作证明，要想增强网络能力，可以：增加网络深度，增加网络宽度；但是为了**减少过拟合，也要减少自由参数**。因此，就自然而然有了这个第一版的Inception网络结构——同一层里面，有卷积1x1, 3x 3,5x 5 **不同的卷积模板，他们可以在不同size的感受野做特征提取**，也算的上是一种混合模型了。因为**Max Pooling本身也有特征提取的作用**，而且和卷积不同，没有参数不会过拟合，也作为一个分支。但是直接这样做，整个网络计算量会较大，且层次并没有变深，因此，在3x3和5x5卷 积前面**先做1x1的卷积，降低input的channel数量，这样既使得网络变深，同时计算量反而小了**；（在每一个卷积之后都有**ReLU**）

### inception v2 v3

用1x3和3x1卷积替代3x3卷积，计算量少了很多，深度变深，思路是一样的。（实际上是1xn和nx1替代nxn，n可以变）,使用的是不对称的卷积核

![incepiton v2][3]



## VGG

特点也是连续conv多，计算量巨大

![做连续卷积][4]


## resnet

特殊之处在于设计了“bottleneck”形式的block（有跨越几层的直连）。最深的model采用的152层

block的结构如下图 

![block][5]

# reference


  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1502709541017.jpg
  [2]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1502709248613.jpg
  [3]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1502710845336.jpg
  [4]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1502710007686.jpg
  [5]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1502710123203.jpg