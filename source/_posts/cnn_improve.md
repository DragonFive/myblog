---
title: 神经网络的压缩优化

date: 2017/8/5 12:04:12

categories:
- 深度学习
tags:
- deeplearning
- 网络优化
- 神经网络
---
[TOC]

回顾一下几个经典模型，我们主要看看深度和caffe模型大小，[神经网络模型演化](https://dragonfive.github.io/2017-07-05/deep_learning_model/)

![各种CNN模型][1]

模型大小(参数量)和模型的深浅并非是正相关。

<!--more-->

# 一些经典的模型设计路线

## fully connect to local connect 全连接到卷积神经网络
Alexnet[1]是一个8层的卷积神经网络，有约**60M个参数**，如果采用**32bit float存下来有200M**。值得一提的是，AlexNet中仍然有3个全连接层，其参数量占比参数总量超过了90%。





# reference

[知乎:为了压榨CNN模型，这几年大家都干了什么](https://zhuanlan.zhihu.com/p/25797790)


  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1502693251377.jpg