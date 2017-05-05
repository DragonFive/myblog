---
title: MDNet学习笔记

date: 2017/4/24 15:04:12

categories:
- 计算机视觉
tags:
- objectTracking
- 目标跟踪
- 深度学习
---
[TOC]

MDNet:[paper-](https://arxiv.org/pdf/1510.07945v2.pdf)
[project site](http://cvlab.postech.ac.kr/research/mdnet/)
[github repository](https://github.com/HyeonseobNam/MDNet)
[作者做的presentation](http://votchallenge.net/vot2015/download/presentation_Hyeonseob.pdf)
[python版实现](https://github.com/edgelord/MDNet)
![enter description here][1]
<!--more-->

# introduction
把CNN用在跟踪问题中，最大的问题是数据量太少（1），这样训练出来的网络容易过拟合，之前王乃岩等的做法是用imageNet来训练（2），但是很难保证数据的一致性，所以用视频跟踪的数据来训练,学习这些序列里目标运动的共性（3）。用一个目标可能在某个视频序列里是要跟踪的目标而在另一个环境中就成了背景（4），跟踪问题一个网络只需要分两类：目标和背景，不需要很大的网络(5).


MDNet的**网络结构**是前面一个是共享的CNN网络用来提取图像的特征，在训练的过程中主要训练前面共享网络的参数，而后面的是针对不用视频序列的分支的二值分类器，训练过程调整前面共享网络的参数，使得二值分类器的结果与groundtruth的结果一致。而在跟踪过程中，新建一个后面专用的domain，然后在线训练这块儿的网络。MDNET另一个特点是层数比resnet和alexnet少。


算法包括两个阶段：**多域表示学习和在线视觉跟踪**，在线视觉跟踪阶段fine-tune的是共享部分的全连接层和后面的分类层。

王乃岩2015年的文章用的方法：Structured output CNN[Wang et al. Arxiv’15] 文章名字叫做： Transferring rich feature hierarchies for robust visual tracking（放在axiv2015上没中）这篇文章是典型的用imagenet训练网络来做追踪的，效果不好的原因是分类与跟踪问题最基本的不同：分类是对目标打标签，而跟踪是定位任意目标的位置。

本文的贡献在于：
- 提出了一个基于CNN的多域学习框架，把域无关的信息从域相关的信用中分出来用来做共享的表示层。
- 第一次使用跟踪的数据做训练，效果比较好

早期的CNN方法针对特定目标做跟踪，比如2010年智能跟踪人，而2014年的缺少数据,2015年的王乃岩和HongS[20]使用imagenet效果并不好,hongS表明了更深的网络会丢失一些空间信息，对目标定位不好。multi_domain_learning最早是用在NLP领域的。在视觉中之前只用在一些域自适应的场景中。

# MDNet网络结构的离线训练
如上面的图所示，包含了共享的层和K个域专用的分支层，黄色和蓝色的包围框表示的分别是正样本和负样本。其中**卷积层与VGG-M 网络一致**,论文为：M. Danelljan, G. Hager, F. Khan, and M. Felsberg. Accurate scale estimation for robust visual tracking. In BMVC, 2014。 而全连接层的输出为512层，用了relu和dropout。K个分支包含一个二值分类器用softmax做交叉熵损失函数。在共享层学习的是一些共有的东西，比如对光照变化、运动抖动、尺度变化的鲁棒性。在训练网络的过程中每一个iteration中，来自某个branch的数据做minibatch，同时只有这个branch工作来完成此次迭代的训练，整个过程是一个随机梯度下降法的过程。


# MDNet的在线跟踪
## 网络更新
跟踪控制和网络更新，**网络更新**有长时更新(使用长时间累积的正样本)和短时更新（用在跟踪出错的时候）的区分，这里不是很懂。

## 难例挖掘
在tracking-by-detection的方法中大部分的负样本是不重要的或者是冗余的，平均低对待这些样本容易造成漂移。[35] Example-based learning for viewbased human face detection 给了**难例挖掘**的方法，随着训练的进行，**负样本越来越难以被分类*。

## 包围框回归

由于CNN的特征抽象层次高并且数据增广策略在选择正样本的时候是在目标的周围选的一些样本，这样会导致最终找到的包围框不能最小的包围到那个目标，所以我们要进行bounding box regression。最早用在目标检测领域,[13,11] DPM算法也这么用。Rich feature
hierarchies for accurate object detection and semantic segmentation。



# 参考资料 

[目标跟踪算法五：MDNet: Learning Multi-Domain Convolutional Neural Networks for Visual Tracking](https://zhuanlan.zhihu.com/p/25312850)

[深度学习在目标跟踪中的应用](https://zhuanlan.zhihu.com/p/22334661)

[目标跟踪之NIUBILITY的相关滤波](https://zhuanlan.zhihu.com/DCF-tracking)

[cvpr论文阅读笔记](http://www.cnblogs.com/wangxiaocvpr/)

[CNN-tracking-文章导读](http://blog.csdn.net/ben_ben_niao/article/details/51315000)




[物体跟踪-CVPR16-tracking](http://blog.csdn.net/ben_ben_niao/article/details/52072659)


[一个人的论文阅读笔记](http://blog.csdn.net/u012905422/article/category/6223501)


  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1493027913275.jpg "1493027913275"