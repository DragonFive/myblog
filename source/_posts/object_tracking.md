title: 视频智能之——目标跟踪
date: 2016/12/27 09:31:12
categories:
- 计算机视觉
tags:
- 目标跟踪
- 深度学习
- 论文
- ct
- kcf
---



单目标跟踪是指：给出目标在跟踪视频第一帧中的初始状态（如位置，尺寸），自动估计目标物体在后续帧中的状态。跟踪过程中会出现目标发生剧烈形变、被其他目标遮挡或出现相似物体干扰等等各种复杂的情况

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-70583d1994fa4e1d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

<!--more-->

cvpr2013的论文 Online Object Tracking: A Benchmark 总结了大量的算法


# 目标跟踪的benchmark视频

[目标跟踪领域benchmark的视频](http://cvlab.hanyang.ac.kr/tracker_benchmark/)





# 以往方法小结

1. 压缩跟踪 compressive tracking

2. 粒子滤波 Particle Filter Object Tracking

别的还有什么质心算法之类的完全没法用，压缩跟踪会好一些，粒子滤波只是指跟踪的过程中帧与帧之间的更跟方法，具体使用中还要配合颜色直方图的特征或者其他特征进行相似度计算。压缩跟踪的话应该指的是一套完整的跟踪过程

这是个很有意思的地方，在很多时候，我们之所以需要跟踪算法，是因为我们的检测算法很慢，跟踪很快。基本上当前排名前几的跟踪算法都很难用在这样的情况下，因为你实际的速度已经太慢了，比如TLD，CT，还有Struct，如果目标超过十个，基本上就炸了。况且还有些跟踪算法自己drift掉了也不知道，比如第一版本的CT是无法处理drift的问题的，TLD是可以的，究其原因还是因为检测算法比较鲁棒啊……

实际中速度极快，实现也简单的纯跟踪算法居然是NCC和Overlap。

这两年有相关滤波系列的跟踪器，速度与精度都相当给力



## TLD跟踪算法
PAMI2011 30fps左右

[TLD（Tracking-Learning-Detection）学习与源码理解之](http://blog.csdn.net/kezunhai/article/details/11675347)

[Tracking-Learning-Detection原理分析](http://johnhany.net/2014/05/tld-the-theory/)

[ TLD（Tracking-Learning-Detection）学习与源码理解](http://blog.csdn.net/zouxy09/article/details/7893011)

## 压缩感知追踪算法 

ECCV2012  60fps左右，压缩感知矩阵很重要。
[压缩跟踪Compressive Tracking综述](http://blog.csdn.net/pkueecser/article/details/8953252)

[压缩跟踪Compressive Tracking](http://blog.csdn.net/zouxy09/article/details/8118360)

## KCF tracker(csk tracker)

ECCV PAMI都有发，是FPS可以做到300fps,

[论文笔记：目标追踪-CVPR2014-Adaptive Color Attributes for Real-time Visual Tracking](http://blog.csdn.net/hlinghling/article/details/44308199)

CSK tracker 有很多改进，主要原因是速度快
融合自适应颜色信息的改进算法有：
CVPR2014 adaptive color attributes for real time visual tracking.pdf

融合part-based信息的能够抵抗部分遮挡的改进算法有：
cvpr2015 real-time part-based visual tracking via adaptive corelation filters.pdf

[目标跟踪算法——KCF入门详解](http://blog.csdn.net/crazyice521/article/details/53525366)


## 其他算法

[时空上下文视觉跟踪（STC）算法的解读与代码复现](http://blog.csdn.net/zouxy09/article/details/16889905)

[最简单的目标跟踪（模版匹配）](http://blog.csdn.net/zouxy09/article/details/13358977)

[基于感知哈希算法的视觉目标跟踪](http://blog.csdn.net/zouxy09/article/details/17471401)

# 深度学习大讲堂参考内容

## 经典目标跟踪方法
目前跟踪算法可以被分为产生式(generative model)和判别式(discriminative model)两大类别。产生式方法着眼于对目标本身的刻画，忽略背景信息，在目标自身变化剧烈或者被遮挡时容易产生漂移。

判别式方法通过训练分类器来区分目标和背景。这种方法也常被称为**tracking-by-detection**。近年来，各种机器学习算法被应用在判别式方法上，其中比较有代表性的有多示例学习方法(multiple instance learning), boosting和结构SVM(structured SVM)等。判别式方法因为**显著区分背景和前景的信息**。值得一提的是，目前大部分深度学习目标跟踪方法也归属于判别式框架。

基于相关滤波(correlation filter)的跟踪方法因为速度快,效果好吸引了众多研究者的目光。相关滤波器通过**将输入特征回归为目标高斯分布来训练 filters**。并在后续跟踪中**寻找预测分布中的响应峰值来定位目标的位置**。相关滤波器在运算中巧妙应用快速傅立叶变换获得了大幅度速度提升。目前基于相关滤波的拓展方法也有很多，包括核化相关滤波器(kernelized correlation filter, KCF), 加尺度估计的相关滤波器(DSST)等。

## 基于深度学习的目标跟踪方法

深度模型的魔力之一来自于对大量标注训练数据的有效学习，而目标跟踪仅仅提供第一帧的bounding-box作为训练数据。这种情况下，在跟踪开始针对当前目标从头训练一个深度模型困难重重。目前基于深度学习的目标跟踪算法采用了几种思路来解决这个问题。
### 利用辅助图片数据预训练深度模型，在线跟踪时微调

在目标跟踪的训练数据非常有限的情况下，使用**辅助的非跟踪训练数据**进行预训练，获取对物体特征的通用表示(general representation )，在实际跟踪时，通过利用当前跟踪目标的**有限样本信息对预训练模型微调**(fine-tune), 使模型对当前跟踪目标有更强的分类性能，这种**迁移学习**的思路极大的减少了对跟踪目标训练样本的需求，也提高了跟踪算法的性能。

这个方面代表性的作品有DLT和SO-DLT。

>DLT(NIPS2013) Learning a Deep Compact Image Representation for Visual Tracking

DLT作为第一个将深度网络运用于单目标跟踪的跟踪算法，首先提出了“离线预训练＋在线微调”的思路，很大程度的解决了跟踪中训练样本不足的问题

>SO-DLT(arXiv2015)Transferring Rich Feature Hierarchies for Robust Visual Tracking

SO-DLT作为large-scale CNN网络在目标跟踪领域的一次成功应用，取得了非常优异的表现.但是SO－DLT离线预训练依然使用的是大量无关联图片，作者认为使用更贴合跟踪实质的时序关联数据是一个更好的选择。


### 利用现有大规模分类数据集预训练的CNN分类网络提取特征

2015年以来，在目标跟踪领域应用深度学习兴起了一股新的潮流。即直接使用ImageNet这样的大规模**分类**数据库上训练出的CNN网络如VGG-Net获得目标的特征表示，之后再用**观测模型(observation model)进行分类获得跟踪结果**。这种做法既避开了跟踪时直接训练large-scale CNN 样本不足的困境，也充分利用了深度特征强大的表征能力。

>FCNT(ICCV15)Visual Tracking with Fully Convolutional Networks

作为应用CNN特征于物体跟踪的代表作品，FCNT的亮点之一在于对ImageNet上预训练得到的CNN特征在目标跟踪任务上的性能做了深入的分析,并根据分析结果设计了后续的网络结构。


>Hierarchical Convolutional Features for Visual Tracking(ICCV15)

这篇是作者在2015年度看到的最简洁有效的利用深度特征做跟踪的论文。其主要思路是提取深度特征，之后利用相关滤波器确定最终的bounding-box。

分类任务预训练的CNN网络本身更关注区分类间物体，忽略类内差别。目标跟踪时只关注一个物体，重点区分该物体和背景信息，明显抑制背景中的同类物体，但是还需要对目标本身的变化鲁棒。分类任务以相似的一众物体为一类，跟踪任务以同一个物体的不同表观为一类，使得这两个任务存在很大差别，这也是两篇文章融合多层特征来做跟踪以达到较理想效果的动机所在。

### 利用跟踪序列预训练，在线跟踪时微调

>MDNet(CVPR2016)Learning Multi-Domain Convolutional Neural Networks for Visual Tracking

意识到图像**分类任务和跟踪之间存在巨大差别**，MDNet提出**直接用跟踪视频预训练CNN获得general的目标表示能力**的方法。但是序列训练也存在问题，即不同跟踪序列跟踪目标完全不一样，某类物体在一个序列中是跟踪目标，在另外一个序列中可能只是背景。不同序列中目标本身的表观和运动模式、环境中光照、遮挡等情形相差甚大。这种情况下，想要用同一个CNN完成所有训练序列中前景和背景区分的任务，困难重重。

速度仍较慢。且boundingbox回归也需要单独训练.

>RTT(CVPR16)Recurrently Target-Attending Tracking

这篇文章的出发点比较有意思，即利用多方向递归神经网络(multi-directional recurrent neural network)来建模和挖掘对整体跟踪有用的可靠目标部分(reliable part)，实际上是二维平面上的RNN建模，最终解决预测误差累积和传播导致的跟踪漂移问题


# 参考资料

[深度学习在目标跟踪中的应用-深度学习大讲堂](https://zhuanlan.zhihu.com/p/22334661)


[ 译文 Online Object Tracking: A Benchmark](http://blog.csdn.net/shanglianlm/article/details/47376323)
