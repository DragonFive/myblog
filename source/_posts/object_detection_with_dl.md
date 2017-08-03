---
title: 目标检测算法总结

date: 2017/7/30 12:04:12

categories:
- 深度学习
tags:
- 目标检测
- 深度学习
- 神经网络
---
[TOC]

 去年总结了一篇关于目标检测的博客 [视频智能之——目标检测](https://dragonfive.github.io/object_detection/)，今年到现在有了新的体会，所以就更新一篇。
 ![目标检测题图][1]
<!--more-->

# 目标检测

## 检测算法划分

目标检测的算法大致可以如下划分：
- 传统方法：
1. 基于Boosting框架：Haar/LBP/积分HOG/ICF/ACF等特征+Boosting等
2. 基于SVM：HOG+SVM or DPM等

- CNN方法：
1. 基于region proposal：以Faster R-CNN为代表的R-CNN家族
2. 基于回归区域划分的：YOLO/SSD
3. 给予强化学习的：attentionNet等。 
4. 大杂烩：Mask R-CNN

## selective search 策略

selective search的策略是，因为目标的层级关系，用到了**multiscale**的思想，那我们就尽可能遍历所有的尺度好了，但是不同于暴力穷举，可以先得到小尺度的区域，然后一次次**合并**得到大的尺寸就好了。既然特征很多，那就把我们知道的特征都用上，但是同时也要照顾下**计算复杂度**，不然和穷举法也没啥区别了。最后还要做的是能够**对每个区域进行排序**，这样你想要多少个候选我就产生多少个。

- 使用**Efficient GraphBased** Image Segmentation中的方法来得到region
- 得到所有region之间两两的相似度
- 合并最像的两个region
- 重新计算新合并region与其他region的相似度
- 重复上述过程直到整张图片都聚合成一个大的region
- 使用一种**随机的计分方式**给每个region打分，按照分数进行ranking，取出**top k**的子集，就是selective search的结果

### 区域划分与合并

首先通过**基于图的图像分割方法?**初始化原始区域，就是将图像分割成很多很多的小块。然后我们使用**贪心策略**，计算每两个相邻的区域的相似度，然后每次合并最相似的两块，直到最终只剩下一块完整的图片。然后这其中每次产生的图像块包括合并的图像块我们都保存下来，这样就得到图像的**分层表示**了呢。

优先合并小的区域

### 颜色空间多样性

作者采用了8中不同的颜色方式，主要是为了考虑场景以及光照条件等。这个策略主要应用于【1】中图像分割算法中原始区域的生成。主要使用的颜色空间有：**（1）RGB，（2）灰度I，（3）Lab，（4）rgI（归一化的rg通道加上灰度），（5）HSV，（6）rgb（归一化的RGB），（7）C，（8）H（HSV的H通道）**

使用L1-norm归一化获取图像**每个颜色通道的25 bins**的直方图，这样每个区域都可以得到一个**75维**的向量。![enter description here][2]，区域之间颜色相似度通过下面的公式计算：

![enter description here][3]
 在区域合并过程中使用需要对新的区域进行计算其直方图，计算方法：
![enter description here][4]
优先合并小的区域.





### 距离计算多样性

 这里的纹理采用SIFT-Like特征。具体做法是对每个颜色通道的8个不同方向计算方差σ=1的高斯微分（GaussianDerivative），每个通道每个方向获取10 bins的直方图（L1-norm归一化），这样就可以获取到一个240维的向量。![enter description here][5]


还有大小相似度(优先合并小的区域)和吻合相似度(boundingbox是否在一块儿)

### 给区域打分

这篇文章做法是，给予最先合并的**图片块**较大的权重，比如最后一块完整图像权重为1，倒数第二次合并的区域权重为2以此类推。但是当我们策略很多，多样性很多的时候呢，这个权重就会有太多的重合了，排序不好搞啊。文章做法是给他们乘以一个随机数，毕竟3分看运气嘛，然后对于相同的区域多次出现的也叠加下权重，毕竟多个方法都说你是目标，也是有理由的嘛。

区域的分数是区域内图片块权重之和。
### reference

[目标检测（1）-Selective Search](https://zhuanlan.zhihu.com/p/27467369)

[论文笔记 《Selective Search for Object Recognition》](http://blog.csdn.net/csyhhb/article/details/50425114)


## RCNN 

RCNN作为第一篇目标检测领域的深度学习文章。这篇文章的创新点有以下几点：将CNN用作目标检测的特征提取器、**有监督预训练的方式初始化**CNN、在CNN特征上做**BoundingBox 回归**。

目标检测区别于目标识别很重要的一点是其需要目标的具体位置，也就是BoundingBox。而产生BoundingBox最简单的方法就是滑窗，可以在卷积特征上滑窗。但是我们知道**CNN是一个层次的结构，随着网络层数的加深，卷积特征的步伐及感受野也越来越大**。

### 整体结构 

![RCNN][6]

RCNN的输入为完整图片，首先通过区域建议算法产生一系列的候选目标区域，其中使用的区域建议算法为**Selective Search,选择2K**个置信度最高的区域候选。然后对于这些目标区域候选提取其**CNN特征AlexNet**，并训练**SVM分类**这些特征。最后为了提高定位的准确性在SVM分类后区域基础上进行**BoundingBox回归**。

#### CNN目标特征提取
RCNN使用ImageNet的有标签数据进行有监督的**预训练Alexnet**，然后再在**本数据集上微调最后一层全连接层**。直到现在，这种方法已经成为CNN初始化的标准化方法。但是训练CNN的样本量还是不能少的，为了尽可能获取最多的正样本，RCNN将**IOU>0.5（IoU 指重叠程度，计算公式为：A∩B/A∪B）的样本都称为正样本**。每次ieration 的batch_size为128，其中正样本个数为32，负样本为96.其实这种设置是偏向于正样本的，因为正样本的数量实在是太少了。由于CNN需要固定大小的输入，因此对于每一个区域候选，首先将其防缩至227\*227，然后通过CNN提取特征。


#### svm训练 
数据是在经过微调的RCNN上取得Fc7层特征，然后训练SVM，并通过BoundingBox回归得到的最终结果。
RCNN的SVM训练将**ground truth样本作为正样本**，而**IOU>0.3的样本作为负样本**，这样也是SVM困难样本挖掘的方法。
### reference

[目标检测（2）-RCNN](https://zhuanlan.zhihu.com/p/27473413)



# 其它reference


[CVPR2017-目标检测相关论文](https://zhuanlan.zhihu.com/p/28088956)


  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501725558357.jpg
  [2]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501731400655.jpg
  [3]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501731416622.jpg
  [4]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501731430205.jpg
  [5]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501731713492.jpg
  [6]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501733503711.jpg