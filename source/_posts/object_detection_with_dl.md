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

- 使用Efficient GraphBased Image Segmentation中的方法来得到region
- 得到所有region之间两两的相似度
- 合并最像的两个region
- 重新计算新合并region与其他region的相似度
- 重复上述过程直到整张图片都聚合成一个大的region
- 使用一种随机的计分方式给每个region打分，按照分数进行ranking，取出top k的子集，就是selective search的结果

### 区域划分与合并

首先通过**基于图的图像分割方法?**初始化原始区域，就是将图像分割成很多很多的小块。然后我们使用**贪心策略**，计算每两个相邻的区域的相似度，然后每次合并最相似的两块，直到最终只剩下一块完整的图片。然后这其中每次产生的图像块包括合并的图像块我们都保存下来，这样就得到图像的**分层表示**了呢。

优先合并小的区域

### 距离计算与区域打分





# reference
[目标检测（1）-Selective Search](https://zhuanlan.zhihu.com/p/27467369)

[CVPR2017-目标检测相关论文](https://zhuanlan.zhihu.com/p/28088956)

  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501725558357.jpg