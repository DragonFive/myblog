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
目标检测的算法大致可以如下划分：
- 传统方法：
1. 基于Boosting框架：Haar/LBP/积分HOG/ICF/ACF等特征+Boosting等
2. 基于SVM：HOG+SVM or DPM等

- CNN方法：
1. 基于region proposal：以Faster R-CNN为代表的R-CNN家族
2. 基于回归区域划分的：YOLO/SSD
3. 给予强化学习的：attentionNet等。 
4. 大杂烩：Mask R-CNN









# reference
[目标检测（1）-Selective Search](https://zhuanlan.zhihu.com/p/27467369)



  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501725558357.jpg