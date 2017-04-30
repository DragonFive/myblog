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


把CNN用在跟踪问题中，最大的问题是数据量太少（1），这样训练出来的网络容易过拟合，之前王乃岩等的做法是用imageNet来训练（2），但是很难保证数据的一致性，所以用视频跟踪的数据来训练（3）。用一个目标可能在某个视频序列里是要跟踪的目标而在另一个环境中就成了背景（4），跟踪问题一个网络只需要分两类：目标和背景，不需要很大的网络(5).


# 参考资料 

[目标跟踪算法五：MDNet: Learning Multi-Domain Convolutional Neural Networks for Visual Tracking](https://zhuanlan.zhihu.com/p/25312850)

[深度学习在目标跟踪中的应用](https://zhuanlan.zhihu.com/p/22334661)

[目标跟踪之NIUBILITY的相关滤波](https://zhuanlan.zhihu.com/DCF-tracking)

[cvpr论文阅读笔记](http://www.cnblogs.com/wangxiaocvpr/)

[CNN-tracking-文章导读](http://blog.csdn.net/ben_ben_niao/article/details/51315000)




[物体跟踪-CVPR16-tracking](http://blog.csdn.net/ben_ben_niao/article/details/52072659)


[一个人的论文阅读笔记](http://blog.csdn.net/u012905422/article/category/6223501)


  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1493027913275.jpg "1493027913275"