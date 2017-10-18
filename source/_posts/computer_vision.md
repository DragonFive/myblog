---
title: 计算机视觉问题小结
date: 2017/6/5 17:38:58

categories:
- 计算机视觉
tags:
- 目标检测
- 深度学习
- 神经网络
- 特征
----

# 特征

## hog

histgram of gradient: 每个像素计算梯度方向和大小，然后8x8个像素组成一个cell, cell 之间不重叠，冲击cell里面各像素的梯度方向脂肪像，每20度算一个方向，累加这些方向上的梯度值得到9维的特征。然后每四个cell组成一个block，block可以重叠，一个block的特征向量是四个cell特征串联起来得到4x9=36维的特征。所有的block再合起来就得到整张图的hog特征 


[HOG特征算法](http://blog.csdn.net/hujingshuang/article/details/47337707)


## haar特征



# 深度学习调参技巧 
训练的基本流程 

1. 数据处理
2. 定义网络结构
3. 配置solver参数
4. 训练：如 caffe train -solver solver.prototxt -gpu 0





