---
title: 计算机视觉问题小结

date: 2017/11/5 17:38:58

categories:
- 计算机视觉
tags:
- 目标检测
- 深度学习
- 神经网络
- 特征
---

# 特征

## hog

histgram of gradient: 每个像素计算梯度方向和大小，然后8x8个像素组成一个cell, cell 之间不重叠，冲击cell里面各像素的梯度方向脂肪像，每20度算一个方向，累加这些方向上的梯度值得到9维的特征。然后每四个cell组成一个block，block可以重叠，一个block的特征向量是四个cell特征串联起来得到4x9=36维的特征。所有的block再合起来就得到整张图的hog特征 


[HOG特征算法](http://blog.csdn.net/hujingshuang/article/details/47337707)


## haar特征



# 深度学习调参技巧 

## 调参优先级
按照对结果的影响调整：

1. 调整学习率
2. 调整momentum一般为0.9 ，调整层数和minibatch size
3. 学习率衰减率/adam的beta1(0.9) beta2(0.999),gama(10^(-8) )



## 调参策略
1. 不要使用网格法，而用随机选取法（不同参数对结果有不同的优先级）
2. 又粗到细搜索参数

 **随机选取法**
不是均匀选取，而应该有一定的尺度，比如learning_rate的选择范围是0.0001～1,如果随机均匀选取，那么在0.0001～0.01采样数占整体的1%， 这时候应该用log形式的采样方法，如
```python
r = -4*np.random.rand()
alpha = np.square(10,r)
```
那么alpha的取值范围在10^-4~1之间，且从0.0001到0.001之间样本概率是1/4

**熊猫调参法，鱼子酱调参法**
与计算资源数量有关

## BN层的真相

BN层最原始的做法是将数据映射成均值为0方差为1的分布，但其实后面还有操作，gamma,beta两个参数来映射到其它的分布。所以有BN则前面的卷积层就不需要偏置项了。

BN层相当于在训练中的一个正则化，因为使用mini batch算出的平均值和方差跟整体的不一样，相当于加了一些噪声。所以batch_size越大，这个正则化的效果就不明显。

在测试阶段没有平均值和方差，就使用指数平均的 方式来计算。

## 为什么训练无法收敛 
一直保持 Loss: 2.303, Train acc 0.10, Test acc 0.10

可以从以下几个参数考虑：第一层的kernel_size，batch_size，learning_rate等


## 训练的基本流程 

1. 数据处理
2. 定义网络结构
3. 配置solver参数
4. 训练：如 caffe train -solver solver.prototxt -gpu 0

## 冻结层 

冻结一层不参与训练：设置其blobs_lr=0


## caffe训练时Loss变为nan的原因

**1. 梯度爆炸**
判定方法：

观察log，注意每一轮迭代后的loss。loss随着每轮迭代越来越大，最终超过了浮点型表示的范围，就变成了NaN。

解决方法：

 减小solver.prototxt中的base_lr，至少减小一个数量级。如果有多个loss layer，需要找出哪个损失层导致了梯度爆炸，并在train_val.prototxt中减小该层的loss_weight

**2. 不当的输入**
原因：输入中就含有NaN

措施：重整你的数据集，确保训练集和验证集里面没有损坏的图片。


## 为什么Caffe中引入了这个inner_num，inner_num等于什么

从FCN全卷积网络的方向去思考。FCN中label标签长度=图片尺寸 
caffe引入inner_num使得输入image的size可以是任意大小，innuer_num大小即为softmax层输入的heightxwidth

## BatchNorm层是否支持in place运算，为什么？

BN是对输入那一层做归一化操作，要对每个元素-均值/标准差，且输入输出规格相当，是可以进行in place

## softmax问题
在实际使用中，efj 常常因为指数太大而出现数值爆炸问题，两个非常大的数相除会出现数值不稳定问题，因此我们需要在分子和分母中同时进行以下处理：

$$\frac{e^{f_{y_i}}}{\sum_j e^{f_j}} = \frac{Ce^{f_{y_i}}}{C\sum_j e^{f_j}} = \frac{e^{f_{y_i}+logC}}{\sum_j e^{f_j+logC}}$$

其中C 的设置是任意的，在实际变成中，往往把C设置成

$$logC = -max f_j$$




[caffe+报错︱深度学习参数调优杂记+caffe训练时的问题+dropout/batch Normalization
](http://blog.csdn.net/sinat_26917383/article/details/54232791)


