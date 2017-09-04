---
title: 机器学习算法小结与对比 

date: 2017/7/30 12:04:12

categories:
- 机器学习
tags:
- 机器学习
- 深度学习
- 神经网络
---

# SVM与LR的区别 


## 从模型解决问题的方式来看
Linear SVM直观上是trade-off两个量 
1. a large margin，就是两类之间可以画多宽的gap ；不妨说是正样本应该在分界平面向左gap/2（称正分界），负样本应该在分解平面向右gap/2（称负分界）
2. L1 error penalty，对所有不满足上述条件的点做L1 penalty

给定一个数据集，一旦完成Linear SVM的求解，所有数据点可以被归成两类
1. 一类是落在对应分界平面外并被正确分类的点，比如落在正分界左侧的正样本或落在负分界右侧的负样本
2. 第二类是落在gap里或被错误分类的点。

假设一个数据集已经被Linear SVM求解，那么往这个数据集里面增加或者删除更多的一类点并不会改变重新求解的Linear SVM平面。不受数据分布的影响。

求解LR模型过程中，**每一个数据点对分类平面都是有影响的**，它的影响力远离它到分类平面的距离指数递减。换句话说，LR的解是**受数据本身分布**影响的。在实际应用中，如果数据维度很高，LR模型都会配合参数的L1 regularization。



# reference

[Linear SVM 和 LR 有什么异同？](https://www.zhihu.com/question/26768865)

[SVM和logistic回归分别在什么情况下使用?](https://www.zhihu.com/question/21704547)