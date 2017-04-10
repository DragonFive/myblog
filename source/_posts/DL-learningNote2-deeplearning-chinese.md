
title: 深度学习中译本-笔记

date: 2017/4/8 17:38:58

categories:
- 计算机视觉
tags:
- deeplearning
- 深度学习
---
[TOC]


![AI][1]

<!--more-->

# 第一章前言
一个人的日常生活需要关于世界的巨量知识。很多这方面的知识是**主观的**、直观的，因此很难通过**形式化的方式表达**清楚。计算机需要获取同样的知识才能表现出智能。
AI 系统需要具备自己**获取知识的能力**，即从原始数据中提取模式的能力。这种能力被称为 机器学习。

**对表示的依赖**都是一个普遍现象。在计算机科学中，如果数据集合被精巧地结构化并被智能地索引，那么诸如搜索之类的操作的处理速度就可以成指数级地加快。

使用机器学习来**发掘表示本身**，而不仅仅把表示映射到输出。这种方法我们称之为 **表示学习**（representation learning）。学习到的表示往往比手动设计的表示表现得更好。并且它们只需**最少的人工干预**，就能让AI系统迅速适应新的任务。表示学习算法的典型例子是 自编码器(autoencoder)。

深度学习（deep learning）通过其他较简单的表示来表达复杂表示，解决了表示学习中的核心问题。下图展示了深度学习系统如何通过**组合较简单的概念**，获得不同层次的抽象（例如转角和轮廓，它们转而由边线定义）来表示图像中人的概念。第一层可以轻易地通过**比较相邻像素的亮度来识别边缘**。有了第一隐藏层描述的边缘，第二隐藏层可以容易地**搜索可识别为角和扩展轮廓**的边集合。给定第二隐藏层中关于角和轮廓的图像描述，第三隐藏层可以找到轮廓和角的特定集合来检测特定对象的整个部分。最后，根据图像描述中包含的对象部分，可以识别图像中存在的对象。

![enter description here][2]


深度学习与其他学习的区别。
![深度学习与其他的区别][3]
# 应用数学与机器学习基础

## 线性代数部分
坐标超过两维度的数组称为**张量（tensor）**，一个数组中的元素分布在若干维坐标的规则网络中。


一组向量的**生成子空间**是原始向量线性组合后所能抵达的点的集合。
确定 `!$ Ax=b$`是否有解相当于确定向量b是否在A列向量的**生成子空间中**。又叫A的**值域**

如果一个矩阵的列空间涵盖整个 `!$R^m$`，那么该矩阵必须包含至少一组 m 个线性无关的向量。这是式 (2.11) 对于每一个向量 b 的取值都有解的充分必要条件。值得注意的是，
这个条件是说该向量集恰好有 m 个线性无关的列向量，而不是至少 m 个。

一个列向量线性相关的**方阵**被称为 **奇异的**（singular）。

范数（包括 `!$L^p$` 范数）是将**向量映射到非负值**的函数。直观上来说，向量 x 的范数衡量从原点到点 x 的距离。比如L2范数衡量的就是欧氏距离

当机器学习问题中**零和非零元素之间的差异非常重要**时，通常会使用 L1 范数。特别是求梯度的时候。



# 参考资料
[bengio 深度学习中译本](https://exacity.github.io/deeplearningbook-chinese/0)


# 深度学习学习资料

[李宏毅2017课程](http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS17.html)

[李宏毅2016课程](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML16.html)

[爱可可爱生活搬运的课程](http://space.bilibili.com/23852932/#!/channel/detail?cid=11583)

[UFLDL教程中文版](http://ufldl.stanford.edu/wiki/index.php/UFLDL%E6%95%99%E7%A8%8B)

[杨立坤的deeplearning](https://github.com/exacity/deeplearningbook-chinese)

[莫凡 tensorflow](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/1-2-install/)

[tensorfly](http://www.tensorfly.cn/home/)

[Neural Networks and Deep Learning中文翻译](https://hit-scir.gitbooks.io/neural-networks-and-deep-learning-zh_cn/content/)

[一文弄懂神经网络中的反向传播法——BackPropagation](http://www.cnblogs.com/charlotte77/p/5629865.html)


[皮果提的深度学习笔记](http://blog.csdn.net/peghoty/article/category/1451403)

[邹晓艺汇总的深度学习学习资料](http://blog.csdn.net/zouxy09/article/details/8782018)

[Deep Learning（深度学习）学习笔记整理系列之（一）](http://blog.csdn.net/zouxy09/article/details/8775360)


  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1491645781301.jpg "1491645781301"
  [2]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1491647505456.jpg "1491647505456"
  [3]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1491648776897.jpg "1491648776897"