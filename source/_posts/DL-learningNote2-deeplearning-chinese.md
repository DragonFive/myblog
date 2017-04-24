---
title: 深度学习中译本-节选

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

# 前言
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
确定 $ Ax=b$是否有解相当于确定向量b是否在A列向量的**生成子空间中**。又叫A的**值域**

如果一个矩阵的列空间涵盖整个 $R^m$，那么该矩阵必须包含至少一组 m 个线性无关的向量。这是式 (2.11) 对于每一个向量 b 的取值都有解的充分必要条件。值得注意的是，
这个条件是说该向量集恰好有 m 个线性无关的列向量，而不是至少 m 个。

一个列向量线性相关的**方阵**被称为 **奇异的**（singular）。
### 范数
范数（包括 $L^p$ 范数）是将**向量映射到非负值**的函数。直观上来说，向量 x 的范数衡量从原点到点 x 的距离。比如L2范数衡量的就是欧氏距离

当机器学习问题中**零和非零元素之间的差异非常重要**时，通常会使用 L1 范数。特别是求梯度的时候。

最大范数$L^{\infty }$，衡量的是向量中具有最大幅值的元素的绝对值：$\left \| x \right \|_\infty =max_i|x_i|$

衡量矩阵的大小需要用到Frobenius范数：
$$\left \| A \right \|_{F }=  \sqrt{ \sum_{i,j}A_{i,j}^2} = \sqrt{Tr(AA\top) }$$

.其类似于向量的$L^2$.，F范数可以用于定义最小平方误差。

### 特殊类型的矩阵和向量 
**对角矩阵**是只在主对角线上含有非零元素的矩阵。单位矩阵就是对角阵。对角阵受欢迎主要是因为其乘法计算很高效。计算乘法$ diag(v)x $，只需要把x中的每个元素xi放大vi倍就可以了。并且对角方阵的逆矩阵。


**正交矩阵**是指行向量和列向量都分别标准正交(都是单位向量然后各自正交)的方阵：
$A^\top A = I$ 这以为着 $A^{-1} = A^\top$.
所以正交矩阵收到关注是因为求逆代价小。

### 特征分解
特征分解是广泛使用的矩阵分解之一，即我们将矩阵分解成一组特征向量和特征值。
$Av = \lambda v$

**矩阵分解**，假设A有n个线性无关的特征向量{v1,v2,..vn},它们对应的特征值是{$\lambda_1,\lambda_2,...\lambda_n$},把每一个特征向量做一个列向量，然后组合起来形成一个矩阵，然后把相应的特征值组成对角阵。就是特征分解，把特征向量进行归一化（L2范数为1，单位向量），可以记做
$ A = Vdiag(\lambda)V^{-1} $
因为我们这里只讨论实对称矩阵，所以A的特征向量组成的矩阵就是正交矩阵，所以求逆就是求转置。
$ A = Vdiag(\lambda) V^\top$

矩阵是**奇异的**，当且仅当含有零特征向量。实对称矩阵的特征分解也可以用于优化二次方程：
$f(x)=x^{\top}A x$，其中限制
$ \left \| x \right \|_2 = 1 $，

当$x$是$A$的特征值的时候，$f$将返回对应的特征值，f的**最大值是最大的特征值**，最小值是最小的特征值。

所有特征值都是正数的矩阵被称为**正定**，所有特征值都是非负数的矩阵被称为**半正定**。半正定矩阵受到关注是因为它们保证$x^{\top}Ax>=0$。


### 奇异值分解  

**奇异值分解，singular value decomposition,SVD**，将矩阵分解为**奇异向量**和奇异值。每个实数矩阵都有奇异值分解，但不一定有特征分解。

$ A = UDV^{\top} $

A是mxn矩阵，则U是mxm正交矩阵，V是nxn正交矩阵 ，D是mxn对角矩阵（不一定是方阵）
SVD分解可以把矩阵求逆拓展到非方矩阵上面。

### 行列式

行列式的绝对值可以衡量矩阵参与矩阵乘法后空间扩大或者缩小了多少。如果行列式是0，那么空间至少沿着某一维完全收缩了，使其失去了所有的体积。

![enter description here][4]

## 概率与信息论 

概率分布用来描述随机变量在每一个可能取值的可能性的大小。

### 边缘概率

离散变量：$P(X=x) = \sum_y P(X=x,Y=y) $.
连续变量：$p(x) = \int p(x,y)dy.$

**条件概率的链式法则**：$P(a,b,c) = P(a|b,c)P(b|c)P(c) $
### 协方差
两个变量的协方差如果是正的，那么两个变量都倾向于同时取得相对较大的值。

中心极限定理表明很多独立随机变量的和近似服从**正态分布**，正态分布是对模型加入的先验知识量最少的分布。

### multinouli分布

多努力分布，又叫范畴分布，是指由一个随机变量在多个分类上的分布，与伯努利的区别在于伯努利指的是两个类。
伯努利函数的概率用sigmoid函数来预测，而多努力函数是用softmax函数与做预测的。

$ softmax(x)_i = \frac{exp(x_i)}{\sum _{j=1}^nexp(x_j)} $ 

[softmax的理解与应用](http://blog.csdn.net/supercally/article/details/54234115)
### 常用函数的有用性质

**logistic sigmoid函数**：$ \sigma(x) = \frac{1}{1+exp(-x)} $，在变量的绝对值很大的情况下回出现饱和现象，这时候就会对输入的微小变化不敏感。

另一个经常遇到的函数是**softplus函数**：$\zeta(x) = log(1+exp(x))$，它的值域是$(0,\infty)$,他是对max函数的平滑max(0,x).


![softplus函数][5]

一些性质：$\sigma(x) = \frac{exp(x)}{exp(x)+exp(0)}$
$\frac{d}{dx}\sigma(x)=\sigma(x)(1-\sigma(x))$
$1-\sigma(x)=\sigma(-x)$
$log\sigma(x)=-\zeta(-x)$


## 数值计算

### 梯度下降法

![三种临界点][6]

**鞍点**是拐点的一种。其二阶导数为0

###  jacobian矩阵和hessian矩阵
f的Jacobian矩阵定义为 $ J_{i,j} = \frac{\partial }{\partial x_j}f(x)_i $
当我们的函数有多维输入时，把二阶导数合并成一个矩阵，称为**Hessian**矩阵。$ H(f)(x)_{i,j} = \frac{\partial ^2}{\partial x_i \partial x_j} f(x) $

Hessian矩阵大多数都是实对称矩阵。所以可以进行特征分解并写成下面的表达式：$d^\top Hd$,当d是H的特征向量时，表达式的值为d对应的特征值，也就是d这个方向上的二阶导数。当d是H的一个特征向量时，这个方向的二阶导数就是对应的特征值。对于其他的方向d，**方向二阶导数就是所有特征值的加权平均**，权重在0和1之间，且与d夹角越小的特征向量的权重越大。最大特征值来确定最大二阶导数，最小特征值确定最小二阶导数。



在临界点（一阶偏导都为0处），我们通过检测Hessian的特征值来判断该临界点是一个局部极大点、局部极小点还是鞍点。当**Hessian是正定**的（所有特征值都是正的），则该临界点是局部极小点。因为方向二阶导数在任何方向都是正的，当Hessian矩阵是负定的，这个点就是局部极大点。如果hessian的特征值中至少一个是正的且至少一个是负的，








# 参考资料
[bengio 深度学习中译本](https://exacity.github.io/deeplearningbook-chinese/0)

名称  | 京东报价 | 李季报价
------------- | ------------- | -------------
vCPU：I7 5930k  |  4299  | 4850
v主板华硕x99 deluxe II  |  3999  | 4350
v内存条：金士顿Fury DDR4  2400 8Gx4 | 1796| 2000
v固态硬盘：三星850EVO 250G M.2接口|  669  | 795
v机械硬盘：希捷2T7200 SATA3|  449  |  515
机箱 ttthernaltake core v51 台式机中塔机箱| 649 | 700
主机电源：海韵x1250w 全模电源|  1999  |2200
v显卡  titan x pascal |  11100  |  14400
v显示器：戴尔U2414H|  1549 |1700
键鼠罗技MK520| 239  |  280
其他|  157 |185
合计| 26905 |31975





# 深度学习学习资料


[爱可可爱生活搬运的cs231N课程](http://space.bilibili.com/23852932/#!/video)

[cs231N课程笔记翻译](https://zhuanlan.zhihu.com/p/21930884?refer=intelligentunit)

[网友的cs231n课程作业与课程内容回顾](http://www.jianshu.com/p/004c99623104)

[cs231n课程课件](http://pan.baidu.com/s/1pKsTivp#list/path=%2F)

[李宏毅2017课程,深度学习偏语音](http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS17.html)

[李宏毅2016课程](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML16.html)


[UFLDL教程中文版](http://ufldl.stanford.edu/wiki/index.php/UFLDL%E6%95%99%E7%A8%8B)

[杨立坤的deeplearning](https://github.com/exacity/deeplearningbook-chinese)


[tensorlayer中文版](http://tensorlayercn.readthedocs.io/zh/latest/)

[莫凡 tensorflow](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/1-2-install/)

[tensorfly](http://www.tensorfly.cn/home/)

[Neural Networks and Deep Learning中文翻译](https://hit-scir.gitbooks.io/neural-networks-and-deep-learning-zh_cn/content/)

[一文弄懂神经网络中的反向传播法——BackPropagation](http://www.cnblogs.com/charlotte77/p/5629865.html)


[皮果提的深度学习笔记](http://blog.csdn.net/peghoty/article/category/1451403)

[邹晓艺汇总的深度学习学习资料](http://blog.csdn.net/zouxy09/article/details/8782018)

[Deep Learning（深度学习）学习笔记整理系列之（一）](http://blog.csdn.net/zouxy09/article/details/8775360)

[cvpr论文阅读笔记](http://www.cnblogs.com/wangxiaocvpr/)

[CNN-tracking-文章导读](http://blog.csdn.net/ben_ben_niao/article/details/51315000)

  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1491645781301.jpg "1491645781301"
  [2]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1491647505456.jpg "1491647505456"
  [3]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1491648776897.jpg "1491648776897"
  [4]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1491899762309.jpg "1491899762309"
  [5]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1491965737743.jpg "1491965737743"
  [6]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1491968682477.jpg "1491968682477"
