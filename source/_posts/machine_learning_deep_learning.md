---
title: 机器学习算法小结与对比 

date: 2017/8/10 12:04:12

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


## 两者的区别

两个模型对**数据和参数**的敏感程度不同，Linear SVM比较依赖penalty的系数和**数据表达空间的测度**，而（带正则项的）LR**比较依赖对参数做L1 regularization的系数**。但是由于他们或多或少都是线性分类器，所以实际上对低维度数据overfitting的能力都比较有限，相比之下对高维度数据，LR的表现会更加稳定，为什么呢？因为Linear SVM在计算margin有多“宽”的时候是依赖数据表达上的距离测度的，换句话说如果这个测度不好（badly scaled，这种情况在高维数据尤为显著），所求得的所谓Large margin就没有意义了，这个问题即使换用kernel trick（比如用Gaussian kernel）也无法完全避免。所以使用Linear SVM之前一般都需要先对数据做normalization，而求解LR（without regularization）时则不需要或者结果不敏感。


Linear SVM和LR都是线性分类器
Linear SVM不直接依赖数据分布，分类平面不受一类点影响；**LR则受所有数据点的影响，如果数据不同类别strongly unbalance一般需要先对数据做balancing**。
Linear SVM**依赖数据表达的距离测度，所以需要对数据先做normalization**；LR不受其影响
Linear SVM依赖penalty的系数，实验中需要做validation
Linear SVM和LR的performance都会收到outlier的影响，其敏感程度而言，谁更好很难下明确结论。

**balance的方法**

1. 调整正、负样本在求cost时的权重，比如按比例加大正样本cost的权重。然而deep learning的训练过程是on-line的因此你需要按照batch中正、负样本的比例调整。
2. 做训练样本选取：如hard negative mining，只用负样本中的一部分。
3. 做训练样本选取：如通过data augmentation扩大正样本数量。


## 方法的选择 



在Andrew NG的课里讲到过：1. 如果Feature的数量很大，跟样本数量差不多，这时候选用LR或者是Linear Kernel的SVM2. 如果Feature的数量比较小，样本数量一般，不算大也不算小，选用SVM+Gaussian Kernel3. 如果Feature的数量比较小，而样本数量很多，需要手工添加一些feature变成第一种情况

当你的数据非常非常非常非常非常大然后完全跑不动SVM的时候，跑LR。多大算是非常非常非常非常非常非常大？ 比如几个G，几万维特征，就勉强算大吧...而实际问题上几万个参数实在完全不算个事儿，太常见了。随随便便就得上spark。读一遍数据就尼玛老半天，一天能训练出来的模型就叫高效了。所以在新时代，LR其实反而比以前用的多了=. =

# 机器学习算法选择 
[](https://www.zhihu.com/question/26726794)
随机森林平均来说最强，但也只在9.9%的数据集上拿到了第一，优点是鲜有短板。SVM的平均水平紧随其后，在10.7%的数据集上拿到第一。神经网络（13.2%）和boosting（~9%）表现不错。数据维度越高，随机森林就比AdaBoost强越多，但是整体不及SVM[2]。数据量越大，神经网络就越强。

## 贝叶斯
是相对容易理解的一个模型，至今依然被垃圾邮件过滤器使用。

## K近邻

典型的例子是KNN，它的思路就是——对于待判断的点，找到离它最近的几个数据点，根据它们的类型决定待判断点的类型。

它的特点是完全跟着数据走，没有数学模型可言。

## 决策树 
决策树的特点是它总是在沿着特征做切分。随着层层递进，这个划分会越来越细。

因为它能够生成清晰的基于特征(feature)选择不同预测结果的树状结构

## 随机森林

[ 天池离线赛 - 移动推荐算法（四）：基于LR, RF, GBDT等模型的预测](http://blog.csdn.net/Snoopy_Yuan/article/details/75808006)

它首先随机选取不同的特征(feature)和训练样本(training sample)**bagging**，生成大量的决策树，然后综合这些决策树的结果来进行最终的分类。

随机森林在现实分析中被大量使用，它相对于决策树，在准确性上有了很大的提升

适用场景：数据维度相对低（几十维），同时对准确性有较高要求时。




# LR相关问题

## LR与BP
[BP神经网络是否优于logistic回归？](https://www.zhihu.com/question/27823925)
首先，神经网络的最后一层，也就是输出层，是一个 Logistic Regression （或者 Softmax Regression ），也就是一个线性分类器，中间的隐含层起到特征提取的作用，把隐含层的输出当作特征，然后再将它送入下一个 Logistic Regression，一层层变换。

神经网络的训练，实际上就是同时训练特征提取算法以及最后的 Logistic Regression的参数。为什么要特征提取呢，因为 Logistic Regression 本身是一个线性分类器，所以，通过特征提取，我们可以把原本线性不可分的数据变得线性可分。要如何训练呢，最简单的方法是**（随机，Mini batch）梯度下降法**

## LR为什么使用sigmoid函数 

源于sigmoid，或者说exponential family所具有的最佳性质，即**maximum entropy**的性质。maximum entropy给了logistic regression一个很好的数学解释。为什么maximum entropy好呢？entropy翻译过来就是熵，所以maximum entropy也就是最大熵。熵用在概率分布上可以表示这个**分布中所包含的不确定度**，熵越大不确定度越大。均匀分布熵最大，因为基本新数据是任何值的概率都均等。而我们现在关心的是，给定某些假设之后，熵最大的分布。也就是说这个分布应该在满足我假设的前提下越均匀越好。比如大家熟知的正态分布，正是假设已知mean和variance后熵最大的分布。首先，我们在建模预测 Y|X，并认为 Y|X 服从bernoulli distribution，所以我们只需要知道 P(Y|X)；其次我们需要一个线性模型，所以 P(Y|X) = f(wx)。接下来我们就只需要知道 f 是什么就行了。而我们可以通过最大熵原则推出的这个 f，就是sigmoid。

在logistic regression （LR）中，这个目标是什么呢？最大化条件似然度。考虑一个二值分类问题，训练数据是一堆（特征，标记）组合，（x1,y1), (x2,y2), .... 其中x是特征向量，y是类标记（y=1表示正类，y=0表示反类）。LR首先定义一个条件概率p(y|x；w）。 **p(y|x；w）表示给定特征x，类标记y的概率分布，其中w是LR的模型参数（一个超平面）**。有了这个条件概率，就可以在训练数据上定义一个似然函数，然后通过最大似然来学习w。这是LR模型的基本原理。

# SVM相关问题 

[解密SVM系列（一）：关于拉格朗日乘子法和KKT条件](http://blog.csdn.net/on2way/article/details/47729419)


## 拉格朗日乘子法 和KKT条件

### 凸函数

前提条件凸函数：下图左侧是凸函数。

![左侧是凸函数][1]

凸的就是开口朝一个方向（向上或向下）。更准确的数学关系就是： 

$$\dfrac{f(x_1)+f(x_2)}{2}>f(\dfrac{x_1+x_2}{2})$$
或者
$$\\ \dfrac{f(x_1)+f(x_2)}{2}<f(\dfrac{x_1+x_2}{2})$$

<对于凸问题，你去求导的话，是不是只有一个极点，那么他就是最优点，很合理。

### 等式条件约束 

当带有约束条件的凸函数需要优化的时候，一个带等式约束的优化问题就通过拉格朗日乘子法完美的解决了。

$$min \quad f = 2x_1^2+3x_2^2+7x_3^2 \\s.t. \quad 2x_1+x_2 = 1 \\ \quad \quad \quad 2x_2+3x_3 = 2$$

可以使用

$$min \quad f = 2x_1^2+3x_2^2+7x_3^2 +\alpha _1(2x_1+x_2- 1)+\alpha _2(2x_2+3x_3 - 2)$$

这里可以看到与α1,α2相乘的部分都为0，根原来的函数是等价的。所以α1,α2的取值为全体实数。现在这个优化目标函数就没有约束条件了吧。然后求导数。

### 不等式约束与KKT条件 
任何原始问题约束条件无非最多3种，等式约束，大于号约束，小于号约束，而这三种最终通过将约束方程化简化为两类：约束方程等于0和**约束方程小于0**。


$$min \quad f = x_1^2-2x_1+1+x_2^2+4x_2+4 \\s.t. \quad x_1+10x_2 > 10 \\ \quad \quad \quad 10 x_1-10x_2 < 10$$

现在将约束拿到目标函数中去就变成： 
$$L(x,\alpha) = f(x) + \alpha_1g1(x)+\alpha_2g2(x)\\ =x_1^2-2x_1+1+x_2^2+4x_2+4+ \alpha_1(10-x_1-10x_2 ) +\\\alpha_2(10x_1-x_2 - 10)$$

其中g是不等式约束，h是等式约束（像上面那个只有不等式约束，也可能有等式约束）。那么KKT条件就是函数的最优值必定满足下面条件：

(1) L对各个x求导为零； 
(2) h(x)=0; 
(3) $\sum\alpha_ig_i(x)=0，\alpha_i\ge0$

第三个式子不好理解，因为我们知道在约束条件变完后，所有的g(x)<=0，且αi≥0，然后求和还要为0，无非就是告诉你，**要么某个不等式gi(x)=0,要么其对应的αi=0**。那么为什么KKT的条件是这样的呢？

## SVM的原问题和对偶问题

原问题
$$\max_{w, b}\frac{1}{||w||}=\min_{w,b}\frac{1}{2}||w||^{2}\\
\text{s,t}\ \ \ \  y_{i}(w^{T}x_{i}+b)\geq 1, i\in {\{1,...,N\}}$$

对偶问题
$$\max_{\alpha, \beta, \alpha_{i}\geq 0}\min_{w}L(w, \alpha, \beta)=\max_{\alpha, \alpha_{i}\geq 0}\min_{w, b}\{\frac{1}{2}||w||^{2}-\sum_{i=1}^{N}\alpha_{i}[y_{i}(w^{T}x_{i}+b)-1]\}$$

## SVM解决过拟合的方法

决定SVM最优分类超平面的恰恰是那些占少数的支持向量，如果支持向量中碰巧存在异常点就会过拟合，解决的方法是加入松弛变量。

另一方面从损失函数角度看，引入了L2正则。

为什么要把原问题转换为对偶问题？

因为原问题是凸二次规划问题，转换为对偶问题更加高效。

为什么求解对偶问题更加高效？

因为只用求解alpha系数，而alpha系数只有支持向量才非0，其他全部为0.

alpha系数有多少个？

样本点的个数

L1还可以用来选择特征

A 为什么**L1可以用来选择特征**

B 因为L1的话会把某些不重要的特征压缩为0

A 为什么L1可以把某些特征压缩为0

B 因为（画图）L1约束是正方形的，经验损失最有可能和L1的正方形的顶点相交，L1比较有棱角。所以可以把某些特征压缩为0





# reference

[Linear SVM 和 LR 有什么异同？](https://www.zhihu.com/question/26768865)

[SVM和logistic回归分别在什么情况下使用?](https://www.zhihu.com/question/21704547)

[百度 – 机器学习面试](http://www.100mian.com/mianshi/baidu/49214.html)

[ svmw问题整理](http://blog.csdn.net/rosenor1/article/details/52318454)

[各种机器学习的应用场景分别是什么？例如，k近邻,贝叶斯，决策树，svm，逻辑斯蒂回归](https://www.zhihu.com/question/26726794)

[机器学习面试问题汇总](http://www.cnblogs.com/hellochennan/p/6654084.html)

[机器学习面试](http://blog.csdn.net/u010496169/article/category/6984158)

[如何准备机器学习工程师的面试 ？](https://www.zhihu.com/question/23259302/answer/174467341)

[ 天池离线赛 - 移动推荐算法（四）：基于LR, RF, GBDT等模型的预测](http://blog.csdn.net/Snoopy_Yuan/article/details/75808006)

  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1504663655806.jpg