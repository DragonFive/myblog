
title: DL从入门到放弃

date: 2017/3/20 22:04:12

categories:
- 计算机视觉
tags:
- deeplearning
- 梯度下降法
- 正则化
- 激活函数
---
[TOC]
Yann Lecun, Geoffrey Hinton, Yoshua Bengio, Andrew Ng

![四大巨头][1]
<!--more-->
[参考资料-nature上的DL综述](http://www.nature.com/nature/journal/v521/n7553/pdf/nature14539.pdf)

Conventional machine-learning techniques were **limited** in their ability to **process natural data in their raw form**  传统的机器学习不能处理原始形式的自然数据， 

constructing a pattern-recognition or machine-learning system required
**careful engineering and considerable domain expertise** to design a **feature extractor** that transformed the raw data (such as the pixel values of an image) into a suitable **internal representation** or** feature vector   ** 需要一些工程学和相当领域的专业知识设计特征提取器来把原始数据转为合适的内部表示或者特征向量。

**Representation learning** is a set of methods that allows a machine to be fed with** raw data** and to automatically discover the** representations** needed for detection or classification. Deep-learning methods are representation-learning methods with **multiple levels of representation**, obtained by composing simple but **non-linear modules** that each **transform** the representation at one level (starting with the raw input) into a representation at a higher, slightly **more abstract level**. With the composition of enough such transformations, very complex functions can be learned.

深度学习使用非线性模块对原始数据进行多层级的转换，得到多层次的数据表示和更高层的抽象，能够学习非常复杂的函数。

 learned features in the first layer of representation typically represent the presence or absence of **edges at particular orientations and locations** in the image. The second layer typically detects motifs by spotting** particular arrangements of edges**, regardless of small variations in the edge positions. The third layer may **assemble motifs into larger combinations** that correspond to parts of familiar objects, and subsequent layers would detect objects as combinations of these parts. 
 
 
 To properly adjust the **weight vector**, the learning algorithm computes a **gradient vector** that, for each weight, indicates by what amount the error would increase or decrease if the weight were increased by a tiny amount. The weight vector is then **adjusted in the opposite direction** to the gradient vector
 
 
 shallow classifiers require a** good feature extractor** that solves the **selectivity–invariance** dilemma — one that produces representations that are selective to the aspects of the image that are important for discrimination, but that are invariant to irrelevant aspects  好的特征描述堆无关的方面具有不变性
 
 make classifiers more powerful, one can use **generic non-linear features**, as with kernel methods20, but generic features such as those **arising with the Gaussian kernel** do not allow the learner to generalize well far from the training examples
 
由高斯核产生的通用的非线性特征对远离训练样本的数据扩展性差 

A deep-learning **architecture** is a multilayer stack of simple **modules**, all (or most) of which are subject to learning, and many of which** compute non-linear input–output mappings**. Each module in the stack transforms its input to** increase both the selectivity and the invariance** of the representation

能同时增加模型的区分能力和无关性 

 multilayer architectures can be trained by simple **stochastic gradient descent**. As long as the modules are relatively smooth functions of their inputs and of their internal weights, one can compute gradients using the **backpropagation** procedure
 
 有了反传机制，就能用简单的SGD来训练多层网络架构。反传机制计算目标函数关于多层模型权重的梯度是基于链式法则算的。
 
 The key insight is that the derivative (or gradient)of the objective with respect to the input of a module can be computed by** working backwards from the gradient with respect to the output of that module**
 
 ## 梯度下降法
 目标函数关于模型输入的梯度可以通过关于输出的梯度逆向算出。反正在用损失函数求权重偏导的时候由于**激活函数**的存在，导致**梯度消失**. 参考自[深度学习如何入门？ - 回答作者: Deeper](http://zhihu.com/question/26006703/answer/135825424)
 
 ![激活函数造成梯度消失][3]

[梯度下降优化算法综述 An overview of gradient descent optimization algorithms](http://blog.csdn.net/heyongluoyao8/article/details/52478715)

梯度下降算法是通过沿着**目标函数**J(θ)参数θ∈ℜ的梯度(一阶导数)**相反方向**−∇θJ(θ)来不断更新模型参数来到达**目标函数的极小值点（收敛）**，更新步长为η。

**随机梯度下降**最大的缺点在于每次更新可能并**不会按照正确**的方向进行，因此可以带来**优化波动**(扰动) 波动的特点可能会使得优化的方向从当前的局部极小值点跳到另一个更好的局部极小值点，这样便可能对于**非凸函数**，最终收敛于一个较好的局部极值点，甚至全局极值点。
![随机梯度下降扰动][4]
相对于随机梯度下降，**Mini-batch梯度下降**降低了收敛波动性，即降低了参数更新的方差，使得更新更加稳定。相对于全量梯度下降，其提高了每次学习的速度。并且其不用**担心内存瓶颈从而可以利用矩阵运算进行高效计算**。一般而言每次更新**随机选择[50,256]个样本**进行学习，但是也要根据具体问题而选择，实践中可以进行多次试验，选择一个更新速度与更次次数都较适合的样本数。mini-batch梯度下降常用于神经网络中。

而梯度下降法虽然效果很好但还是存在一些问题 

- 学习速率值的选取较难、
- 学习速率调整难、
- 不同参数用不同的学习速率(很少出现的特征的速率要打)、
- 容易陷入局部极值点

## 梯度下降优化方法（深度学习）
牛顿法也能用，但不能用在深度学习中，这些方法都是替代学习率的
### Momentum 动量
[On the momentum term in gradient descent learning algorithms.](http://www.sciencedirect.com/science/article/pii/S0893608098001166)



解决SGD在某些极值点附近**震荡**，导致收敛速度慢，可以让SGD有机会离开局部极值点达到更好的极值点。在更新模型参数时，对于那些当前的梯度方向与上一次**梯度方向相同的参数，那么进行加强**，即这些方向上更快了；对于那些当前的梯度方向与上一次梯度方向不同的参数，那么进行削减，即这些方向上减慢了。因此可以获得**更快的收敛速度与减少振荡**。

### NAG 
不仅增加了动量项，并且在计算参数的梯度时，在损失函数中减去了动量项.

![动量与NAG][5]
假设动量因子参数γ=0.9，首先计算当前梯度项，如上图小蓝色向量，然后加上动量项，这样便得到了大的跳跃，如上图大蓝色的向量。这便是只包含动量项的更新。而NAG首先来一个大的跳跃（动量项)，然后加上一个小的使用了动量计算的当前梯度（上图红色向量）进行修正得到上图绿色的向量。这样可以**阻止过快更新来提高响应**

上面两个方法每次更新的时候所有的模型参数都使用同一个学习速率，能够根据损失函数的斜率做到**自适应更新**来加速SGD的收敛。

### adagrad 
adagrad能够对每个参数**自适应不同的学习速率**，对稀疏特征，得到大的学习更新，对非稀疏特征，得到较小的学习更新，因此该优化算法适合处理**稀疏特征数据**
缺点在于需要计算参数梯度序列平方和，并且学习速率趋势是不断衰减最终达到一个非常小的值.

### adadelta和RMSprop

adadelta是对adagrad的一种扩展，为了降低adagrad中学习速率衰减过快问题 

### Adam

adaptive moment estimation 也是不同参数自适应不同学习速率的方法 

![深度学习各SGD优化方法在鞍点处的表现][6]

在鞍点（saddle points）处(即某些维度上梯度为零，某些维度上梯度不为零)，SGD、Momentum与NAG一直在鞍点梯度为零的方向上振荡，很难打破鞍点位置的对称性；Adagrad、RMSprop与Adadelta能够很快地向梯度不为零的方向上转移。


### 如何选择SGD优化器

如果你的数据特征是稀疏的，那么你最好使用自适应学习速率SGD优化方法(Adagrad、Adadelta、RMSprop与Adam)，因为你不需要在迭代过程中对学习速率进行人工调整。adam可能是目前最好的SGD优化方法。 

最近很多论文都是使用原始的SGD梯度下降算法，并且使用简单的学习速率**退火调整**（无动量项）。现有的已经表明：SGD能够收敛于最小值点，但是相对于其他的SGD，它可能花费的时间更长，并且依赖于鲁棒的初始值以及学习速率退火调整策略，并且容易**陷入局部极小值点，甚至鞍点**

## 更多的SGD策略

### Shuffling and Curriculum Learning  训练集随机洗牌与课程学
为了使得学习过程更加无偏，应该在每次迭代中随机**打乱训练集中的样本**。 
 另一方面，在很多情况下，我们是逐步解决问题的，而将训练集按照某个有意义的顺序排列会提高模型的性能和SGD的收敛性
### Batch normalization 
为了方便训练，我们通常会对参数按照0均值1方差进行初始化，随着不断训练，参数得到不同程度的更新，这样这些参数会失去0均值1方差的分布属性，这样会降低训练速度和放大参数变化随着网络结构的加深。 
Batch normalization[18]在每次mini-batch反向传播之后重新对参数进行0均值1方差标准化。这样可以**使用更大的学习速率，以及花费更少的精力**在参数初始化点上。Batch normalization充当着正则化、减少甚至消除掉Dropout的必要性。

### Early stopping 
在验证集上如果连续的多次迭代过程中损失函数不再显著地降低，那么应该提前结束训练

### Gradient noise 
在每次迭代计算梯度中加上一个高斯分布的随机误差 。会增加模型的**鲁棒性**，即使初始参数值选择地不好，并适合对特别深层次的负责的网络进行训练。其原因在于**增加随机噪声会有更多的可能性跳过局部极值点并去寻找一个更好的局部极值点**，这种可能性在深层次的网络中更常见。

## 机器学习中防止过拟合的处理方法 

stFunction，针对Octave而言，我们可以将这个函数作为参数传入到 fminunc 系统函数中（fminunc 用来求函数的最小值，将@costFunction作为参数代进去，注意 @costFunction 类似于C语言中的函数指针），fminunc返回的是函数 costFunction 在无约束条件下的最小值，即我们提供的代价函数 jVal 的最小值，当然也会返回向量 θ 的

[为什么正则化项就可以防止过拟合-邓子明](https://www.zhihu.com/question/20700829/answer/119314862)


如果发生了过拟合问题，我们应该如何处理？

**过多的变量（特征），同时只有非常少的训练数据**，会导致出现过度拟合的问题。因此为了解决过度拟合，有以下两个办法。
![过拟合了怎么办][7]

1. 尽量减少

[机器学习中防止过拟合的处理方法](http://blog.csdn.net/heyongluoyao8/article/details/49429629)

在统计学习中，假设数据满足**独立同分布（i.i.d**，independently and identically distributed），即当前已产生的数据可以对未来的数据进行推测与模拟。但是一般独立同分布的假设往往不成立，即数据的分布可能会发生变化（distribution drift），并且可能当前的**数据量过少，不足以对整个数据集进行分布估计**，因此往往需要防止模型过拟合，提高模型泛化能力。而为了达到该目的的最常见方法便是：**正则化**，即在对模型的**目标函数（objective function）或代价函数（cost function）加上正则项**。

为了防止过拟合，我们需要用到一些方法，如：**early stopping、数据集扩增（Data augmentation）、正则化（Regularization）、Dropout**等。

### early stopping 提早结束

Early stopping便是一种迭代次数截断的方法来防止过拟合的方法，即在模型对训练数据集迭代收敛之前停止迭代来防止过拟合。 
Early stopping方法的具体做法是，在每一个Epoch结束时（**一个Epoch集为对所有的训练数据的一轮遍历**）计算validation data的accuracy，**当accuracy不再提高时，就停止训练**。一般的做法是，在训练的过程中，**记录到目前为止最好的validation accuracy**，当连续10次Epoch（或者更多次）没达到最佳accuracy时，则可以认为accuracy不再提高了。此时便可以停止迭代了（Early Stopping）。这种策略也称为“No-improvement-in-n”，n即Epoch的次数，可以根据实际情况取，如10、20、30……

### 数据集扩增 

更多的数据有时候更优秀。但是往往条件有限，如人力物力财力的不足，而不能收集到更多的数据，需要采用一些计算的方式与策略在已有数据集上进行操作，来得到更多的数据。

- 复制原有数据并加上随机噪声
- 重采样
- 根据当前数据集估计数据分布参数，使用该分布产生更多数据等

### 正则化方法

[邹晓艺专栏 机器学习中的范数规则化之（一）L0、L1与L2范数 ](http://blog.csdn.net/zouxy09/article/details/24971995)
监督机器学习问题无非就是“**minimizeyour error while regularizing your parameters**”，也就是在规则化参数的同时最小化误差。**最小化误差是为了让我们的模型拟合我们的训练数据**，而规则化参数是防止我们的模型过分拟合我们的训练数据。

规则项的使用还可以约束我们的模型的特性。这样就可以将人对这个模型的先验知识融入到模型的学习当中，强行地让学习到的模型具有人想要的特性，例如**稀疏、低秩、平滑**等等。

一般有L1正则与L2正则 
最小化损失函数，就用梯度下降，最大化似然函数，就用梯度上升


监督学习可以看做最小化下面的目标函数：

`!$ w^* = arg min_w \sum_iL(y_i,f(x_i,f(x_ilw)))+\lambda\Omega(w) $`

其中第一项是损失函数，第二项是正则项或者叫惩罚项。

第一项Loss函数，如果是Square loss,那就是最小二乘；如果是Hinge-Loss就是SVM；如果是exp-Loss，那就是boosting了;如果是log-loss，那就是Logistic Regression。


#### L1正则

 L0范数是指向量中非0的元素的个数。L1范数是指向量中**各个元素绝对值之和**，也有个美称叫“**稀疏规则算子**”（**Lasso** regularization）。价值一个亿的问题：为什么L1范数会使权值稀疏？“**它是L0范数的最优凸近似，而且它比L0范数要容易优化求解。**”。实际上，还存在一个更美的回答：任何的规则化算子，如果他在**Wi=0的地方不可微，并且可以分解为一个“求和”的形式**，那么这个规则化算子就可以实现稀疏。

参数稀疏有什么好处呢？为什么要稀疏
1. 特征选择
2. 可解释性




L1正则是基于L1范数，在目标函数后面加上参数的L1范数和项，即**参数绝对值和**
`!$ C = C_0+\frac{\lambda}{n}\sum_w |W|$`
其中`!$ C_0$` 代表原始的代价函数，n是样本的个数，`!$\lambda $`就是正则项系数，权衡正则项与`!$C_0 $`的比重

在计算梯度时，w的梯度变为：
`!$ \frac{\partial c}{\partial \omega} = \frac{\partial c_0}{\partial  \omega}+\frac{\lambda}{n}sgn(\omega) $`
其中，sgn是符号函数，那么使用下式堆参数进行更新：
`!$ \omega = \omega + \alpha\frac{\partial c_0}{\partial  \omega}+\beta\frac{\lambda}{n}sgn(\omega) $`

在梯度下降法中：`!$ \alpha < 0， \beta <0 $` ,所以当w为正时，更新后w会变小，当w为负时更新后w会变大；因此L1正则项是为了使得那些原先处于零（即|w|≈0）附近的参数w往零移动，**使得部分参数为零**，从而**降低模型的复杂度**（模型的复杂度由参数决定），从而防止过拟合，提高模型的泛化能力。

L1正则中有个问题，便是**L1范数在0处不可导**，即|w|在0处不可导，因此在w为0时，使用原来的未经正则化的更新方程来对w进行更新，即令sgn(0)=0
`!$sgn(w)_{w>0}=1,sgn(w)_{w<0}=−1,sgn(w)_{w=0}=0$`

L1正则线性回归即为**Lasso回归**

#### L2正则

在回归里面，有人把有它的回归叫**“岭回归”**（Ridge Regression），有人也叫它“**权值衰减weight decay**”。为它的强大功效是改善机器学习里面一个非常重要的问题：过拟合。

L2范数是指向量各元素的平方和然后求平方根。我们让L2范数的规则项||W||2最小，可以使得W的每个元素都很小，都接近于0，但与L1范数不同，它不会让它等于0，而是接近于0，这里是有很大的区别的哦。**而越小的参数说明模型越简单，越简单的模型则越不容易产生过拟合现象**。为什么越小的参数说明模型越简单？限制了参数很小，实际上就**限制了多项式某些分量的影响很小**

L2正则是基于L2范数，即在目标函数后面加上参数的L2范数和项，即**参数的平方和**与参数的积项，即： 
`!$ C = C_0 + \frac{\lambda}{2n}\sum_ww^2 $`
在计算梯度的时候 ，模型更新为
`!$ w = w + \alpha\frac{\partial C_0}{\partial \omega} + \frac{\lambda}{n}\sum_ww$`

L2正则线性回归即为** Ridge回归，岭回归 **
L2正则项起到使得参数w变小加剧的效果，但是为什么可以防止过拟合呢？一个通俗的理解便是：更小的参数值w意味着模型的复杂度更低，对训练数据的拟合刚刚好（奥卡姆剃刀），不会过分拟合训练数据，从而使得不会过拟合，以提高模型的泛化能力。 
[机器学习中的范数规则化之（一）L0、L1与L2范数](http://blog.csdn.net/zouxy09/article/details/24971995)
#### 总结 
[机器学习中使用「正则化来防止过拟合」到底是一个什么原理？为什么正则化项就可以防止过拟合？](https://www.zhihu.com/question/20700829)

比如L2正则，相当于给模型参数W添加了一个协方差为`!$ \frac{n}{\lambda} $`的零均值高斯分布先验，而对于`!$ \lambda=0 $`,则表示方差为无穷大，表明模型参数的变化可以很大不受约束。而随着`!$\alpha$`的增大，模型约束变大，模型就更稳定了。（因为模型参数选择范围小了，模型参数的已知条件变多了，模型复杂度就降低了）

L2与L1的区别在于，**L1正则是拉普拉斯先验，而L2正则则是高斯先验**。它们都是服从均值为0，协方差为1λ。当λ=0时，即没有先验
![正则化][8]
上图中的模型是线性回归，有两个特征，要优化的参数分别是w1和w2，左图的正则化是l2，右图是l1。蓝色线就是优化过程中遇到的等高线，一圈代表一个目标函数值，圆心就是样本观测值（假设一个样本），半径就是误差值，受限条件就是红色边界（就是正则化那部分），二者相交处，才是最优参数。可见右边的最优参数只可能在坐标轴上，所以就会出现0权重参数，使得模型稀疏。


加入正则化是 在**bias和variance之间做一个tradeoff**. bias就是训练误差呀,variance就是各个特征之间的权值差别了。 


[作者：刑无刀](https://www.zhihu.com/question/20700829/answer/21156998)
![高斯分布][9]









#### dropout
[ImageNet Classification with Deep Convolutional Neural Networks ](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
通过修改神经网络本身结构来实现的
训练开始时，随机得删除一些（可以设定为一半，也可以为1/3，1/4等）隐藏层神经元，即认为这些神经元不存在，同时保持输入层与输出层神经元的个数不变，这样便得到如下的ANN： 

![droupout][10]

按照BP学习算法对ANN中的参数进行学习更新（虚线连接的单元不更新，因为认为这些神经元被临时删除了）。这样一次迭代更新便完成了。下一次迭代中，同样随机删除一些神经元，与上次不一样，做随机选择

## 激活函数 

[神经网络激励函数的作用是什么？有没有形象的解释](https://www.zhihu.com/question/22334626)

激活函数可以引入非线性因素，解决线性模型所不能解决的问题。

从直觉上讲，SVM的核函数与神经网络的激活函数做了同样的事情，实现非线性分类。但是SVM是在输入层特征处进行核变换做的，而激活函数则是在输出层做的。


[深度学习中的激活函数导引](https://mp.weixin.qq.com/s?__biz=MzI1NTE4NTUwOQ==&mid=2650325236&idx=1&sn=7bd8510d59ddc14e5d4036f2acaeaf8d&mpshare=1&scene=1&srcid=1214qIBJrRhevScKXQQuqas4&pass_ticket=w2yCF%2F3Z2KTqyWW%2FUwkvnidRV3HF9ym5iEfJ%2BZ1dMObpcYUW3hQymA4BpY9W3gn4#rd)

新型激活函数ReLU克服了梯度消失，使得深度网络的直接监督式训练成为可能
### 激活函数的作用 
神经网络中激活函数的主要作用是提供网络的非线性建模能力。假设一个示例神经网络中仅包含线性卷积和全连接运算，那么该网络仅能够表达线性映射，即便增加网络的深度也依旧还是线性映射，难以有效建模实际环境中非线性分布的数据。

### Sigmoid
![sigmoid函数][11]

![sigmoid函数图像][12]

sigmoid 在定义域内处处可导，且两侧导数逐渐趋近于0。benjo将这类激活函数定义为 **软饱和激活函数**，饱和就像极限一样，也分为左饱和和右饱和。

![sigmoid函数处处可导][13]

常见的 ReLU 就是一类左侧硬饱和激活函数。
由于在后向传递过程中，**sigmoid向下传导的梯度包含了一个f'(x) 因子**（sigmoid关于输入的导数），因此一旦输入落入饱和区，f'(x) 就会变得接近于0，导致了向底层传递的梯度也变得非常小。此时，网络参数很难得到有效训练。这种现象被称为**梯度消失**。一般来说， sigmoid 网络在 5 层之内就会产生梯度消失现象
sigmoid (0, 1) 的输出还可以被表示作概率，或用于**输入的归一化**

### tanh 函数
![tanh函数 ][14]

![函数曲线][15]

`!$ tanh(x) = 2sigmoid(2x) - 1 $`,也具有软饱和性，收敛速度比sigmoid快。

### ReLU

2006年Hinton教授提出通过**逐层贪心预训练**解决深层网络训练困难的问题，但是深度网络的直接监督式训练的最终突破，最主要的原因是采用了新型激活函数**ReLU**
与传统的sigmoid激活函数相比，ReLU能够**有效缓解梯度消失**问题，从而直接以监督的方式训练深度神经网络

![relu定义][16]

![relu函数曲线][17]

ReLU 在`!$x<0$` 时硬饱和。由于 `!$x>0$`时导数为 1，所以，ReLU 能够在`!$x>0$`时**保持梯度不衰减，从而缓解梯度消失问题**。但随着训练的推进，部分输入会落入**硬饱和区，导致对应权重无法更新。这种现象被称为“神经元死亡**”。

ReLU还经常被“诟病”的一个问题是输出具有**偏移现象[7]，即输出均值恒大于零**。偏移现象和 神经元死亡会共同影响网络的收敛性。

还有 PReLU 、maxout、elu等，不同的激活函数搭配不同的参数初始化策略

## 深度学习学习资料


[UFLDL教程中文版](http://ufldl.stanford.edu/wiki/index.php/UFLDL%E6%95%99%E7%A8%8B)

[杨立坤的deeplearning](https://github.com/exacity/deeplearningbook-chinese)

[莫凡 tensorflow](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/1-2-install/)

[tensorfly](http://www.tensorfly.cn/home/)

[Neural Networks and Deep Learning中文翻译](https://hit-scir.gitbooks.io/neural-networks-and-deep-learning-zh_cn/content/)

[一文弄懂神经网络中的反向传播法——BackPropagation](http://www.cnblogs.com/charlotte77/p/5629865.html)

[一个浙大直博生推荐的学习资料](fb46167ebbdefa3e6c766c784e104234d1284871)

[一个外文的tensorflow入门教程](http://cv-tricks.com/category/tensorflow-tutorial/)

[别人的论文笔记](http://blog.csdn.net/bea_tree/article/category/6242399/1)

[皮果提的深度学习笔记](http://blog.csdn.net/peghoty/article/category/1451403)

[邹晓艺汇总的深度学习学习资料](http://blog.csdn.net/zouxy09/article/details/8782018)

[Deep Learning（深度学习）学习笔记整理系列之（一）](http://blog.csdn.net/zouxy09/article/details/8775360)

[深度学习论文集](https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap)

  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1490521285112.jpg "1490521285112"

  [3]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1490147713221.jpg "1490147713221"
  [4]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1490149294773.jpg "1490149294773"
  [5]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1490149958729.jpg "1490149958729"
  [6]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%90%84SGD%E4%BC%98%E5%8C%96%E6%96%B9%E6%B3%95%E5%9C%A8%E9%9E%8D%E7%82%B9%E5%A4%84%E7%9A%84%E8%A1%A8%E7%8E%B0.gif "深度学习各SGD优化方法在鞍点处的表现"
  [7]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1490521030446.jpg "1490521030446"
  [8]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1490176366325.jpg "1490176366325"
  [9]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1490176507628.jpg "1490176507628"
  [10]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1490183037687.jpg "1490183037687"
  [11]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1490184239110.jpg "1490184239110"
  [12]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1490184259652.jpg "1490184259652"
  [13]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1490184401385.jpg "1490184401385"
  [14]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1490244926736.jpg "1490244926736"
  [15]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1490244945123.jpg "1490244945123"
  [16]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1490254499923.jpg "1490254499923"
  [17]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1490254518999.jpg "1490254518999"