---
title: 《Neural Network and Deep Learning》学习

date: 2017/7/10 12:04:12

categories:
- 深度学习
tags:
- deeplearning
- 梯度下降法
- 正则化
- 激活函数
- 神经网络
---
[TOC]

 Michael Nielsen的《Neural Network and Deep Learning》
![神经网络和深度学习][1]

<!--more-->
# 神经网络基础
## sigmoid神经元
假设在网络的一些权值（或偏移）上做一个小的改变。我们期望的结果是，这些在权值上的小改变，将会为网络的输出结果带来相应的改变，且这种改变也必须是轻微的。我们在后面将会看到，满足这样的性质才能使学习变得可能。我们就可以改变权值和偏移来使得网络的表现越来越接近我们预期。
当使用σ函数时我们就得到了一个平滑的感知机。而且，σ函数的平滑属性才是其关键。sigmoid函数的代数形式保证了其在求微分的时候很方便。
根据我的经验，很多layer后面都会跟上一个激活函数，激活函数用于将该层之前的输入的线性组合进行非线性运算，如果没有激活函数那么整个神经网络就相当于一层了。另外一些激活函数有梯度消失的问题，当层数过深的时候在后向传播的时候梯度趋近于0,relu能够缓解这个问题，同时能够让一些神经元的梯度为0,来减少运算量。

[不同激活函数(activation function)的神经网络的表达能力是否一致？ - 回答作者: 纳米酱](https://zhihu.com/question/41841299/answer/92683898)

我们也有一些存在回馈环路可能性的人工神经网络模型。这种模型被称为**递归神经网络（recurrent neural networks）**。该模型的关键在于，神经元在变为非激活态之前会在一段有限时间内均保持激活状态。这种激活状态可以激励其他的神经元，被激励的神经元在随后一段有限时间内也会保持激活状态。如此就会导致更多的神经元被激活，一段时间后我们将得到一个级联的神经元激活系统。


[用简单的网络结构解决手写数字识别](https://mp.weixin.qq.com/s?__biz=MzIxMjAzNDY5Mg==&mid=400137026&idx=1&sn=f5c8a9ab3e24d7c0bd38058ba211d22a&scene=21#wechat_redirect)

## 损失函数
![损失函数](http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjqmFnpiaugfukXTylgau1kOB5cBdtFib3TiaOQy9ImBB9OwyEnJtR1ibowkldBGM1GG16Tiaq3GexGoGEQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)

为什么要介绍平方代价（quadratic cost）呢？毕竟我们最初所感兴趣的内容不是对图像正确地分类么？为什么不增大正确输出的得分，而是去最小化一个像平方代价类似的间接评估呢？这么做是因为在神经网络中，**被正确分类的图像的数量所关于权重、偏置的函数并不是一个平滑的函数**。大多数情况下，对权重和偏置做出的微小变动并不会影响被正确分类的图像的数量。这会导致我们很难去刻画如何去优化权重和偏置才能得到更好的结果。**一个类似平方代价的平滑代价函数能够更好地指导我们如何去改变权重和偏置来达到更好的效果**。这就是为何我们集中精力去最小化平方代价，只有通过这种方式我们才能让分类器更精确。

[神经网络与深度学习中对为什么梯度下降法能够work做了详尽的解释](https://mp.weixin.qq.com/s?__biz=MzIxMjAzNDY5Mg==&mid=400169015&idx=1&sn=757c015a7d2aea1b79a681158dd107e9&scene=21#wechat_redirect)

ΔC≈∇C⋅Δv. (9)
Δv=−η∇C, (10)
这里的η是个很小的正数（就是我们熟知的学习速率）。等式（9）告诉我们ΔC≈−η∇C⋅∇C=−η||∇C||2。由于||∇C||2≥0，这保证了ΔC≤0，例如，如果我们以等式（10）的方式去改变v，那么C将一直会降低，不会增加。（当然，要在（9）式的近似约束下）。这就是我们想要的特性！

为了使我们的梯度下降法能够正确地运行，我们需要选择足够小的学习速率η使得等式（9）能得到很好的近似。

## 反向传播

如果输入神经元是低激活量的，或者输出神经元已经饱和（高激活量或低激活量），那么权重就会学习得缓慢。

[反向传播背后的四个基本等式](https://mp.weixin.qq.com/s?__biz=MzIxMjAzNDY5Mg==&mid=400329443&idx=1&sn=f7158ee615c2a0d6f0014adae038193e&scene=21#wechat_redirect)

![四个基本等式][2]

反向传播的四个过程


![enter description here][3]

方向传播算法就是将错误量从输出层反向传播。


# 改进神经网络的方式
学习的速度（收敛速度）与很多因素有关，学习率、代价函数的形式、激活函数的形式都有关系。这是使用平方误差作为代价函数。

![代价函数][4]
假设y=0是我们的输出。
![对权值偏导][5]
我们能够从图像看出当神经元输出接近Rendered by QuickLaTeX.com时，曲线变得非常平坦，因此激活函数的导数会变得很小。

## 交叉熵代价函数与sigmoid激活函数
可以用不同的代价函数比如交叉熵（cross-entropy）代价函数来替代二次代价函数来让学习速度加快。

![交叉熵代价函数][6]

**交叉熵函数为什么能作为代价函数**
交叉熵有两个特性能够合理地解释为何它能作为代价函数。首先，它是**非负**的; 其次，如果对于所有的训练输入x，这个神经元的**实际输出值都能很接近我们期待的输出**的话，那么交叉熵将会非常接近0。

。而且交叉熵有另一个均方代价函数不具备的特征，**它能够避免学习速率降低的情况**。为了理解这个，我们需要计算一下交叉熵关于权重的偏导。因为在计算代价函数关于权值的偏导的时候，sigmoid函数的导数会与交叉熵中导数的一部分抵消掉。

![误差关于权值的偏导][7]-


当我们的输出层是线性神经元（linear neurons）的时候使用均方误差，假设我们有一个多层神经网络。假设最后一层的所有神经元都是线性神经元（linear neurons）意味着我们不用sigmoid作为激活函数。

|   激活函数  |  函数   |  导数   |  特点   | 
| --- | --- | --- | --- |
|  sigmoid   |  ![sigmoid][8]   |  ![sigmoid导数][9]   |  有饱和状态   |
|  tanh   | ![tanh][10]    |  ![enter description here][11]   | tanh保持非线性单调，延迟饱和 ，[-1,1]   |
|  relu    |  y=max(0,x)   |  导数为常数   |   节省计算量，避免梯度丢失，网络稀疏  |
|   softplus  |  y=log(1+e^x)  |     |  softplus可以看作是ReLu的平滑  |

**如何选择为自己的模型选择合适的激活函数**

1. 通常使用tanh激活函数要比sigmoid收敛速度更快；
2. 在较深层的神经网络中，选用relu激活函数能使梯度更好地传播回去
3. 当使用softmax作为最后一层的激活函数时，其前一层最好不要使用relu进行激活，而是使用tanh作为替代，否则最终的loss很可能变成Nan；
4. 当选用高级激活函数时，建议的尝试顺序为ReLU->ELU->PReLU->MPELU，因为前两者没有超参数，而后两者需要自己调节参数使其更适应构建的网络结构。

## softmax 层的输出是一个概率分布

**softmax 的单调性**，如果J=K，那么![enter description here][12]为正数， 如果J!=K,则为负数，这个表明了 z增大，则相应的a增大，其它a（输出概率）就见效

**softmax 的非局部性**，任何一个输出激活值a依赖于所有的输入值。


**softmax函数的导数**

[参考softmax分类](http://www.jianshu.com/p/8eb17fa41164)

[Softmax 输出及其反向传播推导](http://shuokay.com/2016/07/20/softmax-loss/)

![softmax函数][13]

当 j = k时， ![enter description here][14]

当j!=k,  ![enter description here][15]



**损失函数是交叉熵损失函数**



# reference


[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html)

[解析深度学习中的激活函数](http://www.jianshu.com/p/6033a5d3ad4b)

[参考：《神经网络与深度学习》连载——哈工大](https://mp.weixin.qq.com/s?__biz=MzIxMjAzNDY5Mg==&mid=503307054&idx=1&sn=d20623df35d1771dc548d545ed38f318&chksm=0f474ec53830c7d3bd43285f1b32a69ee76887676ce276446aed833512ddc1d3515331b954e7&mpshare=1&scene=1&srcid=0712NxuIohdYeyT9HV9KoJD0&pass_ticket=ih%2BTmMdW0BKOpaQftxTEsre0o%2FuiaqArflVqs4UY1MJqSN5yV0Im5QO0FlBgY6QF#rd)


  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1499827436670.jpg
  [2]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501419765215.jpg
  [3]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1500378801942.jpg
  [4]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501232404378.jpg
  [5]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501232423609.jpg
  [6]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501233195651.jpg
  [7]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501241302962.jpg
  [8]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501341517202.jpg
  [9]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501341554626.jpg
  [10]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501341596684.jpg
  [11]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501341636250.jpg
  [12]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501419089972.jpg
  [13]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501341429967.jpg
  [14]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501341467832.jpg
  [15]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501341954136.jpg