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

![错误量](http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjp0hJlhelcn7XVuqEfyDuDwliaD9eJx9aJNxuO5ktGqQsSlLCibKicrR7pglVf8Yq7iauxqG0ueibhVKXQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)

![依据下一层错误量δl+1获取错误量δl的等式](http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjp0hJlhelcn7XVuqEfyDuDwGSCWDfia4mCUkMWibu7LJW3E461HRBgeBTuDia421G8jfzw0eiaBBlYnvQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)

![网络的代价函数相对于偏置的改变速率](http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjp0hJlhelcn7XVuqEfyDuDwgp0HvHMJywr794ic3kyiaecicPXZAz6c4MyR0Gdrup4WzfxAmGBChWREQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)


![网络的代价函数相对于权重的改变速率](http://mmbiz.qpic.cn/mmbiz/58FUuNaBUjp0hJlhelcn7XVuqEfyDuDwWF23AeicSicQJs4Ekia7Qx3j1ntModqNaFmUrQaputUGENiaeazMn5dMKA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)


# reference


[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html)

[参考：《神经网络与深度学习》连载——哈工大](https://mp.weixin.qq.com/s?__biz=MzIxMjAzNDY5Mg==&mid=503307054&idx=1&sn=d20623df35d1771dc548d545ed38f318&chksm=0f474ec53830c7d3bd43285f1b32a69ee76887676ce276446aed833512ddc1d3515331b954e7&mpshare=1&scene=1&srcid=0712NxuIohdYeyT9HV9KoJD0&pass_ticket=ih%2BTmMdW0BKOpaQftxTEsre0o%2FuiaqArflVqs4UY1MJqSN5yV0Im5QO0FlBgY6QF#rd)


  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1499827436670.jpg
