---
title: CNN中卷积计算的内存和速度优化

date: 2017/9/20 12:04:12

categories:
- 深度学习
tags:
- deeplearning
- 网络优化
- 神经网络
---


在现在的DNN中，不管是前向传播还是反向传播，绝大多数时间花费在卷积计算中。因此对于速度提升来说，优化卷积层意义重大。

虽说从参数量来讲，早期的一些网络(alexbnet,VGG，googlnet等)70%以上的参数都是全连接层的。但是现在从架构上的改进已经开始减少全连接层了，比如squeezenet,mobilenet已经使用global avg pooling层取代全连接层了。那么接下来再想提速那就得从卷积层下手了。当然还有一中思路是从量化的方式减少参数量和内存消耗的（如BNN，eBNN），对于提速来说意义并不大。
<!--more-->

# 以往的卷积计算方法

## sum循环法 

时间复杂度最高，为 $O(HWMKKC)$ 最笨的方法，只是用来理解。
```
input[C][H][W];
kernels[M][K][K][C];
output[M][H][W];
for h in 1 to H do
	for w in 1 to W do
		for o in 1 to M do
			sum = 0;
			for x in 1 to K do
				for y in 1 to K do
					for i in 1 to C do
						sum += input[i][h+y][w+x]
						*kernels[o][x][y][i];
			output[o][w][h] = sum;
```

## patch-building DNN convolution algorithms

based on gemm convolution algorithm

