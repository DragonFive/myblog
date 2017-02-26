title: KCF算法-1论文学习
date: 2016/12/28 22:04:12
categories:
- 计算机视觉
tags:
- 目标跟踪
- oepncv
- 论文
- kcf
---


KCF全称为Kernel Correlation Filter 核相关滤波算法，是在2014年提出的，算法在跟踪效果和跟踪速度上都有十分亮眼的表现。
[算法主页](http://www.robots.ox.ac.uk/~joao/circulant/index.html)有论文和代码可供下载。opencv3.1开始嵌入KCF算法

相关滤波算法算是判别式跟踪，主要是使用给出的样本去训练一个判别分类器，判断跟踪到的是目标还是周围的背景信息。使用**轮转矩阵**对样本进行采集，使用**快速傅里叶变化**对算法进行加速计算。

<!--more-->

相关（correlation filter）应用在tracking方面的最为直观的方法：相关就是衡量两个信号相似值。如果两个信号相关值越大，就表明这两个信号的越相似。在tracking的应用当中，我们的目的就是设计这么一个模版，使得当模版和输入的图片目标做相关时，能在目标的中心位置得到最大的响应

由卷积定理的correlation版本可以知道，函数互相关傅里叶变换等于函数傅里叶变换的乘积。如公式②所示。 
F(h)★f=（F(h)）*⊙F(f)——————–②
而已知FFT变换的时间复杂度为O(nlogn),由此可知，②式的时间复杂度也为O(nlogn)。明白为什么相关滤波类型的tracking algorithm的计算时间比较快

![](http://img.blog.csdn.net/20160623172443644)

# 论文大致内容
负样本对训练一个分类器是一个比较重要的存在，但是在训练的时候负样本的数量是比较少的，所以我们本文的算法就是为了更加方便地产生更多的样本，以便于我们能够训练一个更好的分类器。 

而Correlation Filter应用于tracking方面最朴素的想法就是：相关是衡量两个信号相似值的度量，如果两个信号越相似，那么其相关值就越高，而在tracking的应用里，就是需要设计一个滤波模板，使得当它作用在跟踪目标上时，得到的响应最大，最大响应值的位置就是目标的位置。

CSK（论文下载地址）是这个算法改进的初级版本，这篇是主要引进了循环矩阵生成样本，使用相关滤波器进行跟踪，本篇KCF是对CSK进行更进一步的改进，**引进了多通道特征，可以使用比着灰度特征更好的HOG（梯度颜色直方图）特征或者其他的颜色特征**等。

提出了一个快速的效果良好的跟踪算法，把以前只能用**单通道的灰度特征改进为现在可以使用多通道的HOG特征**或者其他特征，而且在现有算法中是表现比较好的，使用HOG替换掉了灰度特征，对实验部分进行了扩充，充分证明自己的效果是比较好的。使用**核函数，对偶相关滤波**去计算。

## 循环移位 cyclic shifts

循环矩阵是对图像进行平移，这样可以增加样本的个数，训练出的分类器效果比较好

一维的情况下就是矩阵想乘的问题了，就是矩阵分析当中学过的左乘一个单位矩阵和右乘一个单位矩阵。左乘是行变换，右乘列变化。目的就是得到更多的样本，每乘一次都是一个新的样本，这样的话就可以多出来n*n个样本了，这就是循环矩阵在这里最大的用处，制造样本的数量


循环矩阵的计算可以直接把所有的样本都转换为对角矩阵进行处理，因为在循环矩阵对样本进行处理的时候，样本并不是真实存在的样本，存在的只是虚拟的样本，可以直接利用循环矩阵所特有的特性，直接把样本矩阵转换为对角矩阵进行计算， 这样可以大幅度加快矩阵之间的计算，因为对角矩阵的运算只需要计算对角线上非零元素的值即可。



# 目标跟踪评价标准 
Visual tracker benchmark是一个相当于比较标准一般的存在，可以对比每个算法在相同的基准之下的性能。这个在CVPR 2013的时候发表的，下面是原文链接： 
[Online Object Tracking: A Benchmark （CVPR 2013)](http://faculty.ucmerced.edu/mhyang/papers/cvpr13_benchmark.pdf )

官方链接： 
[数据集](http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html )

在2015年的时候数据集由原来的50个序列扩充到现在的100个。 
现在多数使用的是Benchmark V1.0来运行测试所有tracking的结果，在官网下载完代码之后，里面已经存在了关于各种tracker在不同数据集上的结果了。

想要验证自己的tracker在这个基准上的结果，说来非常的简单。 
直接的方法： 
首先将代码先拷到benchmark_v1.0/tackers/这个文件夹下，你会发现里面已有好几个算法的代码文件夹了。 注意把代码拷贝进去之后要自己写一个调用函数，benchmark在运行的时候调用我们的算法的函数，就是每个tracker文件夹当中的run_trackers名字。



# 参考资料 



[Visual Object Tracking using Adaptive Correlation Filters 论文笔记](http://www.cnblogs.com/hanhuili/p/4266990.html)


[目标跟踪算法——KCF入门详解](http://blog.csdn.net/crazyice521/article/details/53525366)


[KCF学习（1）-理论](http://blog.csdn.net/zinnc/article/details/52675541)

[【目标跟踪】KCF高速跟踪详解](http://blog.csdn.net/shenxiaolu1984/article/details/50905283)

[kcf hog源码](https://github.com/joaofaro/KCFcpp)

[kcf跟踪算法学习笔记](http://blog.csdn.net/mhz9123/article/details/51670802)

[2014新跟踪算法KCF笔记](http://blog.csdn.net/zwlq1314521/article/details/50427038)

[kcf论文集](https://github.com/DragonFive/Correlation-Filter-Tracking)
