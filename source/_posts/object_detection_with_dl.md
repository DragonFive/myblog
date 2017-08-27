---
title: 目标检测算法总结

date: 2017/7/30 12:04:12

categories:
- 深度学习
tags:
- 目标检测
- 深度学习
- 神经网络
---
[TOC]

 去年总结了一篇关于目标检测的博客 [视频智能之——目标检测](https://dragonfive.github.io/object_detection/)，今年到现在有了新的体会，所以就更新一篇。
 ![目标检测题图][1]
<!--more-->

# 目标检测

## 检测算法划分

目标检测的算法大致可以如下划分：
- 传统方法：
1. 基于Boosting框架：Haar/LBP/积分HOG/ICF/ACF等特征+Boosting等
2. 基于SVM：HOG+SVM or DPM等

- CNN方法：
1. 基于region proposal：以Faster R-CNN为代表的R-CNN家族
2. 基于回归区域划分的：YOLO/SSD
3. 给予强化学习的：attentionNet等。 
4. 大杂烩：Mask R-CNN

## selective search 策略

selective search的策略是，因为目标的层级关系，用到了**multiscale**的思想，那我们就尽可能遍历所有的尺度好了，但是不同于暴力穷举，可以先得到小尺度的区域，然后一次次**合并**得到大的尺寸就好了。既然特征很多，那就把我们知道的特征都用上，但是同时也要照顾下**计算复杂度**，不然和穷举法也没啥区别了。最后还要做的是能够**对每个区域进行排序**，这样你想要多少个候选我就产生多少个。

- 使用**Efficient GraphBased** Image Segmentation中的方法来得到region
- 得到所有region之间两两的相似度
- 合并最像的两个region
- 重新计算新合并region与其他region的相似度
- 重复上述过程直到整张图片都聚合成一个大的region
- 使用一种**随机的计分方式**给每个region打分，按照分数进行ranking，取出**top k**的子集，就是selective search的结果

### 区域划分与合并

首先通过**基于图的图像分割方法?**初始化原始区域，就是将图像分割成很多很多的小块。然后我们使用**贪心策略**，计算每两个相邻的区域的相似度，然后每次合并最相似的两块，直到最终只剩下一块完整的图片。然后这其中每次产生的图像块包括合并的图像块我们都保存下来，这样就得到图像的**分层表示**了呢。

优先合并小的区域

### 颜色空间多样性

作者采用了8中不同的颜色方式，主要是为了考虑场景以及光照条件等。这个策略主要应用于【1】中图像分割算法中原始区域的生成。主要使用的颜色空间有：**（1）RGB，（2）灰度I，（3）Lab，（4）rgI（归一化的rg通道加上灰度），（5）HSV，（6）rgb（归一化的RGB），（7）C，（8）H（HSV的H通道）**

使用L1-norm归一化获取图像**每个颜色通道的25 bins**的直方图，这样每个区域都可以得到一个**75维**的向量。![enter description here][2]，区域之间颜色相似度通过下面的公式计算：

![enter description here][3]
 在区域合并过程中使用需要对新的区域进行计算其直方图，计算方法：
 
![enter description here][4]
优先合并小的区域.





### 距离计算多样性

 这里的纹理采用SIFT-Like特征。具体做法是对每个颜色通道的8个不同方向计算方差σ=1的高斯微分（GaussianDerivative），每个通道每个方向获取10 bins的直方图（L1-norm归一化），这样就可以获取到一个240维的向量。![enter description here][5]


还有大小相似度(优先合并小的区域)和吻合相似度(boundingbox是否在一块儿)

### 给区域打分

这篇文章做法是，给予最先合并的**图片块**较大的权重，比如最后一块完整图像权重为1，倒数第二次合并的区域权重为2以此类推。但是当我们策略很多，多样性很多的时候呢，这个权重就会有太多的重合了，排序不好搞啊。文章做法是给他们乘以一个随机数，毕竟3分看运气嘛，然后对于相同的区域多次出现的也叠加下权重，毕竟多个方法都说你是目标，也是有理由的嘛。

区域的分数是区域内图片块权重之和。
### reference

[目标检测（1）-Selective Search](https://zhuanlan.zhihu.com/p/27467369)

[论文笔记 《Selective Search for Object Recognition》](http://blog.csdn.net/csyhhb/article/details/50425114)


## RCNN 

RCNN作为第一篇目标检测领域的深度学习文章。这篇文章的创新点有以下几点：将CNN用作目标检测的特征提取器、**有监督预训练的方式初始化**CNN、在CNN特征上做**BoundingBox 回归**。

目标检测区别于目标识别很重要的一点是其需要目标的具体位置，也就是BoundingBox。而产生BoundingBox最简单的方法就是滑窗，可以在卷积特征上滑窗。但是我们知道**CNN是一个层次的结构，随着网络层数的加深，卷积特征的步伐及感受野也越来越大**。

对于每一个region proposal 都wrap到固定的大小的scale,227x227(AlexNet Input),对于每一个处理之后的图片，把他都放到CNN上去进行特征提取，得到每个region proposal的feature map,这些特征用固定长度的特征集合feature vector来表示。



本文的亮点在于网络结构和训练集
### 训练集

经典的目标检测算法在区域中提取人工设定的特征（Haar，HOG）。本文则需要训练深度网络进行特征提取。可供使用的有两个数据库： 
一个较大的识别库（ImageNet ILSVC 2012）：标定每张图片中物体的类别。**一千万图像，1000类**。 
一个较小的检测库（PASCAL VOC 2007）：标定每张图片中，物体的类别和位置。**一万图像，20类**。 



### 整体结构 

![RCNN][6]


![网络结构][7]

RCNN的输入为完整图片，首先通过区域建议算法产生一系列的候选目标区域，其中使用的区域建议算法为**Selective Search,选择2K**个置信度最高的区域候选。

然后对这些候选区域预处理成227 × 227 pixel size ，16 pixels of warped image context around the original box 

然后对于这些目标区域候选提取其**CNN特征AlexNet**，并训练**SVM分类**这些特征。最后为了提高定位的准确性在SVM分类后区域基础上进行**BoundingBox回归**。

#### CNN目标特征提取 finetune


RCNN使用ImageNet的有标签数据进行有监督的**预训练Alexnet**，然后再在**本数据集上微调最后一层全连接层**。直到现在，这种方法已经成为CNN初始化的标准化方法。但是训练CNN的样本量还是不能少的，为了尽可能获取最多的正样本，RCNN将**IOU>0.5（IoU 指重叠程度，计算公式为：A∩B/A∪B）的样本都称为正样本**。每次ieration 的batch_size为128，其中正样本个数为32，负样本为96.其实这种设置是偏向于正样本的，因为正样本的数量实在是太少了。由于CNN需要固定大小的输入，因此对于每一个区域候选，首先将其防缩至227\*227，然后通过CNN提取特征。


#### svm训练 
数据是在经过微调的RCNN上取得Fc7层特征，然后训练SVM，并通过BoundingBox回归得到的最终结果。
RCNN的SVM训练将**ground truth样本作为正样本**，而**IOU>0.3的样本作为负样本**，这样也是SVM困难样本挖掘的方法。

**分类器**
对每一类目标，使用一个线性SVM二类分类器进行判别。输入为深度网络输出的4096维特征，输出是否属于此类。 
由于负样本很多，使用hard negative mining方法。 
**正样本**
本类的真值标定框。 
**负样本**
考察每一个候选框，如果和本类所有标定框的重叠都**小于0.3**，认定其为负样本


#### 贪婪非极大值抑制

由于有多达2K个区域候选，我们如何筛选得到最后的区域呢？RCNN使用**贪婪非极大值抑制**的方法，假设ABCDEF五个区域候选，首先根据概率从大到小排列。假设为FABCDE。然后从最大的F开始，计算F与ABCDE是否IoU是否超过某个阈值，如果超过则将ABC舍弃。然后再从D开始，直到集合为空。而这个阈值是筛选得到的，通过这种处理之后一般只会剩下几个区域候选了。

#### BoundingBox回归

为了进一步提高定位的准确率，RCNN在贪婪非极大值抑制后进行BoundingBox回归，进一步微调BoundingBox的位置。不同于DPM的BoundingBox回归，RCNN是在Pool5层进行的回归。而**BoundingBox是类别相关的，也就是不同类的BoundingBox回归的参数是不同的**。例如我们的区域候选给出的区域位置为：也就是区域的中心点坐标以及宽度和高度。

目标检测问题的衡量标准是重叠面积：许多看似准确的检测结果，往往因为候选框不够准确，重叠面积很小。故需要一个位置精修步骤。 回归器对每一类目标，使用一个线性脊回归器进行精修。正则项λ=10000.
输入为深度网络**pool5层**的4096维特征，输出为xy方向的**缩放和平移**。 训练样本判定为本类的候选框中，和真值**重叠面积大于0.6的候选框**。


### 瓶颈

- 速度瓶颈：重复为每个region proposal提取特征是极其费时的，Selective Search对于每幅图片产生2K左右个region proposal，也就是意味着一幅图片需要经过2K次的完整的CNN计算得到最终的结果。
- 性能瓶颈：对于所有的region proposal**放缩**到固定的尺寸会导致我们不期望看到的**几何形变**，而且由于速度瓶颈的存在，不可能采用多尺度或者是大量的数据增强去训练模型。



### reference

[目标检测（2）-RCNN](https://zhuanlan.zhihu.com/p/27473413)


[RCNN学习笔记(0):rcnn简介](http://blog.csdn.net/u011534057/article/details/51240387)


[ RCNN学习笔记(1):](http://blog.csdn.net/u011534057/article/details/51218218)

## SPPNet

论文：Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition


![sppnet与rcnn的区别][8]

图可以看出SPPnet和RCNN的区别，首先是输入不需要放缩到指定大小。其次是增加了一个空间金字塔池化层，使得fc层能够固定参数个数。

空间金字塔池化层是SPPNet的核心

![enter description here][9]
其主要目的是对于任意尺寸的输入产生固定大小的输出。思路是对于任意大小的feature map首先分成16、4、1个块，然后在每个块上最大池化，池化后的特征拼接得到一个固定维度的输出。以满足全连接层的需要。不过因为不是针对于目标检测的，所以输入的图像为一整副图像。

为了简便，在fintune的阶段只修改fc层。

### reference

[目标检测（3）-SPPNet](https://zhuanlan.zhihu.com/p/27485018)



## fast RCNN

无论是RCNN还是SPPNet，其**训练都是多阶段**的。首先通过ImageNet**预训练网络模型**，然后通过检测数据集**微调模型提取每个区域候选的特征**，之后通过**softmax分类**每个区域候选的种类，最后通过**区域回归**，精细化每个区域的具体位置。为了避免多阶段训练，同时在单阶段训练中提升识别准确率，Fast RCNN提出了**多任务目标函数**，将softmax分类以及区域回归的部分纳入了卷积神经网络中。

所有的层在finetune阶段都是可以更新的，使用了truncated SVD方法，MAP是66%.

### 网络结构

![fast RCNN网络结构][10]

类似于RCNN，Fast RCNN首先通过**Selective Search**产生一系列的区域候选，然后通过通过**CNN**提取每个区域候选的特征，之后**训练分类网络softmax以及区域回归网络**。对比SPPNet，首先是将SPP换成了**ROI Poling**。ROI Poling可以看作是空间金字塔池化的简化版本，它通过将区域候选对应的卷积层特征还分为H\*W个块，然后在每个块上进行最大池化就好了。每个块的划分也简单粗暴，直接使用卷积特征尺寸除块的数目就可以了。空间金字塔池化的特征是多尺寸的，而ROI Pooling是**单一尺度**的。而对于H\*W的设定也是参照网络Pooling层的，例如对于VGG-19，网络全连接层输入是7\*7\*512，因此对应于我们的H,W就分别设置为7，7就可以了。另外一点不同在于网络的输出端，无论是SPPNet还是RCNN，CNN网络都是仅用于特征提取，因此输出端只有网络类别的概率。而Fast RCNN的网络输出是**包含区域回归**的。


将网络的**输出改为两个子网络**，一个用以分类（softmax）一个用于回归。最后更改网络的输入，网络的**输入是图片集合以及ROI的集合**。

- Mini-Batch 采样
Mini-Batch的设置基本上与SPPNet是一致的，不同的在于128副图片中，仅来自于两幅图片。其中**25%的样本为正样本，也就是IOU大于0.5的**，其他样本为负样本，同样使用了**困难负样本挖掘**的方法，也就是负样本的IOU区间为[0.1，0.5），负样本的u=0，[u> 1]函数为艾弗森指示函数，意思是如果是背景的话我们就不进行区域回归了。在训练的时候，每个区域候选都有一个正确的标签以及正确的位置作为监督信息。
- ROI Pooling的反向传播
不同于SPPNet，我们的ROI Pooling是可以反向传播的，让我们考虑下正常的Pooling层是如何反向传播的，以**Max Pooling为例，根据链式法则，对于最大位置的神经元偏导数为1，对于其他神经元偏导数为0**。ROI Pooling 不用于常规Pooling，因为很多的区域建议的感受野可能是相同的或者是重叠的，因此在一个Batch_Size内，我们需要对于这些重叠的神经元偏导数进行求和，然后反向传播回去就好啦。

### reference

[目标检测（4）-Fast R-CNN](https://zhuanlan.zhihu.com/p/27582096)

## Faster RCNN

Fast RCNN提到如果去除区域建议算法的话，网络能够接近实时，而 **selective search方法进行区域建议的时间一般在秒级**。产生差异的原因在于卷积神经网络部分运行在GPU上，而selective search运行在CPU上，所以效率自然是不可同日而语。一种可以想到的解决策略是将selective search通过GPU实现一遍，但是这种实现方式忽略了接下来的**检测网络可以与区域建议方法共享计算**的问题。因此Faster RCNN从提高区域建议的速度出发提出了region proposal network 用以通过GPU实现快速的区域建议。通过**共享卷积，RPN在测试时的速度约为10ms**，相比于selective search的秒级简直可以忽略不计。Faster RCNN整体结构为RPN网络产生区域建议，然后直接传递给Fast RCNN。

### faster rcnn 结构
![faster RCNN的结构][11]

对于一幅图片的处理流程为：图片-卷积特征提取-RPN产生proposals-Fast RCNN分类proposals。

### region proposal network

区域建议算法一般分为两类：基于超像素合并的（selective search、CPMC、MCG等），基于滑窗算法的。由于卷积特征层一般很小，所以得到的滑窗数目也少很多。但是产生的滑窗准确度也就差了很多，毕竟感受野也相应大了很多。

![区域建议算法][12]

RPN对于feature map的每个位置进行**滑窗**，通过**不同尺度以及不同比例的K个anchor**产生K个256维的向量，然后分类每一个region是否包含目标以及通过**回归**得到目标的具体位置。



### reference

[目标检测（5）-Faster RCNN](https://zhuanlan.zhihu.com/p/27988828)

## YoLo


### reference
[YOLO：实时快速目标检测](https://zhuanlan.zhihu.com/p/25045711)


# 其它reference


[CVPR2017-目标检测相关论文](https://zhuanlan.zhihu.com/p/28088956)


  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501725558357.jpg
  [2]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501731400655.jpg
  [3]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501731416622.jpg
  [4]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501731430205.jpg
  [5]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501731713492.jpg
  [6]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501733503711.jpg
  [7]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1503646999361.jpg
  [8]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501816659717.jpg
  [9]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501819431880.jpg
  [10]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501820819379.jpg
  [11]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501831614917.jpg
  [12]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1501832022067.jpg