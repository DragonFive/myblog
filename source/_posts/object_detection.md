title: 视频智能之——目标检测
date: 2016/12/26 22:04:12
categories:
- 计算机视觉
tags:
- 目标检测
- 深度学习
- 论文
---





2016年的CVPR会议目标检测（在这里讨论的是2D的目标检测，如图1所示）的方法主要是基于卷积神经网络的框架，代表性的工作有ResNet[1]（Kaiming He等）、YOLO[5]（Joseph Redmon等）、LocNet[7]（Spyros Gidaris等）、HyperNet[3]（Tao Kong等）、ION[2]（Sean Bell等）、G-CNN[6]（Mahyar Najibi等）


![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-3b20c67b56642033.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

<!-- more -->

# CNN目标检测框架

## RCNN
早期，使用窗口扫描进行物体识别，计算量大。RCNN去掉窗口扫描，用聚类方式，对图像进行分割分组，得到多个侯选框的层次组。

![RNN流程](http://upload-images.jianshu.io/upload_images/145616-f4c5c9a89c842dcb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 原始图片通过Selective Search提取候选框，约有2k个
- 侯选框缩放成固定大小
- 经过CNN
- 经两个全连接后，分类

[基于R-CNN的物体检测论文分析](http://blog.csdn.net/hjimce/article/details/50187029)

后面的[fast RCNN](https://arxiv.org/abs/1504.08083)去掉重复计算，并微调选框位置。

[faster RCNN](https://arxiv.org/abs/1506.01497)使用CNN来预测候选框。

## RCNN系列
([RCNN](https://arxiv.org/pdf/1605.06409v1.pdf)、Fast RCNN、Faster RCNN)中，网络由两个子CNN构成。在图片分类中，只需一个CNN，效率非常高。

## YOLO


Faster RCNN需要对20k个anchor box进行判断是否是物体，然后再进行物体识别，分成了两步。 YOLO则把物体框的选择与识别进行了结合，一步输出，即变成”You Only Look Once”。
[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

## SSD

YOLO在 7×7 的框架下识别物体，遇到大量小物体时，难以处理。SSD则在不同层级的feature map下进行识别，能够覆盖更多范围。

[SSD: Single Shot MultiBox Detector](http://159.226.251.230/videoplayer/ssd.pdf?ich_u_r_i=56e086595274f0397af25c4f31ce72e6&ich_s_t_a_r_t=0&ich_e_n_d=0&ich_k_e_y=1645128926751363372457&ich_t_y_p_e=1&ich_d_i_s_k_i_d=10&ich_u_n_i_t=1)

# CVPR2016 目标检测问题

## 检测指标

目标检测中，以下几个指标非常重要：
- 识别精度
- 识别效率
- 定位准确性

较好的工作常常在某个指标上有所提高

### 识别精度
目标检测中衡量检测精度的指标mAP(mean average precision)。简单来讲就是在多个类别的检测中，每一个类别都可以根据recall和precision绘制一条曲线，那么AP就是该曲线下的面积，而mAP是多个类别AP的平均值，这个值介于0到1之间，且越大越好。具有代表性的工作是ResNet、ION和HyperNet。

#### ResNet
ResNet：何凯明的代表作之一，获得了今年的best paper。 文章不是针对目标检测来做的，但其解决了一个最根本的问题：**更有力的特征**。检测时基于Faster R-CNN的目标检测框架，使用ResNet替换VGG16网络可以取得更好的检测结果。

#### ION
ION（inside-outside-network）：这个工作的主要贡献有两个，第一个是如何**在Fast R-CNN的基础之上增加context信息**，所谓context在目标检测领域是指感兴趣的ROI周围的信息，可以是局部的，也可以是全局的。为此，作者提出了IRNN的概念，这也就是outside-network。第二个贡献是所谓skip-connection，通过将deep ConvNet的多层ROI特征进行提取和融合，利用该特征进行每一个位置的分类和进一步回归，这也就是inside-network。

#### HyperNet
HyperNet：文章的出发点为一个很重要的观察：**神经网络的高层信息体现了更强的语义信息，对于识别问题较为有效；而低层的特征由于分辨率较高，对于目标定位有天然的优势，而检测问题恰恰是识别+定位**，因此作者的贡献点在于如何将deep ConvNet的高低层特征进行融合，进而利用融合后的特征进行region proposal提取和进一步目标检测。不同于Faster R-CNN，文章的潜在Anchor是用类似于BING[4]的方法通过扫描窗口的方式生成的，但利用的是CNN的特征，因此取得了更好的性能。

通过以上的改进策略，HyperNet可以在产生大约100个region proposal的时候保证较高的recall，同时目标检测的mAP相对于Fast R-CNN也提高了大约6个百分点。

### 识别效率

#### YOLO
YOLO：这是今年的oral。这个工作在识别效率方面的优势很明显，可以做到每秒钟45帧图像，处理视频是完全没有问题的。YOLO最大贡献是提出了一种全新的检测框架——**直接利用CNN的全局特征预测每个位置可能的目标**，相比于R-CNN系列的**region proposal+CNN **这种两阶段的处理办法可以大大提高检测速度。


#### G-CNN
G-CNN：不管是Fast R-CNN[9]，还是Faster R-CNN，或者像HyperNet这样的变种，都需要考虑数以万计的潜在框来进行目标位置的搜索，这种方式的一个潜在问题是负样本空间非常大，因此需要一定的策略来进行抑制（不管是OHEM[8]（难例挖掘）还是region proposal方法，其本质上还是一种抑制负样本的工作）。G-CNN从另一个角度来克服这个问题。G-CNN在在初始化的时候不需要那么多框，而是通过对图像进行划分（有交叠），产生少量的框（大约180个），通过一次回归之后得到更接近物体的位置。然后以回归之后的框作为原始窗口，不断进行迭代回归调整，得到最终的检测结果。

经过五次调整之后，G-CNN可以达到跟Fast R-CNN相当的识别性能，但速度是Fast R-CNN的5倍（3fps）。

### 定位准确性
在目标检测的评价体系中，有一个参数叫做IoU，简单来讲就是模型产生的目标窗口和原来标记窗口的交叠率。在Pascal VOC中，这个值为0.5。，定位越准确，其得分越高，这也侧面反映了目标检测在评价指标方面的不断进步。

如何产生更准确的目标位置呢？LocNet的解决方案是：针对每一个给定的初始框进行适当的放大，然后用一个CNN的网络回归出这个放大后的框包含的那个正确框的位置。为了达到这个目标，需要定义回归方式，网络以及模型，具体的细节参见[7]。

经过把原始的框（比如selective search生成的）进行再一次回归之后，再放入Fast R-CNN进行检测，效果还是挺惊人的。


## 相关论文 


[1] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition. In CVPR 2016

[2] Bell S, Zitnick C L, Bala K, et al. Inside-outside net: Detecting objects in context with skip pooling and recurrent neural networks. In CVPR 2016

[3] Kong T, Yao A, Chen Y, et al. HyperNet: Towards Accurate Region Proposal Generation and Joint Object Detection. In CVPR 2016

[4] Cheng M M, Zhang Z, Lin W Y, et al. BING: Binarized normed gradients for objectness estimation at 300fps. In CVPR 2014

[5] Redmon J, Divvala S, Girshick R, et al. You only look once: Unified, real-time object detection. In CVPR 2016

[6] Najibi M, Rastegari M, Davis L S. G-CNN: an Iterative Grid Based Object Detector. In CVPR 2016

[7] Gidaris S, Komodakis N. LocNet: Improving Localization Accuracy for Object Detection. In CVPR 2016

[8] Shrivastava A, Gupta A, Girshick R. Training region-based object detectors with online hard example mining. In CVPR 2016

[9] Girshick R. Fast R-CNN. In ICCV 2015

[10] Ren S, He K, Girshick R, et al. Faster R-CNN: Towards real-time object detection with region proposal networks. In NIPS 2015

[11] Liu W, Anguelov D, Erhan D, et al. SSD: Single Shot MultiBox Detector[J]. arXiv preprint arXiv:1512.02325, 2015.






# 参考资料

[对话CVPR2016：目标检测新进展](https://zhuanlan.zhihu.com/p/21533724)

[目标检测-大神](http://www.cosmosshadow.com/ml/%E5%BA%94%E7%94%A8/2015/12/07/%E7%89%A9%E4%BD%93%E6%A3%80%E6%B5%8B.html)

[Detection CNN 之 "物体检测" 篇](http://www.jianshu.com/p/067f6a989d31)

[ 深度学习（十八）基于R-CNN的物体检测-CVPR 2014](http://blog.csdn.net/hjimce/article/details/50187029)
