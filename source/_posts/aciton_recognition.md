title: 动作识别
date: 2016/12/25 22:04:12
categories:
- 计算机视觉
tags:
- 动作识别
- 深度学习
- 论文
- 视频
---



[从CVPR 2014看计算机视觉领域的最新热点](http://www.msra.cn/zh-cn/research/academic-conferences/cvpr-2014.aspx)

深度学习热潮爆发以来，诸多研究者希望能够把它应用于解决计算机视觉的各种任务上，从高层次（high- level）的识别（recognition），分类（classification）到低层次（low-level）的去噪（denoising）。让人不禁联想起当年的稀疏表达（sparse representation）的热潮，而深度学习如今的风靡程度看上去是有过之而无不及。深度学习也有横扫high-level问题的趋势，high- level的很多方向都在被其不断刷新着数据。
<!-- more -->

尚未被深度学习渗透的Low-level Vision
# 计算机视觉与深度学习
计算机视觉的问题可以根据他们的研究对象和目标分成三大类，low- level，mid-level, 和high-level。
1. Low-level问题主要是针对**图像本身及其内在属性的分析**及处理，比如判断图片**拍摄时所接受的光照**，反射影响以及光线方向，进一步推断拍摄物体的几何结构；再如**图片修复**，如何**去除图片拍摄中所遇到的抖动和噪声**等不良影响。
2. High-level问题主要是针对**图像内容的理解和认知**层面的，比如说**识别与跟踪图像中的特定物体**与其行为；根据已识别物体的深入推断，比如**预测物体所处的场景和即将要进行的行为**。
3. Mid-level是介于以上两者之间的一个层面，个人理解是着重于**特征表示**，比如说如何描述high-level问题中的目标物体，使得这种描述有别于其他的物体。可以大致认为，low-level的内容可以服务于mid-level的问题，而mid-level的内容可以服务于high-level的问题。由于这种分类不是很严格，所以也会出现交叉的情况。

**深度学习在计算机视觉界主要是作为一种特征学习的工具**，可以姑且认为是mid-level的。所以之前提到的high- level的问题受深度学习的影响很大就是这个原因。相比较而言low-level问题受到深度学习的冲击会小很多，当然也有深度学习用于去噪（denoise）和去模糊（deblur）等low-level问题的研究。对于受到深度学习良好表现困扰的年轻研究者们，也不妨来探寻low- level很多有意思的研究。这些年，MIT的Bill Freeman组就做了一些很有趣的low-level问题，比如放大视频中出现的肉眼难以察觉的细小变化（[Eulerian Video Magnification for Revealing Subtle Changes in the World](http://people.csail.mit.edu/mrub/vidmag/)），还有这次CVPR的文章[Camouflaging an Object from Many Viewpoints](http://camouflage.csail.mit.edu/)就是讲如何在自然环境中放置和涂染一个立方体，让其产生变色龙般的隐藏效果。诸如此类的研究也让研究这件事变得有趣和好玩。

# 视频智能内容


参考资料 [基于Deep Learning 的视频识别技术](https://yq.aliyun.com/articles/39134)

人工智能在视频上的应用主要一个课题是视频理解，努力解决“语义鸿沟”的问题，其中包括了

 

1. 视频结构化分析：即是对视频进行帧、镜头、场景、故事等分割，从而在多个层次上进行处理和表达。
2. 目标检测和跟踪：如车辆跟踪，多是应用在安防领域。
3. 人物识别：识别出视频中出现的人物。
4. 动作识别：Activity Recognition， 识别出视频中人物的动作。
5. 情感语义分析：即观众在观赏某段视频时会产生什么样的心理体验。
6. 四位一体分析 object action  scene  geometry

短视频、直播视频中大部分承载的是**人物+场景+动作+语音**的内容信息，如上图所示，如何用有效的特征对其内容进行表达是进行该类视频理解的关键。

传统的手工特征有一大堆，目前效果较好的是iDT(Improved Dense Trajectories) ，在这里就不加讨论了。


深度学习对图像内容的表达能力十分不错，在视频的内容表达上也有相应的方法。下面介绍最近几年主流的几种技术方法。



# 知乎问答 

## DT

行为识别常用哪种特征提取？ - 回答作者: Yongcheng Jing http://zhihu.com/question/41068341/answer/102114782

。

一种分类方式是将其分为两大类，一大类是基于局部描述子的statistical information的方法，像HOG3D等。这一类中目前知道的比较好的是**Dense Trajectory（DT）**方法，作者在这个方法上下了很大功夫，由DT方法在ICCV,CVPR,IJCV上发表了好几篇文章（文章名字都很像，都是讲DT这一个东西的，只是做了一些改进，像15年的文章里面考虑了相机抖动、用Fisher encoding代替Bag of feature以取得更好效果等），源码有提供LEAR - Improved Trajectories Video Description，我在JHMDB数据集上做过实验，效果还不错。另一大类是基于pose的行为识别方法，pose可以提取更加细节的信息，先用pose estimation方法估计pose，再从pose中提取各个关节之间的角度、距离等等作为特征。但受pose estimation准确率的影响，目前这种方法不是很常用，但实验发现在理想的pose estimation情况下这种方法准确率是很高的（高于DT），所以可能是未来行为识别领域的一个发展趋势，源码见http://jhmdb.is.tue.mpg.de/challenge/JHMDB/datasets。
另外，在选择实验数据集的时候可以参考12年的CVPR Tutorial，里面详细介绍了目前的开源数据集以及截止12年各个数据集的分类准确率。

## IDT 与deep learning

行为识别(action recognition)有哪些论文适合入门？ - 回答作者: Xiaolong Wang http://zhihu.com/question/33272629/answer/60279003


有关action recognition in videos, ideo里主流的：

Deep Learning之前最work的是INRIA组的Improved Dense Trajectories(IDT) + fisher vector, paper and code: 
[LEAR - Improved Trajectories Video Description](http://lear.inrialpes.fr/people/wang/improved_trajectories)
基本上INRIA的东西都挺work 恩..

然后Deep Learning比较有代表性的就是VGG组的[2-stream:](http://arxiv.org/abs/1406.2199)
其实效果和IDT并没有太大区别，里面的结果被很多人吐槽难复现，我自己也试了一段时间才有个差不多的数字。

然后就是在这两个work上面就有很多改进的方法，目前的state-of-the-art也是很直观可以想到的是xiaoou组的[IDT+2-stream:](http://wanglimin.github.io/papers/WangQT_CVPR15.pdf)

还有前段时间很火，现在仍然很多人关注的G社的[LSTM+2-stream: ](http://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/43793.pdf)

然后安利下zhongwen同学的paper:
http://www.cs.cmu.edu/~zhongwen/pdf/MED_CNN.pdf

最后你会发现paper都必需和IDT比，然后很多还会把自己的method和IDT combine一下说有提高 恩..


## 静态图像的动作识别 

作者：水哥
链接：https://www.zhihu.com/question/33272629/answer/60163859
来源：知乎

视频方面的不了解，可以聊一聊静态图像下的~
```
[1] Action Recognition from a Distributed Representation of Pose and Appearance, CVPR,2010
[2] Combining Randomization and Discrimination for Fine-Grained Image Categorization, CVPR,2011
[3] Object and Action Classification with Latent Variables, BMVC, 2011
[4] Human Action Recognition by Learning Bases of Action Attributes and Parts, ICCV, 2011
[5] Learning person-object interactions for action recognition in still images, NIPS, 2011
[6] Weakly Supervised Learning of Interactions between Humans and Objects, PAMI, 2012
[7] Discriminative Spatial Saliency for Image Classification, CVPR, 2012
[8] Expanded Parts Model for Human Attribute and Action Recognition in Still Images, CVPR, 2013
[9] Coloring Action Recognition in Still Images, IJCV, 2013
[10] Semantic Pyramids for Gender and Action Recognition, TIP, 2014
[11] Actions and Attributes from Wholes and Parts, arXiv, 2015
[12] Contextual Action Recognition with R*CNN, arXiv, 2015
[13] Recognizing Actions Through Action-Specific Person Detection, TIP, 2015
```
在2010年左右的这几年（11,12）主要的思路有3种：

1. 以所交互的物体为线索（person-object interaction），建立交互关系，如文献5,6；
2. 建立关于姿态（pose）的模型，通过统计姿态（或者更广泛的，部件）的分布来进行分类，如文献1,4，还有个poselet上面好像没列出来，那个用的还比较多；
3. 寻找具有鉴别力的区域（discriminative），抑制那些meaningless 的区域，如文献2,7。10和11也用到了这种思想。
 
文献9,10都利用了SIFT以外的一种特征：color name，并且描述了在动作分类中如何融合多种不同的特征。
文献12探讨如何结合上下文（因为在动作分类中会给出人的bounding box）。
比较新的工作都用CNN特征替换了SIFT特征（文献11,12,13），结果上来说12是最新的。

静态图像中以分类为主，检测的工作出现的不是很多，文献4,13中都有关于检测的工作。可能在2015之前分类的结果还不够promising。现在PASCAL VOC 2012上分类mAP已经到了89%，以后的注意力可能会更多地转向检测。

视频的个别看过几篇，与静态图像相比，个人感觉最大的区别在于特征不同。到了中层以后，该怎么做剩下的处理，思路还是差的不远。

# 论文阅读 


## 人体动作行为识别研究综述_李瑞峰

![动作识别.png](http://upload-images.jianshu.io/upload_images/454341-38bb05f6bc21b096.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


## CVPR 2014 Tutorial  Action recognition with bag-of-features
[CVPR 2014 Tutorial on Emerging Topics in Human Activity Recognition](http://michaelryoo.com/cvpr2014tutorial/)


### 研究点 

视频种类：监控视频 电影剪辑  运动场拍 第一视觉视频 
动作级别分类：姿势->动作->人物交互(击球)->人人交互(握手 推 拥抱)->群体动作

监控视频：
1. 动作 (Laptev 05) 
2. 人物交互(Oh et al. 11)  
3. 人人交互[Ryoo and Aggarwal 09][Vahdat, Gao,Ranjbar, Mori11] 
4. 群体动作[Ryoo and arwal 08,11][Lan, Wang, Yang, Mori 10]

### 问题所在 

1. 动作在表现中各有不同 
2. 人工收集训练样本是很难的 
3. 动作词典难以描述 

怎么计算两个动作之间的差距 

都是open  开门 和 开瓶盖 不一样 
### 研究方法

#### 局部特征和特征袋表示 local features and bag of features representations

1. 相关归类论文

- Wang, et al., " Action Recognition by Dense Trajectories", In CVPR’11.
- Jain et al., "Better exploiting motion for better action recognition", In CVPR'13.
- Wang and Schmid, "Action Recognition with Improved Trajectories“, In ICCV'13.
- Kantorov and Laptev, "Efficient feature extraction, encoding and classification for action recognition “, In CVPR'14.

DeepPose: Human Pose Estimation via Deep Neural Networks, Toshev and Szegedy,CVPR 2014.

Mixing Body-Part Sequences for Human Pose Estimation, Cherian,Mairal, Alahari,Schmid, CVPR 2014.

2. shape 形状估计法

背景差分 其实对内部结构变化估计不好 

3. motion  运动 用光流估计运动以此来估计动作  


4. 局部特征法： 

不需要图像分割 不需要目标检测和追踪

缺少全局的结构信息

5. bag of features 动作识别法

[Laptev, Marszałek, Schmid, Rozenfeld 2008]

特征提取-> 时空patches -> 特征描述 -> 特征量化 -> 直方图 -> 非线性 SVM 与卡方核

6. 总结结论 



####  中层动作表示 


1. 相关归类论文

Liu et al., "Recognizing Human Actions by Attributes", In CVPR'11

Discovering discriminative action parts from mid-level video representations Raptis, Kokkinos, and Soatto, CVPR 2012.

Sadanand and Corso. "Action bank: A high-level representation of activity in video." In CVPR’12.

Representing Videos using Mid-level Discriminative Patches Jain, Gupta, Rodriguez and Davis, CVPR 2013

使用CNN的方法

Large-scale Video Classification using Convolutional Neural Networks Karpathy, Shetty, Toderici, Sukthankar,Leung and Fei-Fei, CVPR 2014 


2. 总结问题 

可以：定位运动的部分 标注转动

不如state of the art 的方法 Fisher Vector + Improved dense trajectories

有可能被CNNs取代 



#### 动作时序结构 temporal structure of action

1. 相关归类论文

Modeling Temporal Structure of Decomposable Motion Segments for Activity Classication, J.C. Niebles, C.-W. Chen and L. Fei-Fei, ECCV 2010

Learning Latent Temporal Structure for Complex Event Detection.Kevin Tang, Li Fei-Fei and Daphne Koller, CVPR 2012

Poselet Key-framing: A Model for Human Activity Recognition.Raptis and Sigal, In CVPR 20132012

Scenario-based video event recognition by constraint flow. Kwak, Han and Han, In CVPR 2011

Amer et al., Monte Carlo Tree Search for Scheduling Activity Recognition, In ICCV 2013.

2. 总结问题 

可以对子运动进行时域分析 

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-b10f1bd2580646a3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



## 群体行为识别 Human-human interactions

### 群体识别的应用意义

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-73360a167fdaaf0a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-5aeae05eab0f19f5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-f36bfaa08e6261f3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-513807b05fa477f4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-093cad287f0676bc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)




# 相关资料 
[CVPR 2014 Tutorial on Emerging Topics in Human Activity Recognition](http://michaelryoo.com/cvpr2014tutorial/)

[cvpr2016的tutorials](http://cvpr2016.thecvf.com/program/tutorials)
