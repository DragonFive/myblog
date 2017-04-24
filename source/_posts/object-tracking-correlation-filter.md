
title: 目标跟踪算法之深度学习方法

date: 2017/4/19 22:04:12

categories:
- 计算机视觉
tags:
- objectTracking
- 目标跟踪
- 相关滤波
---
[TOC]

# tracking常用的数据库

## OTB 
**吴毅**的两篇论文给出了统一的数据库，在公共平台上进行对比。
- Wu Y, Lim J, Yang M H. Online object tracking: A benchmark [C]// CVPR, 2013.
- Wu Y, Lim J, Yang M H. Object tracking benchmark [J]. TPAMI, 2015.
<!--more-->
OTB50包括50个序列，后来扩展到OTB100.
![OTB50][1]

2012年以后相关滤波和深度学习崛起了。



## VOT 竞赛数据库
2013年开始的比赛，


- Kristan M, Pflugfelder R, Leonardis A, et al. The visual object tracking vot2013 challenge results [C]// ICCV, 2013.
- Kristan M, Pflugfelder R, Leonardis A, et al. The Visual Object Tracking VOT2014 Challenge Results [C]// ECCV, 2014.
- Kristan M, Matas J, Leonardis A, et al. The visual object tracking vot2015 challenge results [C]// ICCV, 2015.
- Kristan M, Ales L, Jiri M, et al. The Visual Object Tracking VOT2016 Challenge Results [C]// ECCV, 2016.

## OTB和VOT区别：
1. OTB包括**25%的灰度序列，但VOT都是彩色序列**，这也是造成很多颜色特征算法性能差异的原因；
2. 两个库的评价指标不一样，具体请参考论文；
3. VOT库的序列**分辨率**普遍较高，这一点后面分析会提到。
4.  最大差别在于：OTB是随机帧开始，或矩形框加随机干扰初始化去跑，作者说这样更加符合检测算法给的框框；而VOT是第一帧初始化去跑，每次跟踪失败(预测框和标注框不重叠)时，5帧之后再次初始化，VOT以short-term为主，且认为跟踪检测应该在一起永不分离，detecter会多次初始化tracker。

# 目标跟踪算法分类
## 生成模型 
在当前帧对目标区域建模，下一帧寻找与模型最相似的区域就是预测位置，比较著名的有**卡尔曼滤波，粒子滤波，mean-shift**等。举个例子，从当前帧知道了目标区域80%是红色，20%是绿色，然后在下一帧，搜索算法就像无头苍蝇，到处去找最符合这个颜色比例的区域，推荐算法**ASMS**

- Vojir T, Noskova J, Matas J. **Robust scale-adaptive mean-shift for tracking** [J]. Pattern Recognition Letters, 2014.



## 判别模型 （也叫 tracking-by-detection）

### 用深度学习做目标跟踪：
Naiyan Wang, Dit-Yan Yeung. Learning a Deep Compact Image Representation for Visual Tracking . Proceedings of Twenty-Seventh Annual Conference on Neural Information Processing Systems (NIPS) 2013 第一篇？

Naiyan Wang, Siyi Li, Abhinav Gupta, Dit-Yan Yeung, Transferring Rich Feature Hierarchies for Robust Visual Tracking. arXiv:1501.04587  2015

Chao Ma（沈春华澳洲学生）, Jia-Bin Huang, Xiaokang Yang and Ming-Hsuan Yang, "Hierarchical Convolutional Features for Visual Tracking", ICCV 2015  
Github 主页 https://github.com/jbhuang0604/CF2
http://www.cnblogs.com/wangxiaocvpr/p/5925851.html 一篇学习笔记

Lijun Wang, Wanli Ouyang, Xiaogang Wang, Huchuan Lu（卢湖川大连理工）, Visual Tracking with Fully Convolutional Networks, ICCV2015
Project site: http://scott89.github.io/FCNT/

Lijun Wang, Wanli Ouyang, Xiaogang Wang, Huchuan Lu ,STCT: Sequentially Training Convolutional Networks for Visual Tracking.cvpr 2016
Github repository: https://github.com/scott89/STCT
论文笔记：http://blog.csdn.net/u012905422/article/details/52396372
王柳军的博客：https://scott89.github.io/

Zhizhen Chi, Hongyang Li, Huchuan Lu, Ming-Hsuan Yang,Dual Deep Network for Visual Tracking. TIP 2017
Github repository: https://github.com/chizhizhen/DNT
迟至真的博客：http://chizhizhen.github.io/


Seunghoon Hong, Tackgeun You, Suha Kwak and Bohyung Han. Online Tracking by Learning Discriminative Saliency Map with Convolutional Neural Network 
ICML - International Conference on Machine Learning, 2015   
Project site: http://cvlab.postech.ac.kr/research/CNN_SVM/

H. Li, Y. Li, and F. Porikli, DeepTrack: Learning Discriminative Feature Representations by Convolutional Neural Networks for Visual Tracking, BMVC 2014


Hyeonseob Namand Bohyung Han, Learning Multi-Domain Convolutional Neural Networks for Visual Tracking
Project site: http://cvlab.postech.ac.kr/research/mdnet/

Heng Fan, Haibin Ling. "SANet: Structure-Aware Network for Visual Tracking." arXiv (2016). 
凌海滨个人主页：http://www.dabi.temple.edu/~hbling/

Ran Tao, Efstratios Gavves, Arnold W.M. Smeulders. Siamese Instance Search for Tracking. cvpr2016
博客笔记 http://www.dongcoder.com/detail-155114.html

Luca Bertinetto *, Jack Valmadre *, João F. Henriques, Andrea Vedaldi, Philip H.S. Torr, Fully-Convolutional Siamese Networks for Object Tracking, ECCV 2016   
Project site: http://www.robots.ox.ac.uk/~luca/siamese-fc.html
Author homepage: http://www.robots.ox.ac.uk/~luca/index.html 
http://www.robots.ox.ac.uk/~joao/# 牛津大学课题组 CF理论推进者
github repository: https://github.com/bertinetto/siamese-fc

David Held, Sebastian Thrun, Silvio Savarese,Learning to Track at 100 FPS with Deep Regression Networks, ECCV2016. 
Stanford课题组
Github repository: https://github.com/davheld/GOTURN
Project site: http://davheld.github.io/GOTURN/GOTURN.html


Yuankai Qi, Shengping Zhang, Lei Qin, Hongxun Yao, Qingming Huang, Jongwoo Lim, Ming-Hsuan Yang. Hedged Deep Tracking. Cvpr2016
计算所智能实验室黄庆明组。
Project site: https://sites.google.com/site/yuankiqi/hdt/

Guanghan Ning, Zhi Zhang, Chen Huang, Zhihai He, Xiaobo Ren, Haohong Wang, Spatially Supervised Recurrent Convolutional Neural Networks for Visual Object Tracking.arxiv: 2016.
Github repository: https://github.com/Guanghan/ROLO  
Project site: http://guanghan.info/projects/ROLO/ 宁广汉的主页 安徽大学phd.
王潇的论文笔记：http://www.cnblogs.com/wangxiaocvpr/p/5774840.html
王潇的github: https://github.com/wangxiao5791509



### 用相关滤波做跟踪：
第一篇文章：MOSSE
D. S. Bolme, J. R. Beveridge, B. A. Draper, and Y. M. Lui. Visual object tracking using adaptive correlation filters. In CVPR, 2010.

瑞典林雪平大学phd三年级学生 Martin Danelljan  ECO  CCOT作者
ECO: Efficient Convolution Operators for Tracking.CVPR2017
Learning Continuous Convolution Operators for Visual Tracking.ECCV2016.
Github reposityory: https://github.com/martin-danelljan/Continuous-ConvOp
Project site: https://theinformationageblog.wordpress.com/2017/01/12/eccv-2016-videos-beyond-correlation-filters-learning-continuous-convolution-operators-for-visual-tracking/


### benckmark:
吴毅：Annan Li, Min Lin, Yi Wu, Ming-Hsuan Yang and Shuicheng Yan. NUS-PRO: A New Visual Tracking Challenge.PAMI 2016.
Project site: https://sites.google.com/site/li00annan/nus-pro
CNT: Kaihua Zhang, Qingshan Liu, Yi Wu, Minghsuan Yang. "Robust Visual Tracking via Convolutional Networks Without Training." TIP (2016)
TIP:Transaction Image Processing IEEE期刊

王强：在这里可以看到作者收集了很多新的方法：计算所的 
Github repository: https://github.com/foolwood/benchmark_results
VOT的使用方法：http://blog.csdn.net/carrierlxksuper/article/details/47054231

Matthias Mueller, Neil Smith, Bernard Ghanem,A Benchmark and Simulator for UAV Tracking.
阿卜杜拉国王科技大学做的一个无人机航拍数据集。

Esteban Real, Jonathon Shlens, Stefano Mazzocchi, Xin Pan, Vincent Vanhoucke
,YouTube-BoundingBoxes: A Large High-Precision Human-Annotated Data Set for Object Detection in Video
Project site: https://research.google.com/youtube-bb/
Github repository: https://github.com/mbuckler/youtube-bb





# 参考资料 

[目标跟踪算法五：MDNet: Learning Multi-Domain Convolutional Neural Networks for Visual Tracking](https://zhuanlan.zhihu.com/p/25312850)

[深度学习在目标跟踪中的应用](https://zhuanlan.zhihu.com/p/22334661)

[目标跟踪之NIUBILITY的相关滤波](https://zhuanlan.zhihu.com/DCF-tracking)

[cvpr论文阅读笔记](http://www.cnblogs.com/wangxiaocvpr/)

[CNN-tracking-文章导读](http://blog.csdn.net/ben_ben_niao/article/details/51315000)




[物体跟踪-CVPR16-tracking](http://blog.csdn.net/ben_ben_niao/article/details/52072659)


[一个人的论文阅读笔记](http://blog.csdn.net/u012905422/article/category/6223501)



  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1492588025562.jpg "1492588025562"