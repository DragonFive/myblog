
title: 目标跟踪算法之相关滤波

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


赵老师   21号到23号有VALSE 第7届年度研讨会  

这个会是国内计算机视觉领域的顶级学术交流会  作报告的有孙剑(微软face++)、山世光(计算所)、颜水成(360系)、欧阳万里(商汤系)、王亮(自动化所)、程明明(南开系)、吴毅(做跟踪的大牛)等   

## 判别模型 （也叫 tracking-by-detection）



# 参考资料 

[目标跟踪之NIUBILITY的相关滤波](https://zhuanlan.zhihu.com/DCF-tracking)

  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1492588025562.jpg "1492588025562"