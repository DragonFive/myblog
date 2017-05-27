---
title: 不会装cuda配caffe的小白的救赎—玄学DL
tdate: 2017/5/26 15:04:12

categories:
- 计算机视觉
tags:
- nvidia驱动
- 深度学习
- caffe
- docker
- deeplearning
- python
---

DL如今已经快成为全民玄学了，感觉离民科入侵不远了。唯一的门槛可能是环境不好配，特别是caffe这种依赖数10种其它软件打框架。不过有了docker之后，小白也能轻松撸DL了。
![image.png](http://upload-images.jianshu.io/upload_images/454341-5fd9359ddbf72292.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
<!--more-->
# 大致流程
大致流程如下，入门版通过docker pull一个GPU版本的caffe 的image,然后安装nvidia-docker 和 nvidia-docker-plugin 来映射宿主机的nvidia-driver并通过共享volume的方式来支持容器里面能“看到”宿主机的GPU。进阶版通过curl -s命令列出宿主机的配置显卡配置，并通过docker run的方式来启动。明白了吧。

# 如何在镜像里安装ipython notebook




# 参考资料


