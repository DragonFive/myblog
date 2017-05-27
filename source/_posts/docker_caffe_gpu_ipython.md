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
# 从nvidia-docker到docker，需要有这样的操作
大致流程如下，入门版通过docker pull一个GPU版本的caffe 的image,然后安装nvidia-docker 和 nvidia-docker-plugin 来映射宿主机的nvidia-driver并通过共享volume的方式来支持容器里面能“看到”宿主机的GPU。进阶版通过curl -s命令列出宿主机的配置显卡配置，并通过docker run的方式来启动。总结完成。纸上得来终觉浅，绝知此事要躬行，光说不练空把式，唯有实践出真知。
[tensorflow gpu in docker](https://xuxinkun.github.io/2016/10/08/tensorflow-kubernetes/)

使用nvidia-docker
```
sudo nohup nvidia-docker-plugin >/tmp/nvidia-docker.log &  
然后nvidia-docker run
```
使用docker来代替 nvidia-docker
```
curl -s http://localhost:3476/docker/cli
```
下面应该是输出：
```
--device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia7 --devic/dev/nvidia6 --device=/dev/nvidia5 --device=/dev/nvidia4 --device=/dev/nvidia3 --device=/dev/nvidia2 --device=/dev/nvidia1 --device=/dev/nvidia0 --volume-driver=nvidia-docker --volume=nvidia_driver_375.39:/usr/local/nvidia:ro
```
这样其实
```
sudo docker run -ti `curl -s http://localhost:3476/v1.0/docker/cli` -v /mnt/share:/mnt/share -v /mnt/lustre:/mnt/lustre  -v /lib64:/lib64 镜像名 bash
```
所以如果你想用docker的方式来运行GPU版本 那么你就需要指明你的所有的device信息，如果卸载rc文件里，那么只能这样
```
sudo docker run -ti --device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia7 --device=/dev/nvidia6 --device=/dev/nvidia5 \
--device=/dev/nvidia4 --device=/dev/nvidia3 --device=/dev/nvidia2 --device=/dev/nvidia1 --device=/dev/nvidia0 \
--volume-driver=nvidia-docker --volume=nvidia_driver_375.39:/usr/local/nvidia:ro \
 -v /mnt/share:/mnt/share -v /mnt/lustre:/mnt/lustre  -v /lib64:/lib64  镜像名  bash
```

# 在镜像里安装ipython notebook，需要这样做
把大象装进冰箱分三步


[把jupyter-notebook装进docker里](https://segmentfault.com/a/1190000007448177)



# 参考资料

[把jupyter-notebook装进docker里](https://segmentfault.com/a/1190000007448177)
[tensorflow gpu in docker](https://xuxinkun.github.io/2016/10/08/tensorflow-kubernetes/)