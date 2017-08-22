---
title: tensorflow_object_detection
date: 2017/8/23 17:38:58

categories:
- 深度学习
tags:
- deeplearning
- 目标检测
- tensorflow
- faster-rcnn
---




<!--more-->
# 环境配置

## docker  tensorflow-gpu


## protobuf


## slim-models


编译proto的内容

```bash
protoc object_detection/protos/*.prot --python_out=.
```
设置 PYTHONPATH 环境变量

```cpp
export PYTHONPATH=$PYTHONPATH:'pwd':'pwd'/slim
```


# reference

[protobuf的配置](http://zhwen.org/?p=909)

[tf-slim的用法](http://geek.csdn.net/news/detail/126133)

[tf-slim官方教程](https://github.com/tensorflow/models/blob/master/slim/slim_walkthrough.ipynb)

[TensorFlow Object Detection API 实践](https://mp.weixin.qq.com/s?__biz=MzI2MzYwNzUyNg==&mid=2247484024&idx=1&sn=a7ddf704f34d390bd2a64f7651ea4a44&chksm=eab807f1ddcf8ee78c6b28bea6ec7233b7236a9d653dea45859e36ea2c74afe3b350d6f97967&mpshare=1&scene=1&srcid=0822n8yUoxY0I5ITT55mu827&pass_ticket=bpMLj9Sb52k%2FdkdRYGa0rUDxv9zPd7tmsAPy7XAQiBMfzqdlgf06HEq0G62d4Kyz#rd)

