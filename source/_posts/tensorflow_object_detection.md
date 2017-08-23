---
title: tensorflow训练-finetune-压缩模型
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


docker 方式运行tensorflow
```bash
nvidia-docker run -p 8088:8088 -p 6006:6006 -v /home/dragon/code:/root/code  -it dragonfive/tensorflow:gpu bash
```
首先需要更新软件
```bash
apt-get update
```
然后安装git 等工具

```bash

apt-get install git vim

```




## protobuf
[protobuf的配置](http://zhwen.org/?p=909)

### 编译protobuf
解压下载的tar.gz包，cd到protobuf的目录下，执行以下指令：
```
./configure
make
make check
make install
```
### 解决命令找不到

1. 创建文件 /etc/ld.so.conf.d/libprotobuf.conf 包含内容
/usr/local/lib

2. 输入命令
ldconfig

再运行protoc --version 就可以正常看到版本号了



## slim-models


编译proto的内容

```bash
protoc object_detection/protos/*.proto --python_out=.
```
设置 PYTHONPATH 环境变量

```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
注意这里pwd外围的不是单引号，而是esc下面那个键


## 宠物数据集和标注

### 数据下载

下载训练数据里面的 图片
```
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
```
下载训练数据里面的标注信息
```
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
```
### 将训练数据格式转化为tfrecord格式

需要一个id与类名的对应关系的txt文件：pet_label_map.pbtxt

```cpp
item {
  id: 1
  name: 'Abyssinian'
}

item {
  id: 2
  name: 'american_bulldog'
}

item {
  id: 3
  name: 'american_pit_bull_terrier'
}

item {
  id: 4
  name: 'basset_hound'
}

```
**执行脚本**
需要一个把数据转化成 tfrecoder格式的脚本文件: 

**脚本参数**
这个脚本的使用方法是：
```cpp
    python  object_detection/create_pet_tf_record.py  --data_dir=`pwd`\
        --label_map_path=object_detection/data/pet_lable_map.pbtxt \
       --output_dir=`pwd`
```
需要指定data_dir :图像的源文件夹，output_dir：转化成tfrecord格式

--label_map_path：id与class_name的对应表。

**脚本流程**
1. 读取anotations文件夹里面的 trainval.txt ，来确定用来训练的图片的文件名，然后通过shuffle,按比例分成训练集和验证集
2. 然后根据训练集或测试集的文件名来读取anotations文件夹里面xmls下面的xml文件，获得图片对象
3. 把图片对象写入文件

**生成文件**

pet_train.record

pet_val.record

### 运行训练

**准备数据**

这时候把它们拷贝到新创建的文件夹,比如 ```_pet_20170822```

将 object_detection/data/pet_label_map.pbtxt 和  object_detection/samples/configs/faster_rcnn_resnet152_pets.config 两个文件也拷贝到这个目录。

将faster_rcnn_resnet152_pets.config文件内容中几个```PATH_TO_BE_CONFIGURED``` 都替换为 ```_pet_20170822```，第 110 和 111 行内容改为：

```
  #fine_tune_checkpoint: "_pet_20170822/model.ckpt"
  from_detection_checkpoint: false
```

**开始训练**

需要一个训练脚本  train.py 

```cpp
python  object_detection/train.py --logtostderr --train_dir=_pet_20170822/ --pipeline_config_path=_pet_20170822/faster_rcnn_resnet152_pets.config 

```

初次训练的时候用的是cpu
>/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/gradients_impl.py:95: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
2017-08-22 07:52:41.925484: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-08-22 07:52:41.925511: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-08-22 07:52:41.925517: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-08-22 07:52:41.925523: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-08-22 07:52:41.925529: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
2017-08-22 07:52:42.943663: I tensorflow/core/common_runtime/simple_placer.cc:697] Ignoring device specification /device:GPU:0 for node 'prefetch_queue_Dequeue' because the input edge from 'prefetch_queue' is a reference connection and already has a device field set to /device:CPU:0

解决这个问题，参考 https://github.com/tensorflow/models/issues/1695





![gpu训练过程][1]


![训练完成后生成的文件][2]

**监控训练**

```cpp
tensorboard --logdir=_pet_20170822
```

![enter description here][3]




### 评估训练好的网络 


```bash
python object_detection/eval.py \
        --logtostderr \
        --checkpoint_dir=_pet_20170822 \
        --eval_dir=_pet_20170822 \
        --pipeline_config_path=_pet_20170822/faster_rcnn_resnet152_pets.config 
```


![验证的过程][4]

# preTrained model
注意两点
1. restore参数的时候注意有些层不要初始化，因为跟原始层不一样
2. 训练的时候之前训练过的参数不需要重新训练了，就frozen起来

```cpp
checkpoint_exclude_scopes 

trainable_scopes
```


# reference

[protobuf的配置](http://zhwen.org/?p=909)

[tf-slim的用法,用于图像检测和分割](http://geek.csdn.net/news/detail/126133)

[tf-slim官方教程](https://github.com/tensorflow/models/blob/master/slim/slim_walkthrough.ipynb)

[TensorFlow Object Detection API 实践](https://mp.weixin.qq.com/s?__biz=MzI2MzYwNzUyNg==&mid=2247484024&idx=1&sn=a7ddf704f34d390bd2a64f7651ea4a44&chksm=eab807f1ddcf8ee78c6b28bea6ec7233b7236a9d653dea45859e36ea2c74afe3b350d6f97967&mpshare=1&scene=1&srcid=0822n8yUoxY0I5ITT55mu827&pass_ticket=bpMLj9Sb52k%2FdkdRYGa0rUDxv9zPd7tmsAPy7XAQiBMfzqdlgf06HEq0G62d4Kyz#rd)

[Fine-tuning a model from an existing checkpoint](https://github.com/tensorflow/models/tree/master/slim#fine-tuning-a-model-from-an-existing-checkpoint)

[Using pre-trained models](https://github.com/tensorflow/models/blob/master/slim/slim_walkthrough.ipynb)


[图像识别和分类竞赛，数据增强及优化算法](http://www.10tiao.com/html/511/201707/2652000542/2.html)

[tensorflow固化模型](http://www.jianshu.com/p/091415b114e2)

[TensorFlow Mobile模型压缩](http://www.jianshu.com/p/d2637646cda1)

[腾讯NCNN框架入门到应用](http://blog.csdn.net/best_coder/article/details/76201275)

[NCNN-squeezenet](https://github.com/dangbo/ncnn-mobile)

[multi-tracker](https://github.com/Smorodov/Multitarget-tracker)


  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1503395510074.jpg
  [2]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1503410104911.jpg
  [3]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1503399054053.jpg
  [4]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1503408506733.jpg