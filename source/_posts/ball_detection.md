---
title: gluon学习笔记

date: 2017/10/20 12:04:12

categories:
- 深度学习
tags:
- 目标检测
- 深度学习
- 神经网络
---


# 一些可以重复使用的代码
## 取一个batch_size的代码
**scratch版本**
```python
import random
batch_size = 1
def data_iter(num_examples):
    idx = list(range(num_examples))
    random.shuffle(idx)
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i:min(i+batch_size,num_examples)])
        yield X.take(j), y.take(j)
```
**gluon版本**


## 初始化权值

**scratch版本**

```python

def get_params():
    w = nd.random.normal(shape=(num_inputs, 1))*0.1
    b = nd.zeros((1,))
    for param in (w, b):
        param.attach_grad()
    return (w, b)
```

**gluon版本**








# reference

[从零开始码一个皮卡丘检测器](https://zhuanlan.zhihu.com/p/28867241)

[图片标注工具](http://blog.csdn.net/jesse_mx/article/details/53606897)

[ mxnet 使用自己的图片数据训练CNN模型](http://blog.csdn.net/u014696921/article/details/56877979)

[mxnet image API](https://mxnet.incubator.apache.org/api/python/image.html#Image)

[Create a Dataset Using RecordIO](https://mxnet.incubator.apache.org/how_to/recordio.html?highlight=recordio)

[基于MXNet gluon 的SSD模型训练](http://blog.csdn.net/muyouhang/article/details/77727381)

