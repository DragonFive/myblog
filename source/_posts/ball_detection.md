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


# 学到的新知识

## bn放在relu后面 

[BN应该放在relu后](http://minibatch.net/2017/06/11/%E7%BB%8F%E5%85%B8%E8%AE%BA%E6%96%87-Batch-Normalization/)

[用于分类、检测和分割的移动网络 MobileNetV2](https://mp.weixin.qq.com/s/xJromD5Q30KlRhB_kM4kfA)

[如何评价MobileNetV2](https://www.zhihu.com/question/265709710)

## 卷积核的数量
[卷积神经网络 — 从0开始](http://zh.gluon.ai/chapter_convolutional-neural-networks/cnn-scratch.html)

当输入数据有多个通道的时候，每个通道会有对应的权重，然后会对每个通道做卷积之后在通道之间求和。所以当输出只有一个的时候，卷积的channel数目和data的channel数目是一样的。

当输出需要多通道时，每个输出通道有对应权重，然后每个通道上做卷积。所以当输入有n个channel，输出有h个channel时，卷积核channel数目为n * h，每个输出channel对应一个bias ,卷积核的维度为(h,n,w,h)

$$conv(data, w, b)[:,i,:,:] = conv(data, w[i,:,:,:], b[i])$$

![inception v1][1]


![residual][2]

![resnet各种结构][3]



# gluon语法

## nn.Block与nn.sequential的嵌套使用
```python
class RecMLP(nn.Block):
    def __init__(self, **kwargs):
        super(RecMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        with self.name_scope():
            self.net.add(nn.Dense(256, activation="relu"))
            self.net.add(nn.Dense(128, activation="relu"))
            self.dense = nn.Dense(64)

    def forward(self, x):
        return nd.relu(self.dense(self.net(x)))

rec_mlp = nn.Sequential()
rec_mlp.add(RecMLP())
rec_mlp.add(nn.Dense(10))
print(rec_mlp)
```
## 初始化与参数访问
```python
from mxnet import init
params.initialize(init=init.Normal(sigma=0.02), force_reinit=True)
print(net[0].weight.data(), net[0].bias.data())
```
我们也可以通过collect_params来访问Block里面所有的参数（这个会包括所有的子Block）。它会返回一个名字到对应Parameter的dict。

也可以自定义各层的初始化方法，没有自定义的按照net.initialize里面的方法进行定义
```python
from mxnet.gluon import nn
from mxnet import nd
from mxnet import init

def get_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(4,activation="relu"))#,weight_initializer=init.Xavier()))
        net.add(nn.Dense(2,weight_initializer=init.Zero(),bias_initializer=init.Zero()) )
    return net

x = nd.random.uniform(shape=(3,5))
net = get_net()
net.initialize(init.One())
net(x)
print(net[1].weight.data
```
## GPU访问

1. 删除cpu版本mxnet

```bash
pip uninstall mxnet
```

2. 更新GPU版本mxnet
```bash
pip install -U --pre mxnet-cu80
```

3. 查看版本号
```python
import pip
for pkg in ['mxnet', 'mxnet-cu75', 'mxnet-cu80']:
    pip.main(['show', pkg])
```

## 使用jupyter的相关插件
1. notedown插件
可以在jupyter 中查看markdown文件

2. nb_conda
是conda的插件，可以在jupyter里面修改python内核版本


## 优化方法
**momentum**
gluon.Trainer的learning_rate属性和set_learning_rate函数可以随意调整学习率。
```python
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': mom})
```

**adagrad**
Adagrad是一个在迭代过程中不断自我调整学习率，并让模型参数中每个元素都使用不同学习率的优化算法。
```python
    trainer = gluon.Trainer(net.collect_params(), 'adagrad',
                            {'learning_rate': lr})
```

**Adam**

```python
trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
```

通过以上分析, 理论上可以说, 在数据比较稀疏的时候, adaptive 的方法能得到更好的效果, 例如, adagrad, adadelta, rmsprop, adam 等. 在数据稀疏的情况下, adam 方法也会比 rmsprop 方法收敛的结果要好一些, 所以, 通常在没有其它更好的理由的前框下, 我会选用 adam 方法, 可以比较快地得到一个预估结果. 但是, 在论文中, 我们看到的大部分还是最原始的 mini-batch 的 SGD 方法. 因为马鞍面的存在等问题, SGD 方法有时候较难收敛. 另外, SGD 对于参数的初始化要求也比较高. 所以, 如果要是想快速收敛的话, 建议使用 adam 这类 adaptive 的方法

## 延迟执行 

延后执行使得系统有更多空间来做性能优化。但我们推荐每个批量里至少有一个同步函数，例如对损失函数进行评估，来避免将过多任务同时丢进后端系统。
```python
from mxnet import autograd

mem = get_mem()

total_loss = 0
for x, y in get_data():
    with autograd.record():
        L = loss(y, net(x))
    total_loss += L.sum().asscalar()
    L.backward()
    trainer.step(x.shape[0])

nd.waitall()
print('Increased memory %f MB' % (get_mem() - mem))
```

## 多GPU训练 

```python
ctx = [gpu(i) for i in range(num_gpus)]
data_list = gluon.utils.split_and_load(data, ctx)
label_list = gluon.utils.split_and_load(label, ctx)


```
## fintune 微调

[gluon微调](https://fiercex.github.io/post/gluon_features_fine/)


# 一些可以重复使用的代码
## 读取数据
```python
from mxnet import gluon
from mxnet import ndarray as nd

def transform(data, label):
    return data.astype('float32')/255, label.astype('float32')
mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)
```


## 计算精度


```python
def accuracy(output, label):
    return nd.mean(output.argmax(axis=1)==label).asscalar()

```

我们先使用Flatten层将输入数据转成 batch_size x ? 的矩阵，然后输入到10个输出节点的全连接层。照例我们不需要制定每层输入的大小，gluon会做自动推导。


## 激活函数

**sigmoid**
```python
from mxnet import nd
def softmax(X):
    exp = nd.exp(X)
    # 假设exp是矩阵，这里对行进行求和，并要求保留axis 1，
    # 就是返回 (nrows, 1) 形状的矩阵
    partition = exp.sum(axis=1, keepdims=True)
    return exp / partition

```


**relu**
```python
def relu(X):
    return nd.maximum(X, 0)
```

## 损失函数 

**平方误差**
```python
square_loss = gluon.loss.L2Loss()

```
```python
def square_loss(yhat, y):
    # 注意这里我们把y变形成yhat的形状来避免矩阵形状的自动转换
    return (yhat - y.reshape(yhat.shape)) ** 2
 
```

**交叉熵损失**

```python
def cross_entropy(yhat, y):
    return - nd.pick(nd.log(yhat), y)
```

```python
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

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
```python

batch_size = 1
dataset_train = gluon.data.ArrayDataset(X_train, y_train)
data_iter_train = gluon.data.DataLoader(dataset_train, batch_size, shuffle=True)


```



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
```python
net.initialize()


net.collect_params().initialize(mx.init.Normal(sigma=1))
```

## SGD
**scratch版本**

```python
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

```

L2正则

```python
def L2_penalty(w, b):
    return ((w**2).sum() + b**2) / 2
```


**gluon版本**
```
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': weight_decay})
```
这里的weight_decay表明这里添加了L2正则，正则化
w = w -lr * grad - wd * w

## 训练过程


**scratch版本**


```python
    for e in range(epochs):        
        for data, label in data_iter(num_train):
            with autograd.record():
                output = net(data, lambd, *params)
                loss = square_loss(
                    output, label) + lambd * L2_penalty(*params)
            loss.backward()
            SGD(params, learning_rate)
        train_loss.append(test(params, X_train, y_train))
        test_loss.append(test(params, X_test, y_test))
```

**gluon版本**

```python

    for e in range(epochs):        
        for data, label in data_iter_train:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)            
        train_loss.append(test(net, X_train, y_train))
        test_loss.append(test(net, X_test, y_test))

```

```python
%matplotlib inline
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 120
import matplotlib.pyplot as plt

def train(X_train, X_test, y_train, y_test):
    # 线性回归模型
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(1))
    net.initialize()
    # 设一些默认参数
    learning_rate = 0.01
    epochs = 100
    batch_size = min(10, y_train.shape[0])
    dataset_train = gluon.data.ArrayDataset(X_train, y_train)
    data_iter_train = gluon.data.DataLoader(
        dataset_train, batch_size, shuffle=True)
    # 默认SGD和均方误差
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate})
    square_loss = gluon.loss.L2Loss()
    # 保存训练和测试损失
    train_loss = []
    test_loss = []
    for e in range(epochs):
        for data, label in data_iter_train:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
        train_loss.append(square_loss(
            net(X_train), y_train).mean().asscalar())
        test_loss.append(square_loss(
            net(X_test), y_test).mean().asscalar())
    # 打印结果
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['train','test'])
    plt.show()
    return ('learned weight', net[0].weight.data(),
            'learned bias', net[0].bias.data())
```
最终版
```python
def train(train_data, test_data, net, loss, trainer, ctx, num_epochs, print_batches=None):
    """Train a network"""
    print("Start training on ", ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(num_epochs):
        train_loss, train_acc, n, m = 0.0, 0.0, 0.0, 0.0
        if isinstance(train_data, mx.io.MXDataIter):
            train_data.reset()
        start = time()
        for i, batch in enumerate(train_data):
            data, label, batch_size = _get_batch(batch, ctx)
            losses = []
            with autograd.record():
                outputs = [net(X) for X in data]
                losses = [loss(yhat, y) for yhat, y in zip(outputs, label)]
            for l in losses:
                l.backward()
            train_acc += sum([(yhat.argmax(axis=1)==y).sum().asscalar()
                              for yhat, y in zip(outputs, label)])
            train_loss += sum([l.sum().asscalar() for l in losses])
            trainer.step(batch_size)
            n += batch_size
            m += sum([y.size for y in label])
            if print_batches and (i+1) % print_batches == 0:
                print("Batch %d. Loss: %f, Train acc %f" % (
                    n, train_loss/n, train_acc/m
                ))

        test_acc = evaluate_accuracy(test_data, net, ctx)
        print("Epoch %d. Loss: %.3f, Train acc %.2f, Test acc %.2f, Time %.1f sec" % (
            epoch, train_loss/n, train_acc/m, test_acc, time() - start
        ))
```

# reference

[从零开始码一个皮卡丘检测器](https://zhuanlan.zhihu.com/p/28867241)

[图片标注工具](http://blog.csdn.net/jesse_mx/article/details/53606897)

[ mxnet 使用自己的图片数据训练CNN模型](http://blog.csdn.net/u014696921/article/details/56877979)

[mxnet image API](https://mxnet.incubator.apache.org/api/python/image.html#Image)

[Create a Dataset Using RecordIO](https://mxnet.incubator.apache.org/how_to/recordio.html?highlight=recordio)

[基于MXNet gluon 的SSD模型训练](http://blog.csdn.net/muyouhang/article/details/77727381)


[解决conda与ipython notebook的python版本问题](https://groups.google.com/a/continuum.io/forum/m/#!topic/anaconda/RuSpZVPEio8)

[神经网络计算参数量的方法](http://blog.csdn.net/sunshine_in_moon/article/details/51434908)

[神经网络计算特征图的大小的方法](https://www.jianshu.com/p/c56a37093cfa)

[BN应该放在relu后](http://minibatch.net/2017/06/11/%E7%BB%8F%E5%85%B8%E8%AE%BA%E6%96%87-Batch-Normalization/)

  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1513949196873.jpg
  [2]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1514013854016.jpg
  [3]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1514012389756.jpg