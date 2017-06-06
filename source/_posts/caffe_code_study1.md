---
title: caffe-源码学习——只看一篇就够了
tdate: 2017/6/2 15:04:12

categories:
- 计算机视觉
tags:
- 深度学习
- caffe
- deeplearning
- python
---

# 网络模型
## 卷积层

每个样本做一次前向传播时卷积计算量为： $$ i*j  $$




# 数据集

## 生成数据集的均值文件

这里计算图像的均值使用的是pper_image_mean方法，在natural images上训练的时候，这种方式比较好，以imagenet数据集为例，caffe在使用imagenet数据集的时候需要计算均值文件，详细见 [python-caffe
](https://github.com/DragonFive/deep-learning-exercise/blob/master/caffe_python1.ipynb)

## caffe-blob
[【Caffe代码解析】Blob](http://blog.csdn.net/chenriwei2/article/details/46367023)
[caffe源码分析--Blob类代码研究](http://blog.csdn.net/lingerlanlan/article/details/24379689)
[Caffe源码解析1：Blob](http://www.cnblogs.com/louyihang-loves-baiyan/p/5149628.html)

### 结构体分析
Blob 是Caffe作为数据传输的媒介，无论是网络权重参数，还是输入数据，都是转化为Blob数据结构来存储，网络，求解器等都是直接与此结构打交道的。

4纬的结构体（包含数据和梯度)，其4维结构通过shape属性得以计算出来.

**成员变量**

```cpp
 protected:
  shared_ptr<SyncedMemory> data_;// 存放数据
  shared_ptr<SyncedMemory> diff_;//存放梯度
  vector<int> shape_; //存放形状
  int count_; //数据个数
  int capacity_; //数据容量
```

**成员函数**

```
  const Dtype* cpu_data() const;			 //cpu使用的数据
  void set_cpu_data(Dtype* data);		//用数据块的值来blob里面的data。
  const Dtype* gpu_data() const;		//返回不可更改的指针，下同
  const Dtype* cpu_diff() const;
  const Dtype* gpu_diff() const;
  Dtype* mutable_cpu_data();    		//返回可更改的指针，下同
  Dtype* mutable_gpu_data();
  Dtype* mutable_cpu_diff();
  Dtype* mutable_gpu_diff();
  
  int offset(const int n, const int c = 0, const int h = 0,const int w = 0) const
// 通过n,c,h,w 4个参数来计算一维向量的偏移量。

Dtype data_at(const int n, const int c, const int h,const int w) const//通过n,c,h,w 4个参数来来获取该向量位置上的值。

Dtype diff_at(const int n, const int c, const int h,const int w) const//同上

inline const shared_ptr<SyncedMemory>& data() const {
    CHECK(data_);
    return data_;			//返回数据，不能修改
  }

inline const shared_ptr<SyncedMemory>& diff() const {
    CHECK(diff_);
    return diff_;			//返回梯度，不能修改
  }

Reshape(...)//reshape 有多种多态的实现，可以是四个数字，长度为四的vector，其它blob等。

if (count_ > capacity_) {
    capacity_ = count_;
    data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
  }//当空间不够的时候，需要扩大容量，reset。

```
函数名中带mutable的表示可以对返回的指针内容进行修改。



# caffe学习资料收集 

[深度学习Caffe系列教程集合](https://absentm.github.io/2016/05/14/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0Caffe%E7%B3%BB%E5%88%97%E6%95%99%E7%A8%8B%E9%9B%86%E5%90%88/)

[caffe代码阅读1：blob的实现细节-2016.3.14](http://blog.csdn.net/xizero00/article/details/50886829)

[甘宇飞](https://yufeigan.github.io/)

[计算机视觉战队](https://zhuanlan.zhihu.com/Edison-G)

[caffe源码简单解析——Layer层  ](http://blog.163.com/yuyang_tech/blog/static/2160500832015713105052452/)

[Caffe代码导读（0）：路线图](http://blog.csdn.net/kkk584520/article/details/41681085)

[知乎问题-深度学习caffe的代码怎么读？](https://www.zhihu.com/question/27982282)

[Caffe源码解析1：Blob](http://www.cnblogs.com/louyihang-loves-baiyan/p/5149628.html)

[深度学习大讲堂——深度学习框架Caffe源码解析](https://zhuanlan.zhihu.com/p/24343706)

[Caffe代码夜话1](https://zhuanlan.zhihu.com/p/24709689)

[Caffe源码分析（一）](http://blog.leanote.com/post/fishing_piggy/Caffe%E6%BA%90%E7%A0%81%E5%88%86%E6%9E%90%EF%BC%88%E4%B8%80%EF%BC%89)

