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

### 计算量与参数量
每个样本做一次前向传播时卷积计算量为： $ i* j*M*N*K*L  $ ，其中$i*j$是卷积核的大小，$M*L$是输出特征图的大小，K是输入特征图通道数，L是输出特征图通道数。

参数量为：$ i*J*K*L $

所以有个比例叫做计算量参数量之比 CPR，如果在前馈时每个批次batch_size = B, 则表示将B个输入合并成一个矩阵进行计算，那么相当于每次的输出特征图增大来B倍，所以CPR提升来B倍，也就是，每次计算的时候参数重复利用率提高来B倍。

### 源码学习
先用grep函数在caffe根目录下搜索一下包含ConvolutionLayer的文件有哪些，然后从头文件入手慢慢分析，下面是结果，精简来一些无效成分，在caffe的include文件夹下执行：
```bash
grep -n -H -R "ConvolutionLayer"
```
-n表示显示行号，-H表示显示文件名，-R表示递归查找 后面部分表示查找的内容
结果如下
```bash
caffe/layer_factory.hpp:31: * (for example, when your layer has multiple backends, see GetConvolutionLayer
caffe/layers/base_conv_layer.hpp:15: *        ConvolutionLayer and DeconvolutionLayer.
caffe/layers/base_conv_layer.hpp:18:class BaseConvolutionLayer : public Layer<Dtype> {
caffe/layers/base_conv_layer.hpp:20:  explicit BaseConvolutionLayer(const LayerParameter& param)
caffe/layers/deconv_layer.hpp:17: *        opposite sense as ConvolutionLayer.
caffe/layers/deconv_layer.hpp:19: *   ConvolutionLayer computes each output value by dotting an input window with
caffe/layers/deconv_layer.hpp:22: *   DeconvolutionLayer is ConvolutionLayer with the forward and backward passes
caffe/layers/deconv_layer.hpp:24: *   parameters, but they take the opposite sense as in ConvolutionLayer (so
caffe/layers/deconv_layer.hpp:29:class DeconvolutionLayer : public BaseConvolutionLayer<Dtype> {
caffe/layers/deconv_layer.hpp:32:      : BaseConvolutionLayer<Dtype>(param) {}
caffe/layers/im2col_layer.hpp:14: *        column vectors.  Used by ConvolutionLayer to perform convolution
caffe/layers/conv_layer.hpp:31:class ConvolutionLayer : public BaseConvolutionLayer<Dtype> {
caffe/layers/conv_layer.hpp:35:   *    with ConvolutionLayer options:
caffe/layers/conv_layer.hpp:64:  explicit ConvolutionLayer(const LayerParameter& param)
caffe/layers/conv_layer.hpp:65:      : BaseConvolutionLayer<Dtype>(param) {}
caffe/layers/cudnn_conv_layer.hpp:16: * @brief cuDNN implementation of ConvolutionLayer.
caffe/layers/cudnn_conv_layer.hpp:17: *        Fallback to ConvolutionLayer for CPU mode.
caffe/layers/cudnn_conv_layer.hpp:30:class CuDNNConvolutionLayer : public ConvolutionLayer<Dtype> {
caffe/layers/cudnn_conv_layer.hpp:32:  explicit CuDNNConvolutionLayer(const LayerParameter& param)
caffe/layers/cudnn_conv_layer.hpp:33:      : ConvolutionLayer<Dtype>(param), handles_setup_(false) {}
caffe/layers/cudnn_conv_layer.hpp:38:  virtual ~CuDNNConvolutionLayer();

```
主要有三个类包含这个卷积层的实现：
base_conv_layer：主要是卷积层基类的实现
deconv_layer： 目测是反向传播时候的卷积层的逆向过程
cudnn_conv_layer：目测是cudnn实现的卷积层版本继承自BaseConvolutionLayer,GPU版本

接下来我们就打开这三个文件，跳转到相关行，详细看一下。
```
class BaseConvolutionLayer : public Layer<Dtype> {
 public:
  explicit BaseConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

 protected:
  // Helper functions that abstract away the column buffer and gemm arguments.
  // The last argument in forward_cpu_gemm is so that we can skip the im2col if
  // we just called weight_cpu_gemm with the same input.
  void forward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  void forward_cpu_bias(Dtype* output, const Dtype* bias);
  void backward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output);
  void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype*
      weights);
  void backward_cpu_bias(Dtype* bias, const Dtype* input);

#ifndef CPU_ONLY
  void forward_gpu_gemm(const Dtype* col_input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  void forward_gpu_bias(Dtype* output, const Dtype* bias);
  void backward_gpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* col_output);
  void weight_gpu_gemm(const Dtype* col_input, const Dtype* output, Dtype*
      weights);
  void backward_gpu_bias(Dtype* bias, const Dtype* input);
#endif
```
这里给出来CPU和GPU版本的代码的声明，这些代码比较底层，先放一放以后再看。
forward_cpu_gemm:猜测可能是前馈过程计算weight部分，来看看CPP里面的实现吧


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

