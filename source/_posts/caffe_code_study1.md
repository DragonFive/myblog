---
title: caffe-源码学习——只看一篇就够了
date: 2017/6/2 15:04:12

categories:
- 计算机视觉
tags:
- 深度学习
- caffe
- deeplearning
- python
---

# 网络模型

## 激活函数 
## sigmoid
看softmax函数之前先看一下简单的sigmoid, 这个sigmoid layer的cpp实现是非常简洁的。 sigmoid的cpp文件里主要给了三个函数的实现，分别是sigmoid函数，forward_cpu, backward_cpu,在cpp文件里只实现了算法的CPU版本，至于GPU版本的函数实现放在.cu文件里面。
<!--more-->
```cpp
template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 0.5 * tanh(0.5 * x) + 0.5;
}

template <typename Dtype>
void SigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = sigmoid(bottom_data[i]);
  }
}

template <typename Dtype>
void SigmoidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      const Dtype sigmoid_x = top_data[i];
      bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x);
    }
  }
}
```
**sigmoid函数**
注意这里的sigmoid函数与标准的定义不太一样。参见ufld里面的定义
[神经网络UFLD
](http://ufldl.stanford.edu/wiki/index.php/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)

而在这里 sigmoid = 0.5 * tanh(0.5 * x) + 0.5, sigmoid变化范围为从0-1, tanh从-1到1，乘于0.5再加上0.5两者变化范围就一样了。
**forward_cpu**
这个很容易就能看懂，就是对每一个bottom元素计算sigmoid就得到来top的元素。

**backward_cpu**
发现新版的代码真的很好懂，sigmoid函数的到函数是sigmoid*(1-sigmoid) , 所以这里就直接利用来。其中propagate_down表明这一层是否要反传。


### softmaxlayer
这段代码比较复杂，比较好的注释如下。但是这个注释针对的代码版本比较老。
[caffe深度学习网络softmax层代码注释](http://blog.csdn.net/u010668083/article/details/44857455)

这里我们分析比较新的代码，当前(20170622)比较新的代码是20161202提交的代码，结构如下
```cpp

/template <typename Dtype>
void SoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  //softmax层的输出应该与输入层一致
  top[0]->ReshapeLike(*bottom[0]);
  vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));
  sum_multiplier_.Reshape(mult_dims);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  vector<int> scale_dims = bottom[0]->shape();
  // scale_尺寸为：num*1*height*width
  scale_dims[softmax_axis_] = 1;
  scale_.Reshape(scale_dims);
}
//前向计算，得到softmax的值
template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = bottom[0]->shape(softmax_axis_);
  int dim = bottom[0]->count() / outer_num_;
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  // 先找到最大值
  for (int i = 0; i < outer_num_; ++i) {//outer_num就是num输出数据的数目
    // initialize scale_data to the first plane
    caffe_copy(inner_num_, bottom_data + i * dim, scale_data);//dim表示每个数据有多少个不同类别的值.
    for (int j = 0; j < channels; j++) {
      for (int k = 0; k < inner_num_; k++) {
        scale_data[k] = std::max(scale_data[k],//每个元素表示应该是当前位置中所有类别和channel里面最大的那一个。
            bottom_data[i * dim + j * inner_num_ + k]);
      }
    }
    // subtraction 减去最大值 详细分析见后面 
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_,
        1, -1., sum_multiplier_.cpu_data(), scale_data, 1., top_data);
    // exponentiation 求指数
    caffe_exp<Dtype>(dim, top_data, top_data);
    // sum after exp 求和
    caffe_cpu_gemv<Dtype>(CblasTrans, channels, inner_num_, 1.,
        top_data, sum_multiplier_.cpu_data(), 0., scale_data);
    // division 做除法
    for (int j = 0; j < channels; j++) {
      caffe_div(inner_num_, top_data, scale_data, top_data);
      top_data += inner_num_;
    }
  }
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = top[0]->shape(softmax_axis_);
  int dim = top[0]->count() / outer_num_;
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  for (int i = 0; i < outer_num_; ++i) {
    // compute dot(top_diff, top_data) and subtract them from the bottom diff
    for (int k = 0; k < inner_num_; ++k) {
      scale_data[k] = caffe_cpu_strided_dot<Dtype>(channels,
          bottom_diff + i * dim + k, inner_num_,
          top_data + i * dim + k, inner_num_);
    }
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
        -1., sum_multiplier_.cpu_data(), scale_data, 1., bottom_diff + i * dim);
  }
  // elementwise multiplication
  caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxLayer);
#endif

INSTANTIATE_CLASS(SoftmaxLayer);

```
**caffe_cpu_gemm减法 **
在forward_cpu函数里做减法的时候调用来caffe_cpu_gemm函数，这个函数的实现在 src/caffe/util/math_functions.cpp里面
[caffecpugemm-函数](http://blog.csdn.net/seven_first/article/details/47378697#1-caffecpugemm-函数)
```cpp
template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}
```
>功能： C=alpha*A*B+beta*C 
A,B,C 是输入矩阵（一维数组格式） 
CblasRowMajor :数据是行主序的（二维数据也是用一维数组储存的） 
TransA, TransB：是否要对A和B做转置操作（CblasTrans CblasNoTrans） 
M： A、C 的行数 
N： B、C 的列数 
K： A 的列数， B 的行数 
lda ： A的列数（不做转置）行数（做转置） 
ldb： B的列数（不做转置）行数（做转置）

所以这里求减法：
```cpp
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_,
        1, -1., sum_multiplier_.cpu_data(), scale_data, 1., top_data);
```
就是： $top\_data = top\_data - 1* sum\_multiplier\_.cpu\_data()*scale\_data$， 这里用top_data来减而不用bottom一方面是因为bottom是const的，取的是cpu_data(), 而top_data是mutable_cpu_data,另一方面之前已经把数据从bottom拷贝到top里面去了。

**caffe_exp指数**
caffe_exp函数是用来求指数的，其中一个实现是这样的。
```cpp
template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}
```
> 功能：  y[i] = exp(a[i] )

所以， forward_cpu里面的指数就很容易理解了。
```
caffe_exp<Dtype>(dim, top_data, top_data);
```
top_data[i] = exp(topdata[i])
**caffe_cpu_gemv求和**
```cpp
template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}
```
>功能： y=alpha*A*x+beta*y 
其中X和Y是向量，A 是矩阵 
M：A 的行数 
N：A 的列数 
cblas_sgemv 中的 参数1 表示对X和Y的每个元素都进行操作

forward_cpu里面的求和就很容易理解了
```cpp
    // sum after exp 求和
    caffe_cpu_gemv<Dtype>(CblasTrans, channels, inner_num_, 1.,
        top_data, sum_multiplier_.cpu_data(), 0., scale_data);
```

$scale\_data =\sum top\_data[i]*sum\_multiplier\_.cpu\_data()[i];$




**caffe_div除法**
```cpp
template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
  vsDiv(n, a, b, y);
}
```
> 功能 y[i] = a[i] / b[i]

```cpp
    // division 做除法
    for (int j = 0; j < channels; j++) {
      caffe_div(inner_num_, top_data, scale_data, top_data);
      top_data += inner_num_;
```
$top\_data[i] = top\_data[i] / scale\_data[i];$

**反向传播公式推导**
[Caffe Softmax层的实现原理,知乎](https://www.zhihu.com/question/28927103)

看完softmax layer的实现，我们再来看一下SoftmaxWithLossLayer的代码实现。



## 卷积层

### 计算量与参数量
每个样本做一次前向传播时卷积计算量为： $ i* j*M*N*K*L  $ ，其中$i*j$是卷积核的大小，$M*L$是输出特征图的大小，K是输入特征图通道数，L是输出特征图通道数。

参数量为：$ i*J*K*L $

所以有个比例叫做计算量参数量之比 CPR，如果在前馈时每个批次batch_size = B, 则表示将B个输入合并成一个矩阵进行计算，那么相当于每次的输出特征图增大来B倍，所以CPR提升来B倍，也就是，每次计算的时候参数重复利用率提高来B倍。

卷积层：局部互连，权值共享，



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
forward_cpu_gemm:猜测可能是前馈过程计算weight部分，来看看CPP里面的实现吧。

在BaseConvolutionLayer中的卷积的实现中有一个重要的函数就是**im2col以及col2im，im2colnd以及col2imnd**。前面的两个函数是二维卷积的正向和逆向过程，而后面的两个函数是n维卷积的正向和逆向过程。
```cpp
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
	  // 如果没有1x1卷积，也没有skip_im2col  
      // 则使用conv_im2col_cpu对使用卷积核滑动过程中的每一个kernel大小的图像块  
      // 变成一个列向量，形成一个height=kernel_dim_  
      // width = 卷积后图像heght*卷积后图像width  
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

```
**参考资料**
[ caffe代码阅读10：Caffe中卷积的实现细节](http://blog.csdn.net/xizero00/article/details/51049858)



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

