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
先用grep函数在caffe根目录下搜索一下包含ConvolutionLayer的文件有哪些，然后从头文件入手慢慢分析，下面是结果，精简来一些无效成分
```
root@aa0afc645153:caffe# grep -n -H -R "ConvolutionLayer"

*************************************************************************************************************
************************************************首先是头文件信息*****************************************
*************************************************************************************************************

build/install/include/caffe/layer_factory.hpp:31: * (for example, when your layer has multiple backends, see GetConvolutionLayer
build/install/include/caffe/layers/base_conv_layer.hpp:15: *        ConvolutionLayer and DeconvolutionLayer.
build/install/include/caffe/layers/base_conv_layer.hpp:18:class BaseConvolutionLayer : public Layer<Dtype> {
build/install/include/caffe/layers/base_conv_layer.hpp:20:  explicit BaseConvolutionLayer(const LayerParameter& param)
build/install/include/caffe/layers/deconv_layer.hpp:17: *        opposite sense as ConvolutionLayer.
build/install/include/caffe/layers/deconv_layer.hpp:19: *   ConvolutionLayer computes each output value by dotting an input window with
build/install/include/caffe/layers/deconv_layer.hpp:22: *   DeconvolutionLayer is ConvolutionLayer with the forward and backward passes
build/install/include/caffe/layers/deconv_layer.hpp:24: *   parameters, but they take the opposite sense as in ConvolutionLayer (so
build/install/include/caffe/layers/deconv_layer.hpp:29:class DeconvolutionLayer : public BaseConvolutionLayer<Dtype> {
build/install/include/caffe/layers/deconv_layer.hpp:32:      : BaseConvolutionLayer<Dtype>(param) {}
build/install/include/caffe/layers/im2col_layer.hpp:14: *        column vectors.  Used by ConvolutionLayer to perform convolution
build/install/include/caffe/layers/conv_layer.hpp:31:class ConvolutionLayer : public BaseConvolutionLayer<Dtype> {
build/install/include/caffe/layers/conv_layer.hpp:35:   *    with ConvolutionLayer options:
build/install/include/caffe/layers/conv_layer.hpp:64:  explicit ConvolutionLayer(const LayerParameter& param)
build/install/include/caffe/layers/conv_layer.hpp:65:      : BaseConvolutionLayer<Dtype>(param) {}
build/install/include/caffe/layers/cudnn_conv_layer.hpp:16: * @brief cuDNN implementation of ConvolutionLayer.
build/install/include/caffe/layers/cudnn_conv_layer.hpp:17: *        Fallback to ConvolutionLayer for CPU mode.
build/install/include/caffe/layers/cudnn_conv_layer.hpp:30:class CuDNNConvolutionLayer : public ConvolutionLayer<Dtype> {
build/install/include/caffe/layers/cudnn_conv_layer.hpp:32:  explicit CuDNNConvolutionLayer(const LayerParameter& param)
build/install/include/caffe/layers/cudnn_conv_layer.hpp:33:      : ConvolutionLayer<Dtype>(param), handles_setup_(false) {}
build/install/include/caffe/layers/cudnn_conv_layer.hpp:38:  virtual ~CuDNNConvolutionLayer();
Binary file build/install/lib/libcaffe.so.1.0.0-rc5 matches
Binary file build/install/lib/libcaffe.so matches
include/caffe/layer_factory.hpp:31: * (for example, when your layer has multiple backends, see GetConvolutionLayer
include/caffe/layers/base_conv_layer.hpp:15: *        ConvolutionLayer and DeconvolutionLayer.
include/caffe/layers/base_conv_layer.hpp:18:class BaseConvolutionLayer : public Layer<Dtype> {
include/caffe/layers/base_conv_layer.hpp:20:  explicit BaseConvolutionLayer(const LayerParameter& param)
include/caffe/layers/deconv_layer.hpp:17: *        opposite sense as ConvolutionLayer.
include/caffe/layers/deconv_layer.hpp:19: *   ConvolutionLayer computes each output value by dotting an input window with
include/caffe/layers/deconv_layer.hpp:22: *   DeconvolutionLayer is ConvolutionLayer with the forward and backward passes
include/caffe/layers/deconv_layer.hpp:24: *   parameters, but they take the opposite sense as in ConvolutionLayer (so
include/caffe/layers/deconv_layer.hpp:29:class DeconvolutionLayer : public BaseConvolutionLayer<Dtype> {
include/caffe/layers/deconv_layer.hpp:32:      : BaseConvolutionLayer<Dtype>(param) {}
include/caffe/layers/im2col_layer.hpp:14: *        column vectors.  Used by ConvolutionLayer to perform convolution
include/caffe/layers/conv_layer.hpp:31:class ConvolutionLayer : public BaseConvolutionLayer<Dtype> {
include/caffe/layers/conv_layer.hpp:35:   *    with ConvolutionLayer options:
include/caffe/layers/conv_layer.hpp:64:  explicit ConvolutionLayer(const LayerParameter& param)
include/caffe/layers/conv_layer.hpp:65:      : BaseConvolutionLayer<Dtype>(param) {}
include/caffe/layers/cudnn_conv_layer.hpp:16: * @brief cuDNN implementation of ConvolutionLayer.
include/caffe/layers/cudnn_conv_layer.hpp:17: *        Fallback to ConvolutionLayer for CPU mode.
include/caffe/layers/cudnn_conv_layer.hpp:30:class CuDNNConvolutionLayer : public ConvolutionLayer<Dtype> {
include/caffe/layers/cudnn_conv_layer.hpp:32:  explicit CuDNNConvolutionLayer(const LayerParameter& param)
include/caffe/layers/cudnn_conv_layer.hpp:33:      : ConvolutionLayer<Dtype>(param), handles_setup_(false) {}
include/caffe/layers/cudnn_conv_layer.hpp:38:  virtual ~CuDNNConvolutionLayer();



docs/tutorial/layers/convolution.md:8:* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1ConvolutionLayer.html)
src/caffe/layers/conv_layer.cu:8:void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
src/caffe/layers/conv_layer.cu:26:void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
src/caffe/layers/conv_layer.cu:58:INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);
src/caffe/layers/cudnn_conv_layer.cu:11:void CuDNNConvolutionLayer<Dtype>::Forward_gpu(
src/caffe/layers/cudnn_conv_layer.cu:49:void CuDNNConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
src/caffe/layers/cudnn_conv_layer.cu:115:INSTANTIATE_LAYER_GPU_FUNCS(CuDNNConvolutionLayer);



*************************************************************************************************************
******************************************接下来是CPP文件信息*****************************************
*************************************************************************************************************


src/caffe/layers/cudnn_conv_layer.cpp:18:void CuDNNConvolutionLayer<Dtype>::LayerSetUp(
src/caffe/layers/cudnn_conv_layer.cpp:20:  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
src/caffe/layers/cudnn_conv_layer.cpp:91:void CuDNNConvolutionLayer<Dtype>::Reshape(
src/caffe/layers/cudnn_conv_layer.cpp:93:  ConvolutionLayer<Dtype>::Reshape(bottom, top);
src/caffe/layers/cudnn_conv_layer.cpp:235:CuDNNConvolutionLayer<Dtype>::~CuDNNConvolutionLayer() {
src/caffe/layers/cudnn_conv_layer.cpp:266:INSTANTIATE_CLASS(CuDNNConvolutionLayer);
src/caffe/layers/base_conv_layer.cpp:12:void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
src/caffe/layers/base_conv_layer.cpp:185:void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
src/caffe/layers/base_conv_layer.cpp:256:void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
src/caffe/layers/base_conv_layer.cpp:274:void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
src/caffe/layers/base_conv_layer.cpp:282:void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
src/caffe/layers/base_conv_layer.cpp:300:void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
src/caffe/layers/base_conv_layer.cpp:316:void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
src/caffe/layers/base_conv_layer.cpp:325:void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
src/caffe/layers/base_conv_layer.cpp:343:void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
src/caffe/layers/base_conv_layer.cpp:351:void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
src/caffe/layers/base_conv_layer.cpp:369:void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
src/caffe/layers/base_conv_layer.cpp:385:void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
src/caffe/layers/base_conv_layer.cpp:393:INSTANTIATE_CLASS(BaseConvolutionLayer);
src/caffe/layers/conv_layer.cpp:8:void ConvolutionLayer<Dtype>::compute_output_shape() {
src/caffe/layers/conv_layer.cpp:25:void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
src/caffe/layers/conv_layer.cpp:43:void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
src/caffe/layers/conv_layer.cpp:76:STUB_GPU(ConvolutionLayer);
src/caffe/layers/conv_layer.cpp:79:INSTANTIATE_CLASS(ConvolutionLayer);
src/caffe/test/test_convolution_layer.cpp:151:class ConvolutionLayerTest : public MultiDeviceTest<TypeParam> {
src/caffe/test/test_convolution_layer.cpp:155:  ConvolutionLayerTest()
src/caffe/test/test_convolution_layer.cpp:171:  virtual ~ConvolutionLayerTest() {
src/caffe/test/test_convolution_layer.cpp:193:TYPED_TEST_CASE(ConvolutionLayerTest, TestDtypesAndDevices);
src/caffe/test/test_convolution_layer.cpp:195:TYPED_TEST(ConvolutionLayerTest, TestSetup) {
src/caffe/test/test_convolution_layer.cpp:206:      new ConvolutionLayer<Dtype>(layer_param));
src/caffe/test/test_convolution_layer.cpp:219:  layer.reset(new ConvolutionLayer<Dtype>(layer_param));
src/caffe/test/test_convolution_layer.cpp:231:TYPED_TEST(ConvolutionLayerTest, TestSimpleConvolution) {
src/caffe/test/test_convolution_layer.cpp:245:      new ConvolutionLayer<Dtype>(layer_param));
src/caffe/test/test_convolution_layer.cpp:267:TYPED_TEST(ConvolutionLayerTest, TestDilatedConvolution) {
src/caffe/test/test_convolution_layer.cpp:289:      new ConvolutionLayer<Dtype>(layer_param));
src/caffe/test/test_convolution_layer.cpp:311:TYPED_TEST(ConvolutionLayerTest, Test0DConvolution) {
src/caffe/test/test_convolution_layer.cpp:322:      new ConvolutionLayer<Dtype>(layer_param));
src/caffe/test/test_convolution_layer.cpp:349:TYPED_TEST(ConvolutionLayerTest, TestSimple3DConvolution) {
src/caffe/test/test_convolution_layer.cpp:374:      new ConvolutionLayer<Dtype>(layer_param));
src/caffe/test/test_convolution_layer.cpp:396:TYPED_TEST(ConvolutionLayerTest, TestDilated3DConvolution) {
src/caffe/test/test_convolution_layer.cpp:421:      new ConvolutionLayer<Dtype>(layer_param));
src/caffe/test/test_convolution_layer.cpp:443:TYPED_TEST(ConvolutionLayerTest, Test1x1Convolution) {
src/caffe/test/test_convolution_layer.cpp:455:      new ConvolutionLayer<Dtype>(layer_param));
src/caffe/test/test_convolution_layer.cpp:470:TYPED_TEST(ConvolutionLayerTest, TestSimpleConvolutionGroup) {
src/caffe/test/test_convolution_layer.cpp:483:      new ConvolutionLayer<Dtype>(layer_param));
src/caffe/test/test_convolution_layer.cpp:498:TYPED_TEST(ConvolutionLayerTest, TestSobelConvolution) {
src/caffe/test/test_convolution_layer.cpp:519:      new ConvolutionLayer<Dtype>(layer_param));
src/caffe/test/test_convolution_layer.cpp:552:  layer.reset(new ConvolutionLayer<Dtype>(layer_param));
src/caffe/test/test_convolution_layer.cpp:574:  layer.reset(new ConvolutionLayer<Dtype>(layer_param));
src/caffe/test/test_convolution_layer.cpp:591:TYPED_TEST(ConvolutionLayerTest, TestNDAgainst2D) {
src/caffe/test/test_convolution_layer.cpp:621:    ConvolutionLayer<Dtype> layer(layer_param);
src/caffe/test/test_convolution_layer.cpp:642:    ConvolutionLayer<Dtype> layer_2d(layer_param);
src/caffe/test/test_convolution_layer.cpp:673:    ConvolutionLayer<Dtype> layer_nd(layer_param);
src/caffe/test/test_convolution_layer.cpp:709:TYPED_TEST(ConvolutionLayerTest, TestGradient) {
src/caffe/test/test_convolution_layer.cpp:721:  ConvolutionLayer<Dtype> layer(layer_param);
src/caffe/test/test_convolution_layer.cpp:727:TYPED_TEST(ConvolutionLayerTest, TestDilatedGradient) {
src/caffe/test/test_convolution_layer.cpp:745:  ConvolutionLayer<Dtype> layer(layer_param);
src/caffe/test/test_convolution_layer.cpp:751:TYPED_TEST(ConvolutionLayerTest, TestGradient3D) {
src/caffe/test/test_convolution_layer.cpp:773:  ConvolutionLayer<Dtype> layer(layer_param);
src/caffe/test/test_convolution_layer.cpp:779:TYPED_TEST(ConvolutionLayerTest, Test1x1Gradient) {
src/caffe/test/test_convolution_layer.cpp:791:  ConvolutionLayer<Dtype> layer(layer_param);
src/caffe/test/test_convolution_layer.cpp:797:TYPED_TEST(ConvolutionLayerTest, TestGradientGroup) {
src/caffe/test/test_convolution_layer.cpp:808:  ConvolutionLayer<Dtype> layer(layer_param);
src/caffe/test/test_convolution_layer.cpp:817:class CuDNNConvolutionLayerTest : public GPUDeviceTest<Dtype> {
src/caffe/test/test_convolution_layer.cpp:819:  CuDNNConvolutionLayerTest()
src/caffe/test/test_convolution_layer.cpp:835:  virtual ~CuDNNConvolutionLayerTest() {
src/caffe/test/test_convolution_layer.cpp:857:TYPED_TEST_CASE(CuDNNConvolutionLayerTest, TestDtypes);
src/caffe/test/test_convolution_layer.cpp:859:TYPED_TEST(CuDNNConvolutionLayerTest, TestSetupCuDNN) {
src/caffe/test/test_convolution_layer.cpp:871:      new CuDNNConvolutionLayer<TypeParam>(layer_param));
src/caffe/test/test_convolution_layer.cpp:884:  layer.reset(new CuDNNConvolutionLayer<TypeParam>(layer_param));
src/caffe/test/test_convolution_layer.cpp:896:TYPED_TEST(CuDNNConvolutionLayerTest, TestSimpleConvolutionCuDNN) {
src/caffe/test/test_convolution_layer.cpp:909:      new CuDNNConvolutionLayer<TypeParam>(layer_param));
src/caffe/test/test_convolution_layer.cpp:931:TYPED_TEST(CuDNNConvolutionLayerTest, TestSimpleConvolutionGroupCuDNN) {
src/caffe/test/test_convolution_layer.cpp:943:      new CuDNNConvolutionLayer<TypeParam>(layer_param));
src/caffe/test/test_convolution_layer.cpp:958:TYPED_TEST(CuDNNConvolutionLayerTest, TestSobelConvolutionCuDNN) {
src/caffe/test/test_convolution_layer.cpp:979:      new CuDNNConvolutionLayer<TypeParam>(layer_param));
src/caffe/test/test_convolution_layer.cpp:1012:  layer.reset(new CuDNNConvolutionLayer<TypeParam>(layer_param));
src/caffe/test/test_convolution_layer.cpp:1034:  layer.reset(new CuDNNConvolutionLayer<TypeParam>(layer_param));
src/caffe/test/test_convolution_layer.cpp:1051:TYPED_TEST(CuDNNConvolutionLayerTest, TestGradientCuDNN) {
src/caffe/test/test_convolution_layer.cpp:1062:  CuDNNConvolutionLayer<TypeParam> layer(layer_param);
src/caffe/test/test_convolution_layer.cpp:1068:TYPED_TEST(CuDNNConvolutionLayerTest, TestGradientGroupCuDNN) {
src/caffe/test/test_convolution_layer.cpp:1078:  CuDNNConvolutionLayer<TypeParam> layer(layer_param);
src/caffe/test/test_deconvolution_layer.cpp:15:// Since ConvolutionLayerTest checks the shared conv/deconv code in detail,
src/caffe/layer_factory.cpp:38:shared_ptr<Layer<Dtype> > GetConvolutionLayer(
src/caffe/layer_factory.cpp:59:    return shared_ptr<Layer<Dtype> >(new ConvolutionLayer<Dtype>(param));
src/caffe/layer_factory.cpp:66:    return shared_ptr<Layer<Dtype> >(new CuDNNConvolutionLayer<Dtype>(param));
src/caffe/layer_factory.cpp:74:REGISTER_LAYER_CREATOR(Convolution, GetConvolutionLayer);

```



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

