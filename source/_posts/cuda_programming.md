---
title: cuda_programming
date: 2017/8/16 17:38:58

categories:
- 深度学习
tags:
- deeplearning
- cuda
- GPU编程
- caffe
---

获取设备->分配显存->数据传输host to device->kernel函数执行->数据传输device to host->释放显存->重置设备

```cpp
cudaSetDevice(0);
cudaMalloc((void **)&d_a, sizeof(float)*n);
cudaMemcpy(d_a,a,sizeof(float)*n,cudaMemcpyHostToDevice);
gpu_kernel<<<blocks,threads>>> (***);
cudaMemcpy(a,d_a,sizeof(float)*n,cudaMemcpyDeviceToHost);
cudaFree(d_a);
cudaDeviceReset();


```
<!--more-->


# CUDA C编程


## cuda运行时函数

cuda运行时提供了丰富的函数，功能涉及设备管理、存储管理、数据传输、线程管理、流管理、事件管理、纹理管理、执行控制等。


### 设备管理函数 

函数声明一般这样 
```
extern __host__ cudaError_t CUDARTAPI 函数名(参数列表)
```

**cudaGetDeviceCount**
获得计算能力大于等于1.0的GPU数量
```cpp
int count;
cudaGetDeviceCount(&count);
```

**cudaSetDevice**
设置使用的GPU索引号，如果不设置默认使用0号GPU

```cpp
int gpuid = 0;
cudaSetDevice(gpuid);

```

**cudaGetDevice**
获得当前线程的GPU设备号

```cpp
int gpuid;
cudaGetDevice(&gpuid);

```
**cudaSetValidDevices**

设置多个device,len表示签名设备号数组的长度;

```cpp
cudaSetValidDevices(int &device_arr, int len);
```


### 存储管理函数 

**cudaMalloc**

在GPU上分配大小为size的现行存储空间，起始地址为 *devPtr
```cpp

cudaMalloc(void **devPtr,size_t size);

```
**cudaMallocPitch**

在GPU上分配大小为PitchxHight的逻辑2D线性存储空间，首地址为```*devPtr```, 其中Pitch是返回的width对齐后的存储空间的宽度

```cpp
cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height);
```
```
devPtr[x] = devPtr[rowid*pitch+column]
```


**cudaFree**
清空指定的GPU存储区域，可释放cudaMalloc和cudaMallocPitch分类的GPU存储区域

```cpp

cudaFree(void *devPtr);

```

**cudaMemset**
将GPU端的devPtr指针指向的count长度的存储空间赋值为value.
```cpp
cudaMemset(void 8DevPTR， int value,size_t count);
```

**cudaHostAlloc**
在主机端(CPU)根据flag值来分配页锁定存储, 

```cpp
cudaHostAlloc(void **pHost, size_t size, usigned int flags);
```

flags可以有四种取值

```cpp
cudaHostAllocDefault   分配默认存储
cudaHostAllocPortable  分配的存储可以被cuda索引
cudaHostAllocMapped 分配的存储映射到GPU
。。。

```

### 数据传输函数 

**cudaMemcpy**
```cpp
cudaMemcpy(void * dst, const void *src, size_t count, enum cudaMemcpyKind kind);
```

主机(cpu内存)与设备间的数据传输函数，源地址是```*src```，目标地址是```*dst```,传输长度为```count```,kind指定了传输的方向，kind可选值域如下：
```cpp
cudaMemcpyHostToHost = 0;
cudaMemcpyHostToDevice = 0;
cudaMemcpyDeviceToHost = 0;
cudaMemcpyDeviceToDevice = 0;
```


# reference

《GPU编程与优化》——方民权