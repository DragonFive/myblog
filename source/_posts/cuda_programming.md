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
**cudaFree**
清空指定的GPU存储区域，可释放cudaMalloc和cudaMallocPitch分类的GPU存储区域

```cpp

cudaFree(void *devPtr);

```


# reference

《GPU编程与优化》——方民权