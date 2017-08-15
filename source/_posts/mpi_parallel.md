---
title: 深度学习框架的并行优化方法小结
date: 2017/8/11 17:38:58

categories:
- 深度学习
tags:
- deeplearning
- mpi
- caffe
---

 
# 并行程序

## 并行实现实现方式： 
1. 任务并行：将任务分配带若干计算核上; 
2. **数据并行**：将数据进行分割，然后由不同的计算核进行处理，**每个核在规模相当的数据集上大致采用相同的操作**。这不由使我想到了**CAFFE中的对GPU的运用来实现并行训练**的思路，就是将数据集进行分割，每个GPU并行处理各自对应的数据集。 

多指令多数据流又分为分布式内存系统和共享内存系统。 
**分布式内存系统**： 
每个处理器由独立的内存，通过**消息传递函数**来通信。 
共享式内存系统： 
多个处理器能访问内存系统中的相同内存，通过共享内存进行通信。 
**MPI**就是用来在分布式系统中为各处理器进行消息传递的API。 

各个核能够直接访问自己的内存，而运行在不同核之间的进程需要交换内存数据的时候，只能通过消息传递API来实现。消息传递的API至少要提供一个发送函数和接收函数。**进程之间通过它们的序号（rank）**进行识别。


## 并行程序的流程 
a、任务或者**数据划分**，就是要识别出任务中可以进行并行执行的部分。 
b、不同任务之间的**通信**; 
c、**聚合**，将任务和通信进行集合，聚合成更大的任务; 
d、**分配**，将聚合的任务分配到进程或线程中。



1、MPI是进程级别的，通过通信在进程之间进行消息传递。 
2、编程模型复杂： 
a、需要进行任务划分; 
b、通信延迟和负载不均衡;通信延迟很好理解，负载不均衡是因为分布式的系统，每个处理的任务量不同？待进一步的解释 ；
c、可靠性差，一个进程出错，整个程序崩溃。第一感觉就是这简直是MPI的命门。在分布式系统中某一个进程出错是很容易的，为MPI的命运担忧。

# 通信函数 

## 一般函数 

```cpp
int MPI_Send (void *buf, int count, MPI_Datatype datatype,int dest, int tag,MPI_Comm comm)
```
参数buf为发送缓冲区；count为发送的数据个数；datatype为发送的数据类型；dest为消息的目的地址(进程号)，其取值范围为0到np－1间的整数(np代表通信器comm中的进程数) 或MPI_PROC_NULL；tag为消息标签，其取值范围为0到MPI_TAG_UB间的整数；**comm为通信器**

```cpp
mpi_recv:接收信息   MPI_Probe：预测一下消息的size
```

## mpi聚合通信 
collective communication。聚合通信是在通信子中的所有的进程都参与的通信方式。 

### 同步 MPI_Barrier
MPI_Barrier就是这样的一个函数，他确保除非所有的进程同时调用，否则他不会允许任何进程通过这个节点
对于所有的进程来说，聚合通信必然包含了一个**同步点**。也就是说所有的进程必须在他们又一次执行新动作之前都到达某个点。这跟GPU中线程同步的概念很相似，很好理解。

### 广播 
广播机制： 
一个进程将相同的数据发送给通信子中所有的进程。该机制最主要的应用是将输入数据发送给并行程序，或者将**配置参数**发送给所有的进程

```cpp

MPI_Bcast(
    void* data,//数据
    int count,//数据个数
    MPI_Datatype datatype,
    int root,//根进程编号
    MPI_Comm communicator)
```

### MPI_Scatter 数据分发

MPI_Scatter与MPI_Bcast非常相似，都是**一对多**的通信方式，不同的是后者的**0号进程**将相同的信息发送给所有的进程，而前者则是将一段array 的不同部分发送给所有的进程

![scatter与bcast的区别][1]


```cpp

MPI_Scatter(
    void* send_data,//存储在0号进程的数据，array
    int send_count,//具体需要给每个进程发送的数据的个数
    //如果send_count为1，那么每个进程接收1个数据；如果为2，那么每个进程接收2个数据
    MPI_Datatype send_datatype,//发送数据的类型
    void* recv_data,//接收缓存，缓存 recv_count个数据
    int recv_count,
    MPI_Datatype recv_datatype,
    int root,//root进程的编号
    MPI_Comm communicator)
```

通常send_count等于array的元素个数除以进程个数。

### MPI_Gather
MPI_Gather和MPI_scatter刚好相反，他的作用是从所有的进程中将每个进程的数据集中到根进程中，**同样根据进程的编号对array元素排序**

![mpi_gather][2]


```cpp

MPI_Gather(
    void* send_data,
    int send_count,
    MPI_Datatype send_datatype,
    void* recv_data,
    int recv_count,//注意该参数表示的是从单个进程接收的数据个数，不是总数
    MPI_Datatype recv_datatype,
    int root,
    MPI_Comm communicator)
```
### MPI_Allgather 多对多通信
当数据分布在所有的进程中时，MPI_Allgather将所有的数据聚合到每个进程中。

![mpi_Allgather][3]




# reference

[MPI学习笔记之并行程序概述](http://blog.csdn.net/sinat_22336563/article/details/69486937)


  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1502761049076.jpg
  [2]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1502761558789.jpg
  [3]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1502761637900.jpg