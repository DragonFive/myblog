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

 
目前的深度学习领域就是海量的数据加上大量的数学运算，所以计算量相当的大，训练一个模型跑上十天半个月啥的是常事。那此时分布式的意义就出现了，既然一张GPU卡跑得太慢就来两张，一台机器跑得太慢就用多台机器。


**数据并行**


![数据并行][1]

<!--more-->
每一个节点（或者叫进程）都有一份模型，然后各个节点取不同的数据，通常是一个batch_size，然后各自完成前向和后向的计算得到梯度，这些进行训练的进程我们成为**worker**，除了worker，还有**参数服务器**，简称ps server，这些worker会把各自计算得到的梯度送到ps server，然后由ps server来进行update操作，然后把update后的模型再传回各个节点。因为在这种并行模式中，被划分的是数据，所以这种并行方式叫**数据并行**。

数据并行有**同步模式和异步模式**之分。同步模式中，所有训练程序同时训练一个批次的训练数据，完成后经过同步，再同时交换参数。参数交换完成后所有的训练程序就有了共同的新模型作为起点，再训练下一个批次。而异步模式中，训练程序完成一个批次的训练数据，立即和参数服务器交换参数，不考虑其他训练程序的状态。异步模式中一个训练程序的最新结果不会立刻体现在其他训练程序中，直到他们进行下次参数交换。

[ 卷积神经网络的并行化模型](http://blog.csdn.net/xsc_c/article/details/42420167)

# parameter server

limu的parameter server， MSRA的adam和google的tensorflow。

[最近比较火的parameter server是什么？](https://www.zhihu.com/question/26998075)

[李沐：Parameter Server for Distributed Machine Learning](http://www.cs.cmu.edu/~muli/file/ps.pdf)

参数服务器是个编程框架，用于方便分布式并行程序的编写，其中重点是对大规模参数的分布式存储和协同的支持。

参数服务器就类似于MapReduce，是大规模机器学习在不断使用过程中，抽象出来的框架之一。重点支持的就是**参数的分布式**，毕竟巨大的模型其实就是巨大的参数。
## 架构：
集群中的节点可以分为**计算节点和参数服务节点**两种。其中，计算节点负责对分配到自己本地的训练数据（块）计算学习，并更新对应的参数；参数服务节点采用分布式存储的方式，各自存储全局参数的一部分，并作为服务方接受计算节点的参数查询和更新请求。简而言之吧，计算节点负责干活和更新参数，参数服务节点则负责存储参数。
## 冗余和恢复：
类似MapReduce，每个参数在参数服务器的集群中都在多个不同节点上备份（3个也是极好的），这样当出现节点失效时，冗余的参数依旧能够保证服务的有效性。当有新的节点插入时，把原先失效节点的参数从冗余参数那边复制过来，失效节点的接班人就加入队伍了。
## 并行计算：
并行计算这部分主要在计算节点上进行。 类似于MapReduce，分配任务时，会将数据拆分给每个worker节点。参数服务器在开始学习前，也会把大规模的训练数据拆分到每个计算节点上。单个计算节点就对本地数据进行学习就可以了。学习完毕再把参数的更新梯度上传给对应的参数服务节点进行更新。


## 流程
1.分发训练数据 -> 节点1 节点2   节点3   ... 节点i  ... 节点N
2.节点i 学习过程：遍历本地的训练数据，统计所有需要的参数(key)向分布式的参数服务器查询需要的参数（注意，本地数据对应到的参数只是全局参数的一小部分）得到查询到的参数值，用于模型的本地训练一轮训练完毕，得到所有对应参数的更新，将更新上传给参数服务器
3.参数服务器更新参数过程：参数服务器得到计算节点传过来的局部更新，**汇总后更新本地数据**






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

![scatter与bcast的区别][2]


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

![mpi_gather][3]


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

![mpi_Allgather][4]

## 数据归约 Reduce
Reduce——规约是来自函数式编程的一个经典概念。数据规约包含通过一个函数将一批数据分成较小的一批数据。比如将一个数组的元素通过加法函数规约为一个数字。

### mpi_reduce
与MPI_Gather类似，MPI_Reduce在每个进程上都有一组输入元素，并将**一个输出元素数组返回给根进程**。 输出元素包含被规约的结果。

```cpp
MPI_Reduce(
    void* send_data,
    void* recv_data,
    int count,
    MPI_Datatype datatype,
    MPI_Op op,
    int root,
    MPI_Comm communicator)
```
>send_data参数指向的是每个进程想要规约的datatype类型的元素数组。
>recv_data仅与根进程相关。
>recv_data数组包含规约的结果，并具有sizeof（datatype）* count的大小的内存。
>op参数是要应用于数据的操作。

mpi支持的操作有 


>MPI_MAX - 返回最大值.
MPI_MIN - 返回最小值.
MPI_SUM -元素和.
MPI_PROD - 元素乘积.
MPI_LAND - 逻辑与.
MPI_LOR - 逻辑或
MPI_BAND -按位与
MPI_BOR - 按位或
MPI_MAXLOC - 返回最大值和拥有该值的进程编号
MPI_MINLOC - 返回最小值和拥有该值的进程编号.```


如果每个进程中的数组拥有两个元素，那么规约结果是对两个对位的元素进行规约的。

![两个元素的归约结果][5]

### mpi_allReduce

![归约后分发给所有的进程][6]


# parameter-server



# reference

[MPI学习笔记之并行程序概述](http://blog.csdn.net/sinat_22336563/article/details/69486937)


[ 卷积神经网络的并行化模型](http://blog.csdn.net/xsc_c/article/details/42420167)

[知乎 parameter server](https://www.zhihu.com/search?type=content&q=parameter+server)

[分布式机器学习系统笔记（一）——模型并行，数据并行，参数平均，ASGD](http://blog.csdn.net/xbinworld/article/details/74781605)

[深度学习及并行化实现概述](http://djt.qq.com/article/view/1245)

  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1505026360037.jpg
  [2]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1502761049076.jpg
  [3]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1502761558789.jpg
  [4]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1502761637900.jpg
  [5]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1502762764619.jpg
  [6]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1502762804609.jpg