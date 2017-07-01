---
title: heapter-ubfluxdb_grafaba
date: 2017/6/30 15:04:12

categories:
- 虚拟化
tags:
- docker
- k8s
- heapter
- 资源监控
---

在k8s集群中，默认提供的资源监控方式是 cadvisor+influxdb+grafana,K8S已经将cAdvisor功能集成到kubelet组件中。每个Node节点可以直接进行web访问。
cAdvisor web界面访问： http://< Node-IP >:4194

<!--more-->

但是cadvisor只能搜集本node上面的资源信息，对于集群中其它结点的资源使用情况检查不了。而heapter是一个只需运行一份就能监控集群中所有node的资源信息，所以我们使用主流的方案：heapter+ubfluxdb+grafaba. heapter用来采集信息，ubfluxdb用来存储，而grafaba用来展示信息。

# 安装配置influxdb
InfluxDB的不同版本，安装都是通过rpm直接安装，区别只是数据库的“表”不一样而已，所以会影响到Grafana过滤数据，这些不是重点，重点是Grafana数据的清理。

首先下载安装influxdb。在[influxdb](https://repos.influxdata.com)里找到适合的版本下载安装。

## 安装influxdb
```bash
wget https://repos.influxdata.com/centos/7/x86_64/stable/influxdb-1.2.4.x86_64.rpm
rpm -vih  influxdb-1.2.4.x86_64.rpm
```
安装之后发现influxdb需要8086和8088两个端口，但这两个端口经常被占用，所以我们打算使用容器来运行

使用的镜像是[tutum/influxdb:0.8.8](https://hub.docker.com/r/tutum/influxdb/)

# reference
[Kubernetes监控之Heapster介绍](http://dockone.io/article/1881)

[Kubernetes技术分析之监控](http://dockone.io/article/569)

[部署分布式(cadvisor+influxdb+grafana)](http://www.pangxie.space/docker/580)