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

wget https://repos.influxdata.com/centos/7/x86_64/stable/influxdb-1.2.4.x86_64.rpm


# reference
[Kubernetes监控之Heapster介绍](http://dockone.io/article/1881)

[Kubernetes技术分析之监控](http://dockone.io/article/569)

[部署分布式(cadvisor+influxdb+grafana)](http://www.pangxie.space/docker/580)