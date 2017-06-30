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





# reference