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
# 方案配置

但是cadvisor只能搜集本node上面的资源信息，对于集群中其它结点的资源使用情况检查不了。而heapter是一个只需运行一份就能监控集群中所有node的资源信息，所以我们使用主流的方案：heapter+ubfluxdb+grafaba. heapter用来采集信息，ubfluxdb用来存储，而grafaba用来展示信息。

## 使用influxdb
InfluxDB的不同版本，安装都是通过rpm直接安装，区别只是数据库的“表”不一样而已，所以会影响到Grafana过滤数据，这些不是重点，重点是Grafana数据的清理。

首先下载安装influxdb。在[influxdb](https://repos.influxdata.com)里找到适合的版本下载安装。


```bash
wget https://repos.influxdata.com/centos/7/x86_64/stable/influxdb-1.2.4.x86_64.rpm
rpm -vih  influxdb-1.2.4.x86_64.rpm
```
安装之后发现influxdb需要8086和8088两个端口，但这两个端口经常被占用，所以我们打算使用容器来运行

发现influxdb1.0的才能被grafana使用

使用的镜像是[tutum/influxdb:0.8.8](https://hub.docker.com/r/tutum/influxdb/), 
步骤和参数 [Docker学习系列3-Influxdb使用入门](http://blog.csdn.net/u011537073/article/details/52852759)

Connected to http://localhost:8086 version 0.10.3
InfluxDB shell 0.10.3
> SHOW DATABASES
name: databases
---------------
name
_internal
 
新建testdb数据库
> CREATE DATABASE testdb
> SHOW DATABASES
name: databases
---------------
name
_internal
testdb
> use testdb
Using database testdb
 
新建root用户
> CREATE USER "root" WITH PASSWORD 'root' WITH ALL PRIVILEGES
> show users
user    admin
root    true
 
插入一条测试数据
> INSERT cpu,host=test,region=us_west value=0.64
> SELECT * FROM /.*/ LIMIT 1
name: cpu
---------
time                    host    region  value
1458115488163303455     test    us_west 0.64
 
>

[influxdb的简单操作](http://www.linuxdaxue.com/influxdb-basic-operation.html)
## 使用grafana
使用的grafana版本[grafana](https://hub.docker.com/r/tutum/grafana/)
采用的镜像来自[ist0ne的google container](https://hub.docker.com/u/ist0ne/?page=2)

[ist0ne的github](https://github.com/ist0ne/google-containers)

k get deployment --all-namespaces

有两种连接influxdb的方式：一种是proxy,通过influxdb的service名字+port方式
一种是direct,通过nodeip+nodeport方式，前者与influxdb具体所在的port无法，但是需要设置好



## heapster 

使用google_container遇到问题：
heapster的镜像无法启动;
>Back-off restarting failed container

参考资料[k8s的资源](https://github.com/kubernetes/kubernetes/issues/22684)，[source-config](https://github.com/kubernetes/heapster/blob/master/docs/source-configuration.md#current-sources)
查看log:
```bash
kubectl logs -p --namespace=kube-system  heapster-1014378573-6s75z
```

>Failed to create source provide: open /var/run/secrets/kubernetes.io/serviceaccount/token: no such file or directory

解决方案：
```yaml
--source=kubernetes:http://10.10.31.25:8080?inClusterConfig=false 
```

--sink的设置：
https://github.com/kubernetes/heapster/blob/master/docs/sink-configuration.md


接下来发现问题：
>Failed to create infuxdb: failed to ping InfluxDB server at "monitoring-influxdb.kube-system.svc:8086" - Get http://monitoring-influxdb.kube-system.svc:8086/ping: dial tcp: lookup monitoring-influxdb.kube-system.svc on 10.10.11.110:53: no such host

解决方法：
>--sink=influxdb:http://10.10.31.26:30031

# 其它信息记录




# reference
[Kubernetes监控之Heapster介绍](http://dockone.io/article/1881)

[Kubernetes技术分析之监控](http://dockone.io/article/569)

[部署分布式(cadvisor+influxdb+grafana)](http://www.pangxie.space/docker/580)

[Try InfluxDB and Grafana by docker](https://blog.laputa.io/try-influxdb-and-grafana-by-docker-6b4d50c6a446)

[Run Heapster in a Kubernetes cluster with an InfluxDB backend and a Grafana UI](https://github.com/kubernetes/heapster/blob/master/docs/influxdb.md)


