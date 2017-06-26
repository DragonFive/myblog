---
title: 不会装cuda配环境的小学生怎么躺撸caffe
date: 2017/5/26 15:04:12

categories:
- 计算机视觉
tags:
- nvidia驱动
- 深度学习
- caffe
- docker
- k8s
- deeplearning
- python
---
本文首发于个人博客 [不会装cuda配环境的小学生怎么躺撸caffe](https://dragonfive.github.io/docker_caffe_gpu_ipython/)
收录于简书专题[[深度学习·计算机视觉与机器学习](http://www.jianshu.com/c/1249336e61cb)](http://www.jianshu.com/c/1249336e61cb)

DL如今已经快成为全民玄学了，感觉离民科入侵不远了。唯一的门槛可能是环境不好配，特别是caffe这种依赖数10种其它软件打框架。不过有了docker和k8s之后，小学生也能站撸DL了。

![enter description here][1]



<!--more-->
# 从nvidia-docker到docker，需要有这样的操作
大致流程如下，入门版通过docker pull一个GPU版本的caffe 的image,然后安装nvidia-docker 和 nvidia-docker-plugin 来映射宿主机的nvidia-driver并通过共享volume的方式来支持容器里面能“看到”宿主机的GPU。进阶版通过curl -s命令列出宿主机的配置显卡配置，并通过docker run的方式来启动。总结完成。纸上得来终觉浅，绝知此事要躬行，光说不练空把式，唯有实践出真知。
[tensorflow gpu in docker](https://xuxinkun.github.io/2016/10/08/tensorflow-kubernetes/)

IntelMPI：适用于单机多卡，Ethernet或者InfiniBand网络。IntelMPI比Mvapich速度更快，对GPU更友好，没有Mvapich中常遇到的CudaMemCpyAsync错误

使用nvidia-docker
```
sudo nohup nvidia-docker-plugin >/tmp/nvidia-docker.log &  
然后nvidia-docker run
```
使用docker来代替 nvidia-docker
```
curl -s http://localhost:3476/docker/cli
```
下面应该是输出：
```
--device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia7 --devic/dev/nvidia6 --device=/dev/nvidia5 --device=/dev/nvidia4 --device=/dev/nvidia3 --device=/dev/nvidia2 --device=/dev/nvidia1 --device=/dev/nvidia0 --volume-driver=nvidia-docker --volume=nvidia_driver_375.39:/usr/local/nvidia:ro
```
这样其实
```
sudo docker run -ti `curl -s http://localhost:3476/v1.0/docker/cli` -v /mnt/share:/mnt/share -v /mnt/lustre:/mnt/lustre  -v /lib64:/lib64 镜像名 bash
```
所以如果你想用docker的方式来运行GPU版本 那么你就需要指明你的所有的device信息，如果卸载rc文件里，那么只能这样
```
sudo docker run -ti --device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia7 --device=/dev/nvidia6 --device=/dev/nvidia5 \
--device=/dev/nvidia4 --device=/dev/nvidia3 --device=/dev/nvidia2 --device=/dev/nvidia1 --device=/dev/nvidia0 \
--volume-driver=nvidia-docker --volume=nvidia_driver_375.39:/usr/local/nvidia:ro \
 -v /mnt/share:/mnt/share -v /mnt/lustre:/mnt/lustre  -v /lib64:/lib64  镜像名  bash
```
当前你也可以有这样风骚的走位
```
sudo docker run -ti  $(ls /dev/nvidia* | xargs -I{} echo '--device={}')   -v /mnt/share:/mnt/share -v /mnt/lustre:/mnt/lustre  -v /lib64:/lib64  镜像名  bash
```

# 在镜像里安装ipython notebook，需要这样做
把大象装进冰箱分四步，映射端口，开通open-ssh服务器，安装jupyter,配置密码
在镜像中执行
0. 映射端口
在dock run的时候加-p参数 
1. 开通ssh
```
sudo apt-get install openssh-server
```
2. 安装jupyter
```
apt-get update
#安装python dev包
apt-get install python-dev
#安装jupyter
pip install jupyter
```
3. 设置密码
分三小步
a. 生成jupyter配置文件，这个会生成配置文件.jupyter/jupyter_notebook_config.py
```
jupyter notebook --generate-config
```
b. 从密码到ssa密文
在命令行输入ipython，进入ipython命令行
```
#使用ipython生成密码
In [1]: from notebook.auth import passwd
In [2]: passwd()
Enter password: 
Verify password: 
Out[2]: 'sha1:38a5ecdf288b:c82dace8d3c7a212ec0bd49bbb99c9af3bae076e'
````
c. 改配置
```
#去配置文件.jupyter/jupyter_notebook_config.py中修改以下参数
c.NotebookApp.ip='*'                          #绑定所有地址
c.NotebookApp.password = u'刚才生成的密文也就是sha1:38a5ecdf288b:c82dace8d3c7a212ec0bd49bbb99c9af3bae076e'
c.NotebookApp.open_browser = False            #启动后是否在浏览器中自动打开，注意F大写
c.NotebookApp.port =8888                      #指定一个访问端口，默认8888，注意和映射的docker端口对应
```

然后执行ipython notebook --allow-root就可以在宿主机上用docker里面的环境了，爽歪歪。

[把jupyter-notebook装进docker里](https://segmentfault.com/a/1190000007448177)

# 使用k8s与脚本一键式训练网络
## 简要介绍

用k8s启动docker能够有效的管理宿主机资源，保证任务能够在资源许可的情况下顺利地执行，同时能够保护宿主机的安全。但是从使用k8s到训练网络这中间隔着几十步的操作、配置和交互，需要遵循相应的顺序和格式，比较繁琐。这里通过一种expect脚本的方式简化这种操作，让用户可以简洁而又正确的使用k8s.

## 操作步骤

1. 根据需要的资源修改yaml文件
2. 修改执行脚本里的资源和网络文件位置
3. 执行expect脚本
## 修改yaml文件

这里举个例子并有一些注意事项。

下面的yaml文件使用RC的方式创建pod
 
 ```yaml
 apiVersion: v1
kind: ReplicationController
metadata:
  name: xxx-master2
  labels:
    name: xxx-master2
spec:
  replicas: 2
  selector:
    name: xxx-master2
  template:
    metadata:
      labels:
        name: xxx-master2
    spec:
      nodeSelector:
        ip: five
      containers:
      - name: xxx-master
        image: 10.10.31.26:5000/xxx_cuda8:2.1
        #command: ["/bin/sleep", "infinity"]
        #securityContext:
        #  privileged: true
        ports:
        - containerPort: 6380
        imagePullPolicy: IfNotPresent
        resources:
          requests:
            alpha.kubernetes.io/nvidia-gpu: 1
            #cpu: 2
            #memory: 30Gi
          limits:
            alpha.kubernetes.io/nvidia-gpu: 1
            #cpu: 2
            #memory: 30Gi
        volumeMounts:
        - mountPath: /usr/local/nvidia/
          name: nvidia-driver
          readOnly: true
        - mountPath: /mnt/lustre/xxx/xxx/
          name: sensenet
        - mountPath: /mnt/lustre/share/
          name: share
      volumes:
      - hostPath:
          path: /var/lib/nvidia-docker/volumes/nvidia_driver/375.39
        name: nvidia-driver
      - hostPath:
          path: /mnt/lustre/xxx/xxx
        name: xxx
      - hostPath:
          path: /mnt/lustre/share/
        name: share
 ```
需要注意的是：

```yaml
  nodeSelector:
      ip: five
```
表示选择标签为ip=five的结点，这句话也可以不要。注意后面的resource的gpu数不要超过物理机GPU总数。


## 修改expect脚本

注意这个expect 脚本运行前需保证各个pod都处于running 状态。

xxx-pod-cfg.exp脚本内容

```yaml
#!/usr/bin/expect -f

# 设置超时时间

set timeout 30000
# 设置GPU个数
set gpuNum 1
# 创建rc创建pod
exec kubectl create -f /mnt/lustre/xxx/xxx/yaml/xxxx_cuda_controller_test.yaml
sleep 10

# 首先通过k8s获得每个pod的ip与hostname对 放在一个临时文件中
exec kubectl get po -o=custom-columns=IP:status.podIP,NAME:.metadata.name >hehe

# 接下来把每个ip对 放在数组里面去
set fd [open "hehe" r]
gets $fd line
set numIp 0
while { [gets $fd line] >= 0 } {
        set ips($numIp) [ lindex $line 0 ]
        set hns($numIp) [ lindex $line 1 ]
        incr numIp
        #puts $numIp
}
#puts $ips(1)

# 接下来登录每个pod上面去修改hosts文件
for {set i 0} {$i<$numIp} {incr i} {
        set sshIp $ips($i)
        set sshUrl "root@"
        append sshUrl $sshIp
        # 连接上这个pod
        spawn ssh $sshUrl
        # 修改这个pod的文件
        expect "password:"
        send "12345678\r"
        # 下面把ip数组复制进文件里面
        for { set j 0} {$j<$numIp} {incr j} {
                set ip $ips($j)
                set hn $hns($j)
                append ip " " $hn
                expect "#*"
                send "echo $ip >> /etc/hosts\r"
        }
        # 如果有必要的话，可以在这里设置mpirun的位置
        expect "#*"
        send "exit\r"
        expect eof
}

# 下面生成第一个pod的key，并且copy到其它的pod里面.
set sshIp $ips(0)
set sshUrl "root@"
append sshUrl $sshIp
spawn ssh $sshUrl
expect "password:"
send "12345678\r"

expect "#*"
send "ssh-keygen \r"
expect "id_rsa):*"
send "\r"
expect "passphrase):*"
send "\r"
expect "again:*"
send "\r"

# 接下来保证第一个pod能ssh连上其它所有的pod
for {set i 1} {$i<$numIp} {incr i} {
        set cmd "ssh-copy-id -i ~/.ssh/id_rsa.pub "
        set ip $ips($i)
        append cmd $ip
        expect "#*"
        send "$cmd \r"
        expect "yes/no)?*"
        send "yes\r"
        expect "password:*"
        send "12345678\r"
}


# 下面制作hostfile文件,把ip数组写进文件里
expect "#*"
send "cd /root\r"
for {set i 0} {$i<$numIp} {incr i} {
        #set content $ips($i)
        expect "#*"
        send "echo $ips($i) >>hostfile\r"
}

# 下面开始训练resnet200
expect "#*"
send "/mnt/lustre/share/intel64/bin/mpirun -n $numIp -ppn $gpuNum -f hostfile -env I_MPI_FABRICS shm:tcp /mnt/lustre/xxx/xxx/example/build/tools/caffe train --solver=/mnt/lustre/xxx/xxx/example/resnet200/resnet200_solver.prototxt\r"


expect eof
exit
```

1. 下面这句表示超时时间，设置大一点比较好，不然可能提前结束
```yaml
  set timeout 30000 
```
2. 下面这句是每个pod里面的GPU数量，根据实际情况自己设置
```yaml
  set gpuNum 1
```
3. 创建rc来创建pod ,sleep10保证在执行下一句之前pod能处于running 状态，根据需要时间可以调长 
```yaml
exec kubectl create -f /mnt/lustre/xxx/xxxx/yaml/xxx_cuda_controller_test.yaml 
sleep 10
```
4. 下面是获取ip与hostname对
```
  exec kubectl get po -l name==sensenet-master2  -o=custom-columns=IP:status.podIP,NAME:.metadata.name >hehe
```
-l后面跟的是你要获取的pod的过滤器，也就是label的值，这里我是用上面rc创建的两个pod,自动给pod打标签为name=xxx-master2,所以这里这样写。

5. 训练网络的例子，根据自己需要进行修改
```
  send "/mnt/lustre/share/intel64/bin/mpirun -n $numIp -ppn $gpuNum -f hostfile -env I_MPI_FABRICS shm:tcp /mnt/lustre/xxx/xxx/example/build/tools/caffe train --solver=/mnt/lustre/xxx/xxxx/example/resnet200/resnet200_solver.prototxt\r"
```
## 执行expect脚本
```
   expect sensenet-pod-cfg.exp
```

# 自己制作的某个支持cuda的dockerfile
```
FROM 10.10.31.26:5000/nvidia/cuda:8.0-cudnn5-runtime-centos7
# 作者
MAINTAINER xxx "xxx.com"
# 先安装一批需要的软件
COPY local_base.repo  /etc/yum.repos.d/local_base.repo
COPY requirements.txt /root/requirements.txt
COPY sshd_config /etc/ssh/sshd_config

RUN yum clean all -y && yum clean metadata -y \
        && yum clean dbcache -y && yum makecache -y \
        && yum update -y \
        && yum install -y  \
        boost boost-devel \
        glog glog-devel \
        protobuf protobuf-devel protobuf-python \
        hdf5-devel hdf5 \
        openssh-server \
        lmdb-devel lmdb \
        leveldb leveldb-devel \
        opencv opencv-devel opencv-python \
        openblas-devel openblas \
        && echo 'root:12345678' | chpasswd \
        && yum clean all \
        && ssh-keygen -t dsa -f /etc/ssh/ssh_host_dsa_key \
        && ssh-keygen -t rsa -f /etc/ssh/ssh_host_rsa_key \
        && ssh-keygen -t ecdsa -f /etc/ssh/ssh_host_ecdsa_key \
        && ssh-keygen -t ed25519 -f /etc/ssh/ssh_host_ed25519_key \
        && mkdir /var/run/sshd

```

# 参考资料

[把jupyter-notebook装进docker里](https://segmentfault.com/a/1190000007448177)
[tensorflow gpu in docker](https://xuxinkun.github.io/2016/10/08/tensorflow-kubernetes/)


  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1496650170493.jpg
