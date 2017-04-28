---
title: ubuntu安装Nvidia驱动-cudnn-anaconda-tensorflow-tensorlayer

date: 2017/4/27 15:04:12

categories:
- 计算机视觉
tags:
- nvidia驱动
- cuda
- anaconda
- tensorflow
- tensorlayer
- 深度学习
- deeplearning
---
[TOC]

# 安装Nvidia驱动

## 查询NVIDIA驱动
首先去官网(http://www.nvidia.com/Download/index.aspx?lang=en-us) 查看适合自己显卡的驱动
## 安装NVIDIA驱动
安装之前先卸载已经存在的驱动版本：
```
sudo apt-get remove --purge nvidia*
```
若电脑是集成显卡（NVIDIA独立显卡忽略此步骤），需要在安装之前禁止一项：
```
sudo service lightdm stop
```
执行以下指令安装驱动：
```
sudo add-apt-repository ppa:xorg-edgers/ppa
sudo apt-get update
sudo apt-get install nvidia-375 #注意在这里指定自己的驱动版本！
```
安装完成之后输入以下指令进行验证：
```
sudo nvidia-smi
```
若列出了GPU的信息列表则表示驱动安装成功。如果没看到，重启再试一下,linux装驱动需要重启才加载吧

## 可能出的问题

### add-apt-repository 命令不存在
```
sudo apt-get update
sudo apt-get install python-software-properties
sudo apt-get install software-properties-common
```
然后关掉terminator

### 输入nvidia-smi 说驱动没装上

重装系统换成英文版ubuntu




