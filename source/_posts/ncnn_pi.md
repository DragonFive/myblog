---
title: ncnn_pi
date: 2017/8/24 17:38:58

categories:
- 深度学习
tags:
- deeplearning
- tensorflow
- 嵌入式
- 模型压缩
---






<!--more-->


# 安装ncnn

```cpp
git clone https://github.com/Tencent/ncnn
sudo apt-get install libprotobuf-dev protobuf-compiler
cd ncnn
mkdir build && cd build
cmake ..
make -j
make install
```








# reference

[在树莓派上用TensorFlow玩深度学习](https://www.codelast.com/%E5%8E%9F%E5%88%9B-%E5%9C%A8%E6%A0%91%E8%8E%93%E6%B4%BE%E4%B8%8A%E7%94%A8tensorflow%E7%8E%A9%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0deep-learning/)

[腾讯NCNN框架入门到应用](http://blog.csdn.net/best_coder/article/details/76201275)

[ncnn wiki](https://github.com/Tencent/ncnn/wiki)