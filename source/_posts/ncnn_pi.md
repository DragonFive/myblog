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
进入 ncnn/build/tools 目录下，如下所示，我们可以看到已经生成了 caffe2ncnn 可ncnn2mem这两个可执行文件，这两个可执行文件的作用是将caffe模型生成ncnn 模型，并且对模型进行加密。在ncnn/build/tools/tensorflow下面也有tensorflow2ncnn，可以把tensorflow模型转化乘ncnn模型


# tensorflow 的安装



# tensorflow的模型

1. the .ckpt file is the old version output of saver.save(sess), which is the equivalent of your .ckpt-data (see below)
2. the "checkpoint" file is only here to tell some TF functions which is the latest checkpoint file.
3. .ckpt-meta contains the metagraph, i.e. the structure of your computation graph, without the values of the variables (basically what you can see in tensorboard/graph).
4. .ckpt-data contains the values for all the variables, without the structure. To restore a model in python, you'll usually use the meta and data files with (but you can also use the .pb file):
```
saver = tf.train.import_meta_graph(path_to_ckpt_meta)
saver.restore(sess, path_to_ckpt_data)
```
5. I don't know exactly for .ckpt-index, I guess it's some kind of index needed internally to map the two previous files correctly. Anyway it's not really necessary usually, you can restore a model with only .ckpt-meta and .ckpt-data.
6. the .pb file can save your whole graph (meta + data). To load and use (but not train) a graph in c++ you'll usually use it, created with freeze_graph, which creates the .pb file from the meta and data. Be careful, (at least in previous TF versions and for some people) the py function provided by freeze_graph did not work properly, so you'd have to use the script version. Tensorflow also provides a tf.train.Saver.to_proto() method, but I don't know what it does exactly.





# reference

[在树莓派上用TensorFlow玩深度学习](https://www.codelast.com/%E5%8E%9F%E5%88%9B-%E5%9C%A8%E6%A0%91%E8%8E%93%E6%B4%BE%E4%B8%8A%E7%94%A8tensorflow%E7%8E%A9%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0deep-learning/)

[腾讯NCNN框架入门到应用](http://blog.csdn.net/best_coder/article/details/76201275)

[ncnn wiki](https://github.com/Tencent/ncnn/wiki)

[tensorflow模型的各种版本](https://stackoverflow.com/questions/44516609/tensorflow-what-is-the-relationship-between-ckpt-file-and-ckpt-meta-and-ckp)

[tensorfow各种版本的模型](https://stackoverflow.com/questions/35508866/tensorflow-different-ways-to-export-and-run-graph-in-c/43639305#43639305)

