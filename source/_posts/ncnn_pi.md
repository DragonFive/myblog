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

Here's my solution utilizing the V2 checkpoints introduced in TF 0.12.

There's no need to convert all variables to constants or freeze the graph.

Just for clarity, a V2 checkpoint looks like this in my directory models:
```
checkpoint  # some information on the name of the files in the checkpoint
my-model.data-00000-of-00001  # the saved weights
my-model.index  # probably definition of data layout in the previous file
my-model.meta  # protobuf of the graph (nodes and topology info)
```
**Python part (saving)**
```
with tf.Session() as sess:
    tf.train.Saver(tf.trainable_variables()).save(sess, 'models/my-model')
```
If you create the Saver with tf.trainable_variables(), you can save yourself some headache and storage space. But maybe some more complicated models need all data to be saved, then remove this argument to Saver, just make sure you're creating the Saver after your graph is created. It is also very wise to give all variables/layers unique names, otherwise you can run in different problems.

**Python part (inference)**
```
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('models/my-model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('models/'))
    outputTensors = sess.run(outputOps, feed_dict=feedDict)
```
## 导出pb文件 

```
python export_inference_graph.py \
  --alsologtostderr \
  --model_name=mobilenet_v1 \
  --image_size=224 \
  --output_file=/tmp/mobilenet_v1_224.pb
```

[Exporting the Inference Graph](https://github.com/tensorflow/models/tree/master/slim#fine-tuning-a-model-from-an-existing-checkpoint)

[作者的pretrained model](https://pan.baidu.com/s/1i5xFjal)

## frozen pb文件

```bash
python tensorflow/tensorflow/python/tools/freeze_graph.py \ 
--input_graph=models/slim/mobilenet_v1_224.pb \
--input_checkpoint=tmp_data/mobilenet_v1_1.0_224.ckpt \
--input_binary=true --output_graph=tmp_data/frozen_mobilenet.pb --output_node_names=mobilenetv1/Predictions/Reshape_1
```
这里有个点在于怎么确定一个网络的output_node_names,参考

[这个人自己搞的mobilenet](https://github.com/Zehaos/MobileNet/issues/4)

## image_label




下载label数据 

```
curl -L "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz" |
  tar -C tensorflow/examples/label_image/data -xz
```

```
python tensorflow/tensorflow/examples/label_image/label_image.py --graph=tmp_data/inception_v3_2016_08_28_frozen.pb --image=cat123.jpg --input_layer=input --output_layer=InceptionV3/Predictions/Reshape_1 --input_mean=128 --input_std=128  --labels=tmp_data/imagenet_slim_labels.txt
```


# reference

[在树莓派上用TensorFlow玩深度学习](https://www.codelast.com/%E5%8E%9F%E5%88%9B-%E5%9C%A8%E6%A0%91%E8%8E%93%E6%B4%BE%E4%B8%8A%E7%94%A8tensorflow%E7%8E%A9%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0deep-learning/)

[腾讯NCNN框架入门到应用](http://blog.csdn.net/best_coder/article/details/76201275)

[ncnn wiki](https://github.com/Tencent/ncnn/wiki)

[tensorflow模型的各种版本](https://stackoverflow.com/questions/44516609/tensorflow-what-is-the-relationship-between-ckpt-file-and-ckpt-meta-and-ckp)

[tensorfow各种版本的模型](https://stackoverflow.com/questions/35508866/tensorflow-different-ways-to-export-and-run-graph-in-c/43639305#43639305)


[mobilenet的使用](https://hackernoon.com/building-an-insanely-fast-image-classifier-on-android-with-mobilenets-in-tensorflow-dc3e0c4410d4)

[retrain_mobilenet](https://hackernoon.com/creating-insanely-fast-image-classifiers-with-mobilenet-in-tensorflow-f030ce0a2991)
