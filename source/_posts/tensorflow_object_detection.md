---
title: tensorflow_object_detection
date: 2017/8/23 17:38:58

categories:
- 深度学习
tags:
- deeplearning
- 目标检测
- tensorflow
- faster-rcnn
---




<!--more-->
# 环境配置

## docker  tensorflow-gpu


## protobuf


## slim-models


编译proto的内容

```bash
protoc object_detection/protos/*.prot --python_out=.
```
设置 PYTHONPATH 环境变量

```cpp
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
注意这里pwd外围的不是单引号，而是esc下面那个键


## 宠物数据集和标注

### 数据下载

下载训练数据里面的 图片
```
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
```
下载训练数据里面的标注信息
```
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
```
### 将训练数据格式转化为tfrecord格式

需要一个id与类名的对应关系的txt文件：pet_label_map.pbtxt

```cpp
item {
  id: 1
  name: 'Abyssinian'
}

item {
  id: 2
  name: 'american_bulldog'
}

item {
  id: 3
  name: 'american_pit_bull_terrier'
}

item {
  id: 4
  name: 'basset_hound'
}

```
**执行脚本**
需要一个把数据转化成 tfrecoder格式的脚本文件: 

```cpp

r"""Convert the Oxford pet dataset to TFRecord for object_detection.

See: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
     Cats and Dogs
     IEEE Conference on Computer Vision and Pattern Recognition, 2012
     http://www.robots.ox.ac.uk/~vgg/data/pets/

Example usage:
    ./create_pet_tf_record --data_dir=/home/user/pet \
        --output_dir=/home/user/pet/output
"""

import hashlib
import io
import logging
import os
import random
import re

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'data/pet_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS


def get_class_name_from_filename(file_name):
  """Gets the class name from a file.

  Args:
    file_name: The file name to get the class name from.
               ie. "american_pit_bull_terrier_105.jpg"

  Returns:
    example: The converted tf.Example.
  """
  match = re.match(r'([A-Za-z_]+)(_[0-9]+\.jpg)', file_name, re.I)
  return match.groups()[0]


def dict_to_tf_example(data,
                       label_map_dict,
                       image_subdirectory,
                       ignore_difficult_instances=False):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
      Pascal dataset directory holding the actual image data.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  img_path = os.path.join(image_subdirectory, data['filename'])
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  for obj in data['object']:
    difficult = bool(int(obj['difficult']))
    if ignore_difficult_instances and difficult:
      continue

    difficult_obj.append(int(difficult))

    xmin.append(float(obj['bndbox']['xmin']) / width)
    ymin.append(float(obj['bndbox']['ymin']) / height)
    xmax.append(float(obj['bndbox']['xmax']) / width)
    ymax.append(float(obj['bndbox']['ymax']) / height)
    class_name = get_class_name_from_filename(data['filename'])
    classes_text.append(class_name.encode('utf8'))
    classes.append(label_map_dict[class_name])
    truncated.append(int(obj['truncated']))
    poses.append(obj['pose'].encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example


def create_tf_record(output_filename,
                     label_map_dict,
                     annotations_dir,
                     image_dir,
                     examples):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    label_map_dict: The label map dictionary.
    annotations_dir: Directory where annotation files are stored.
    image_dir: Directory where image files are stored.
    examples: Examples to parse and save to tf record.
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  for idx, example in enumerate(examples):
    if idx % 100 == 0:
      logging.info('On image %d of %d', idx, len(examples))
    path = os.path.join(annotations_dir, 'xmls', example + '.xml')

    if not os.path.exists(path):
      logging.warning('Could not find %s, ignoring example.', path)
      continue
    with tf.gfile.GFile(path, 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

    tf_example = dict_to_tf_example(data, label_map_dict, image_dir)
    writer.write(tf_example.SerializeToString())

  writer.close()


# TODO: Add test for pet/PASCAL main files.
def main(_):
  data_dir = FLAGS.data_dir
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  logging.info('Reading from Pet dataset.')
  image_dir = os.path.join(data_dir, 'images')
  annotations_dir = os.path.join(data_dir, 'annotations')
  examples_path = os.path.join(annotations_dir, 'trainval.txt')
  examples_list = dataset_util.read_examples_list(examples_path)

  # Test images are not included in the downloaded data set, so we shall perform
  # our own split.
  random.seed(42)
  random.shuffle(examples_list)
  num_examples = len(examples_list)
  num_train = int(0.7 * num_examples)
  train_examples = examples_list[:num_train]
  val_examples = examples_list[num_train:]
  logging.info('%d training and %d validation examples.',
               len(train_examples), len(val_examples))

  train_output_path = os.path.join(FLAGS.output_dir, 'pet_train.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'pet_val.record')
  create_tf_record(train_output_path, label_map_dict, annotations_dir,
                   image_dir, train_examples)
  create_tf_record(val_output_path, label_map_dict, annotations_dir,
                   image_dir, val_examples)

if __name__ == '__main__':
  tf.app.run()
```

**脚本参数**
这个脚本的使用方法是：
```cpp
    python  object_detection/create_pet_tf_record.py  --data_dir=`pwd`\
        --label_map_path=object_detection/data/pet_lable_map.pbtxt \
       --output_dir=`pwd`
```
需要指定data_dir :图像的源文件夹，output_dir：转化成tfrecord格式

--label_map_path：id与class_name的对应表。

**脚本流程**
1. 读取anotations文件夹里面的 trainval.txt ，来确定用来训练的图片的文件名，然后通过shuffle,按比例分成训练集和验证集
2. 然后根据训练集或测试集的文件名来读取anotations文件夹里面xmls下面的xml文件，获得图片对象
3. 把图片对象写入文件

**生成文件**

pet_train.record

pet_val.record

这时候把它们拷贝到新创建的文件夹,比如 ```_pet_20170822```

将 object_detection/data/pet_label_map.pbtxt 和  object_detection/samples/configs/faster_rcnn_resnet152_pets.config 两个文件也拷贝到这个目录。

```
  #fine_tune_checkpoint: "_pet_20170822/model.ckpt"
  from_detection_checkpoint: false
```
# reference

[protobuf的配置](http://zhwen.org/?p=909)

[tf-slim的用法,用于图像检测和分割](http://geek.csdn.net/news/detail/126133)

[tf-slim官方教程](https://github.com/tensorflow/models/blob/master/slim/slim_walkthrough.ipynb)

[TensorFlow Object Detection API 实践](https://mp.weixin.qq.com/s?__biz=MzI2MzYwNzUyNg==&mid=2247484024&idx=1&sn=a7ddf704f34d390bd2a64f7651ea4a44&chksm=eab807f1ddcf8ee78c6b28bea6ec7233b7236a9d653dea45859e36ea2c74afe3b350d6f97967&mpshare=1&scene=1&srcid=0822n8yUoxY0I5ITT55mu827&pass_ticket=bpMLj9Sb52k%2FdkdRYGa0rUDxv9zPd7tmsAPy7XAQiBMfzqdlgf06HEq0G62d4Kyz#rd)


[](http://www.10tiao.com/html/511/201707/2652000542/2.html)

