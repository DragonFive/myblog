---
title: go-turn-learning-note 
date: 2017/6/15 17:38:58

categories:
- tracking
tags:
- computervision
- caffe
- tracking
- deeplearning
- 深度学习
---
[TOC]

goturn 的代码是用caffe写的，是我们学caffe，深度学习和目标跟踪的好的学习资料。

[gotrun主页](http://davheld.github.io/GOTURN/GOTURN.html)

[github主页](https://github.com/davheld/GOTURN)

<!--more-->

## 编译问题

caffe_dir找不到

修改cmakefile.txt


vot_test
修改cmakefile.txt


## 下载

## 阅读源码

在docker容器里面运行脚本，但是不能显示结果。

在根目录（GOTURN）运行
```bash
bash scripts/show_tracker_test.sh VOT2014
```
可以用训练好的tracker测试vot2014，看看代码是怎么弄的，打开脚本show_tracker_test.sh，最后一句是
>build/show_tracker_vot  something

说明是调用的goturn里面这个已经编译好的这个工具show_tracker_vot，那个这个工具的源码在哪里呢。让我们来看看cmakefile吧。

执行
```bash
grep -nr show_tracker_vot CMakeLists.txt
```
输出的是：
> 99:add_executable (show_tracker_vot src/visualizer/show_tracker_vot.cpp)
101:target_link_libraries (show_tracker_vot ${PROJECT_NAME})

说明这个工具由 src/visualizer/show_tracker_vot.cpp编译得到，那么我们就打开这个文件学习一下源码。下面是代码的主要部分：


```cpp
  // Set up the neural network.
  const bool do_train = false;
  Regressor regressor(model_file, trained_file, gpu_id, do_train);

  Tracker tracker(show_intermediate_output);

  // Get videos.
  LoaderVOT loader(videos_folder);
  std::vector<Video> videos = loader.get_videos();

  // Visualize the tracker performance.
  TrackerVisualizer tracker_visualizer(videos, &regressor, &tracker);
  tracker_visualizer.TrackAll(start_video_num, pause_val);

```
这里有几个重要的类：Regressor/Tracker/TrackerVisualizer，待会我们一一来看，do_train和show_intermediate_output都是false.





```
grep -nr TrackerVisualizer
tracker/tracker_manager.h:54:class TrackerVisualizer : public TrackerManager
tracker/tracker_manager.h:57:  TrackerVisualizer(const std::vector<Video>& videos,
tracker/tracker_manager.cpp:71:TrackerVisualizer::TrackerVisualizer(const std::vector<Video>& videos,
tracker/tracker_manager.cpp:78:void TrackerVisualizer::ProcessTrackOutput(
tracker/tracker_manager.cpp:101:void TrackerVisualizer::VideoInit(const Video& video, const size_t video_num) {
visualizer/show_tracker_alov.cpp:63:  TrackerVisualizer tracker_visualizer(videos, &regressor, &tracker);
visualizer/show_tracker_vot.cpp:60:  TrackerVisualizer tracker_visualizer(videos, &regressor, &tracker);
```





