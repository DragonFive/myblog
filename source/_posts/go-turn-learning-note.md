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

## 

在docker容器里面运行脚本，但是不能显示结果

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