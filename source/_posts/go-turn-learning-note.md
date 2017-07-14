---
title: 手把手教你看懂goturn的源码
date: 2017/6/15 17:38:58

categories:
- 深度学习
- 计算机视觉
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

![enter description here][1]

<!--more-->

# 代码学习

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
** 跟踪tracker的结果 **
在GOTRUN/src下执行
```bash
grep -nr Tracker
```
结果为：
>tracker/tracker_manager.h:10:class TrackerManager
>tracker/tracker.h:12:class Tracker
>train/tracker_trainer.h:12:class TrackerTrainer

tracker三个类：tracker/trackermanager/trackerTrainer

tracker.h 
```cpp
class Tracker
{
public:

  Tracker(const bool show_tracking);
//使用当前图像和指定的回归器即可完成Tracking
  // Estimate the location of the target object in the current image.
  virtual void Track(const cv::Mat& image_curr, RegressorBase* regressor,
             BoundingBox* bbox_estimate_uncentered);
//使用image和BoundingBox完成tracker的初始化
  // Initialize the tracker with the ground-truth bounding box of the first frame.
  void Init(const cv::Mat& image_curr, const BoundingBox& bbox_gt,
            RegressorBase* regressor);

  // Initialize the tracker with the ground-truth bounding box of the first frame.
  // VOTRegion is an object for initializing the tracker when using the VOT Tracking dataset.
  void Init(const std::string& image_curr_path, const VOTRegion& region,
            RegressorBase* regressor);

private:
  // Show the tracking output, for debugging.可视化跟踪结果
  void ShowTracking(const cv::Mat& target_pad, const cv::Mat& curr_search_region, const BoundingBox& bbox_estimate) const;

  // Predicted prior location of the target object in the current image.
  // This should be a tight (high-confidence) prior prediction area.  We will
  // add padding to this region.预测的当前帧中目标的bbox
  BoundingBox bbox_curr_prior_tight_;

  // Estimated previous location of the target object. 前一帧中估计的位置
  BoundingBox bbox_prev_tight_;

  // Full previous image. 前一帧图像
  cv::Mat image_prev_;

  // Whether to visualize the tracking results 是否要可视化结果
  bool show_tracking_;
};
```
从上面的代码和简单的注释可以看出，比较重要的类是init和track, 接下来我们查看和tracker.h在同一个目录里面的tracker.cpp里面的实现。

下面是init的函数实现，初始化的时候认为目标当前帧的位置和上一帧一致。
```cpp
void Tracker::Init(const cv::Mat& image, const BoundingBox& bbox_gt,
                   RegressorBase* regressor) {
  image_prev_ = image;
  bbox_prev_tight_ = bbox_gt;

  // Predict in the current frame that the location will be approximately the same
  // as in the previous frame.
  // TODO - use a motion model?
  bbox_curr_prior_tight_ = bbox_gt;

  // Initialize the neural network.
  regressor->Init();
}
```

下面是tracker的函数实现，

```cpp
void Tracker::Track(const cv::Mat& image_curr, RegressorBase* regressor,
                    BoundingBox* bbox_estimate_uncentered) {
  // Get target from previous image.
  cv::Mat target_pad;
  CropPadImage(bbox_prev_tight_, image_prev_, &target_pad);

  // Crop the current image based on predicted prior location of target.
  cv::Mat curr_search_region;
  BoundingBox search_location;
  double edge_spacing_x, edge_spacing_y;
  (bbox_curr_prior_tight_, image_curr, &curr_search_region, &search_location, &edge_spacing_x, &edge_spacing_y);
//看来主要是这个Regress能够回归处当前帧中的目标的大致位置;
  // Estimate the bounding box location of the target, centered and scaled relative to the cropped image.
  BoundingBox bbox_estimate;
  regressor->Regress(image_curr, curr_search_region, target_pad, &bbox_estimate);

  // Unscale the estimation to the real image size.当前帧中估计出的紧密的位置;
  BoundingBox bbox_estimate_unscaled;
  bbox_estimate.Unscale(curr_search_region, &bbox_estimate_unscaled);

  // Find the estimated bounding box location relative to the current crop.
  bbox_estimate_unscaled.Uncenter(image_curr, search_location, edge_spacing_x, edge_spacing_y, bbox_estimate_uncentered);

  if (show_tracking_) {
    ShowTracking(target_pad, curr_search_region, bbox_estimate);
  }

  // Save the image.
  image_prev_ = image_curr;

  // Save the current estimate as the location of the target.
  bbox_prev_tight_ = *bbox_estimate_uncentered;

  // Save the current estimate as the prior prediction for the next image.
  // TODO - replace with a motion model prediction?
  bbox_curr_prior_tight_ = *bbox_estimate_uncentered;
}

```
这段代码里有两处比较重要：一处是regressor->regess用来回归出当前帧中的扩充位置，具体在下面分析，第二处是  BoundingBox bbox_estimate_unscaled;这个BoundingBox结构体。从前面include的头文件可以看出这个boundingbox定义在 #include "helper/bounding_box.h"，这么看来这个helper文件夹里定义的都是辅助函数。我们来看一下这个目录里都定义了什么？
```bash
root@aa0afc645153:helper# ls
bounding_box.cpp  bounding_box.h  helper.cpp  helper.h  high_res_timer.cpp  high_res_timer.h  image_proc.cpp  image_proc.h
```
具体boundingbox定义了什么，后面分析，先回归正题。

注意到track函数里面，两次调用了CropPadImage函数，我们就来对这个函数分析一下，先追踪一个这个函数在那儿定义的呢？
在src目录下执行：
```bash
grep -nr CropPadImage
```
得到结果 
>helper/image_proc.cpp:58:void CropPadImage(const BoundingBox& bbox_tight, const cv::Mat& image, cv::Mat* pad_image,
helper/**image_proc.cpp:63**:  ComputeCropPadImageLocation(bbox_tight, image, pad_image_location);
helper/image_proc.h:11:void CropPadImage(const BoundingBox& bbox_tight, const cv::Mat& image, cv::Mat* pad_image);
helper/image_proc.h:12:void CropPadImage(const BoundingBox& bbox_tight, const cv::Mat& image, cv::Mat* pad_image,
helper/**image_proc.h:** 18:void ComputeCropPadImageLocation(const BoundingBox& bbox_tight, const cv::Mat& image, BoundingBox* pad_image_location);

同样这个辅助函数也定义在helper路径下面，那么既然绕不开，我们就来看一看吧。






**跟踪regressor的结果**
执行：
```bash
grep -nr Regressor
```
结果为:

>network/**regressor.h**:17:class Regressor : public RegressorBase {
network/regressor.h:22:  Regressor(const std::string& train_deploy_proto,
network/regressor.h:28:  Regressor(const std::string& train_deploy_proto,
network/**regressor_train_base.cpp:**7:RegressorTrainBase::RegressorTrainBase(const std::string& solver_file)
network/regressor_base.h:14:class RegressorBase
network/regressor_base.h:17:  RegressorBase();
network/**regressor_train_base.h**:32:class RegressorTrainBase 
network/regressor_train_base.h:35:  RegressorTrainBase(const std::string& solver_file);
network/**regressor_train.h:**7:class RegressorTrain : public Regressor, public RegressorTrainBase
network/regressor_train.h:10:  RegressorTrain(const std::string& deploy_proto,
network/regressor_train.h:16:  RegressorTrain(const std::string& deploy_proto,
network/regressor.cpp:16:Regressor::Regressor(const string& deploy_proto,
network/**regressor.cpp:**29:Regressor::Regressor(const string& deploy_proto,
network/**regressor_train.cpp**:10:RegressorTrain::RegressorTrain(const std::string& deploy_proto,


从结果可以看出总共regressor这个类主要在三个头文件里： regressor.h, regressor_train_base.cpp, regressor_train.h。所以我们先从头文件regression.h下手。

```cpp
class Regressor : public RegressorBase
```

从这句话可以看出类的继承关系，regressor继承自regressorbase, 而从上面的头文件可知，regressorbase类定义在network/regressor_base.h里面。







**跟踪TrackerVisualizer的结果**

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


  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1499156314202.jpg