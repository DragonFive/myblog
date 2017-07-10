---
title: cpp_new_feature
date: 2017/6/8 17:38:58

categories:
- 编程
tags:
- cpp
---
[TOC]


<!--more-->

# effective_Cpp

1. constructors（构造函数）被声明为 explicit（显式）通常比 non-explicit（非显式）更可取，因为它们可以防止编译器执行意外的（常常是无意识的）type conversions（类型转换）。

2. copy constructor（拷贝构造函数）被用来以一个 object（对象）来初始化同类型的另一个 object（**新对象**），copy assignment operator（拷贝赋值运算符）被用来将一个 object（对象）中的值拷贝到同类型的另一个 object（对象）。

3. 如果 const 出现在星号左边，则指针 pointed to（指向）的内容为 constant（常量）；如果 const 出现在星号右边，则 pointer itself（指针自身）为 constant

4. 

# reference
[c++11新特性](http://blog.csdn.net/wangshubo1989/article/details/50575008)