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

3. 如果 const 出现在星号左边，则指针 pointed to（指向）的内容为 constant（常量）；如果 const 出现在星号右边，则 pointer itself（指针自身）为 constant。声明一个 iterator 为 const 就类似于声明一个 pointer（指针）为 const（也就是说，声明一个 T\* const pointer（指针））：不能将这个 iterator 指向另外一件不同的东西，但是它所指向的东西本身可以变化。如果你要一个 iterator 指向一个不能变化的东西（也就是一个 const T\* pointer（指针）的 STL 对等物），你需要一个 const_iterator：
>const std::vector<int>::iterator iter
std::vector<int>::const_iterator cIter

member functions（成员函数）在只有 constness（常量性）不同时是可以被 overloaded（重载）的，但这是 C++ 的一个重要特性。
```cpp
  const char& operator[](std::size_t position) const   // operator[] for
  { return text[position]; }                           // const objects

  char& operator[](std::size_t position)               // operator[] for
  { return text[position]; }                           // non-const objects
```
改变一个返回 built-in type（内建类型）的函数的返回值总是非法的。
4. 

# reference
[c++11新特性](http://blog.csdn.net/wangshubo1989/article/details/50575008)