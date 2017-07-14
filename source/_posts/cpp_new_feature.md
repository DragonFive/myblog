---
title: 7年c++使用经验心得
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

```cpp
const std::vector<int>::iterator iter
std::vector<int>::const_iterator cIter
```

member functions（成员函数）在只有 constness（常量性）不同时是可以被 overloaded（重载）的，但这是 C++ 的一个重要特性。



```cpp
  const char& operator[](std::size_t position) const   // operator[] for
  { return text[position]; }                           // const objects

  char& operator[](std::size_t position)               // operator[] for
  { return text[position]; }                           // non-const objects
```
改变一个返回 built-in type（内建类型）的函数的返回值总是非法的。
4. mutable 将 non-static data members（非静态数据成员）从 bitwise constness（二进制位常量性）的约束中解放出来：

```cpp
class CTextBlock {
public:


  std::size_t length() const;

private:
  char *pText;

  mutable std::size_t textLength;         // these data members may
  mutable bool lengthIsValid;             // always be modified, even in
};                                        // const member functions

std::size_t CTextBlock::length() const
{
  if (!lengthIsValid) {
    textLength = std::strlen(pText);      // now fine
    lengthIsValid = true;                 // also fine
  }

  return textLength;
}
```


5. 根据 const member function（成员函数）实现它的 non-const 版本的技术却非常值得掌握
```cpp
class TextBlock {
public:

  ...

  const char& operator[](std::size_t position) const     // same as before
  {
    ...
    ...
    ...
    return text[position];
  }

  char& operator[](std::size_t position)         // now just calls const op[]
  {
    return
      const_cast<char&>(                         // cast away const on op[]'s return type;
        static_cast<const TextBlock&>(*this)     // add const to *this's type;
          [position]                             // call const version of op[]
      );
  }

...

};
```
6. 通常它有更高的效率。assignment-based（基于赋值）的版本会首先调用 default constructors（缺省构造函数）初始化 theName，theAddress 和 thePhones，然而很快又在 default-constructed（缺省构造）的值之上赋予新值。那些 default constructions（缺省构造函数）所做的工作被浪费了。而 member initialization list（成员初始化列表）的方法避免了这个问题，因为**initialization list（初始化列表**中的 arguments（参数）就可以作为各种 data members（数据成员）的 constructor（构造函数）所使用的 arguments（参数）。在这种情况下，theName 从 name 中 copy-constructed（拷贝构造），theAddress 从 address 中 copy-constructed（拷贝构造），thePhones 从 phones 中 copy-constructed（拷贝构造）。对于大多数类型来说，只调用一次 copy constructor（拷贝构造函数）的效率比先调用一次 default constructor（缺省构造函数）再调用一次 copy assignment operator（拷贝赋值运算符）的效率要高（有时会高很多）。
```cpp

ABEntry::ABEntry(const std::string& name, const std::string& address,
                 const std::list<PhoneNumber>& phones)

: theName(name),
  theAddress(address),                  // these are now all initializations
  thePhones(phones),
  numTimesConsulted(0)

{}    
```
在一个 class（类）内部，data members（数据成员）按照它们被声明的顺序被初始化。例如，在 ABEntry 中，theName 总是首先被初始化，theAddress 是第二个，thePhones 第三，numTimesConsulted 最后。即使它们在 member initialization list（成员初始化列表）中以一种不同的顺序排列（这不幸合法）.

7. 赋值运算符函数需要注意的点，因为当前对象已经经过了初始化，所以已经有了存储空间，那么使用前就需要先**释放**之前申请的存储空间;另外，在赋值之前需要先判断这两个对象**是否是同一个对象**。


8. struct与class的区别，如果没有标明访问权限级别，在struct中默认的是public,而在class中默认的是private.如果 base classes（基类）将 copy assignment operator（拷贝赋值运算符）声明为 private，编译器拒绝为从它继承的 derived classes（派生类）生成 implicit copy assignment operators（隐式拷贝赋值运算符）。如果成员变量中有指针，引用或者const类型就需要自己定义默认构造函数、默认复制构造函数、默认赋值操作符。
9. 将复制构造函数和赋值操作符函数声明为private，同时不给出定义就可以防止被复制。
10. polymorphic base classes（多态基类）应该声明 virtual destructor（虚拟析构函数）。如果一个 class（类）有任何 virtual functions（虚拟函数），它就应该有一个 virtual destructor（虚拟析构函数）。
11. 高效的拷贝构造函数。
```cpp
Widget& Widget::operator=(Widget rhs)   // rhs is a copy of the object
{                                       // passed in — note pass by val

  swap(rhs);                            // swap *this's data with
                                        // the copy's
  return *this;
}
```
12. 当你写一个拷贝函数，需要保证（1）拷贝所有本地数据成员以及（2）调用所有基类中的适当的拷贝函数。
```cpp
PriorityCustomer::operator=(const PriorityCustomer& rhs)
{
  logCall("PriorityCustomer copy assignment operator");

  Customer::operator=(rhs);           // assign base class parts
  priority = rhs.priority;

  return *this;
}
```
不要试图依据类内的一个拷贝函数实现同一类里的另一个拷贝函数。作为代替，将通用功能放入第三个供双方调用的函数
13. shared_ptr 和auto_ptr能够有效的管理堆上的资源，保证他们能够被释放，share_ptr更好用。而且支持拷贝资源管理。
```cpp
class Lock {
public:
 explicit Lock(Mutex *pm)       // init shared_ptr with the Mutex
 : mutexPtr(pm, unlock)         // to point to and the unlock func
 {                              // as the deleter

   lock(mutexPtr.get());        // see Item 15 for info on "get"
 }
private:
 std::tr1::shared_ptr<Mutex> mutexPtr;    // use shared_ptr
};
```
shared_ptr 和 auto_ptr 都提供一个 get 成员函数进行显示转换。
14. 在一个独立的语句中将 new 出来的对象存入智能指针。智能指针能够提供一种异常安全的功能，reset函数能够修改智能指针指向的内容，然后删除之前的内容，这意味着如果新内容创建失败就不会delete之前的内容，保证了异常安全。copy and swap策略可以保证线程安全。
15. 避免返回对象内部构件的句柄（引用，指针，或迭代器）。这样会提高封装性，帮助 const 成员函数产生 cosnt 效果，并将空悬句柄产生的可能性降到最低.
16. inline函数：在类内声明的函数默认成为inline函数。inline 函数一般都是定义在头文件中的。
# reference
[c++11新特性](http://blog.csdn.net/wangshubo1989/article/details/50575008)

[effective-cpp](https://wizardforcel.gitbooks.io/effective-cpp/content/)


[c++单例模型](http://blog.yangyubo.com/2009/06/04/best-cpp-singleton-pattern/)