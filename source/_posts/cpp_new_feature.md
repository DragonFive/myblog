---
title: c++使用7年后的经验总结
date: 2017/7/31 17:38:58

categories:
- 编程
tags:
- cpp
---
[TOC]

一年前写了一篇[c++11新特性详解](https://zhuanlan.zhihu.com/p/21930436), 这里总结了一些c++的编程经验。

![enter description here][1]
<!--more-->

# c++11 new feature

## variadic Templates

**可变的模板参数**，参数量可以变化，举个例子：

### 对于函数
```cpp
void print()//对于print没有参数的时候，调用这个函数。
{

}

template <typename T, typename... Types>
void print(const T& firstArg, const Types&... args)
{
	cout<<firstArg<<endl;
	print(args...);//这里使用了递归的方式
}

//下面的调用方式
print(7.5, "hello", 1.3, bitset<16>(377));

```
注意了，上面的这种方式由于不知道有多少个参数，所以使用了递归的方式。为了**处理最后的状态**，上面声明了零参数的print;

如果想知道args有多少个，可以使用sizeof...(args);

这种方式可以把参数逐一分开

### 对于类

tuple的实现就是依赖可变参数


```cpp
template <typename... Values> class tuple;
template<> class tuple<>(); // 处理终止条件

template<typename Head, typename... Tail>
class tuple<Head, Tail...> :private tuple<Tail...>//实现递归定义
{
		typedef tuple<Tail...> inherited;// 重新起个名字;
	public :
		tuple(){}
		tuple(Head v, Tail... vtail):m_head(v), inherited(vtail){}
	protected:
		Head m_head;
};

```


## 右值引用，移动构造函数，与 move函数

[C\++札记 C++11中右值引用与移动构造函数](https://mp.weixin.qq.com/s?__biz=MzIwNTc4NTEwOQ==&mid=2247483724&idx=1&sn=e6b4304547607ba7d7714716913f5259&chksm=972ad036a05d592076dc6f37a3df5cc5ee18678eae56c35d5008c7ff046ec5239b5bf55eb9c0&mpshare=1&scene=1&srcid=0825UmZbQZ3YCp4bvw8rWvlT&pass_ticket=gVdGvS19sLXA%2FE1%2BMc0kM3FFWwFheq%2FmuFwE1zEFYm9r5SAU364mtiB7sMLj0ez3#rd)

**左值与右值**

C++中，可以这么理解：对于一个表达式，凡是对其取地址（&）操作可以成功的都是左值，否则就是右值。

需要注意的是：临时对象是右值 。

**左值引用与右值引用**

右值引用的形式为：类型 && a= 被引用的对象 ，此外需要注意的是**右值引用只能绑定到右值**，如int && x = 3;而形如 int x = 3; int && y = x则是错误的，因为x是一个左值。

左值引用只能绑定到左值上。

**拷贝构造函数与移动构造函数**

拷贝构造函数就是需要拷贝一份样本给自己对象，而移动构造函数就是把样本的内存拿过来自己用。
```cpp
Test(Test && rhs):m_p(rhs.m_p)
{
    rhs.m_p = nullptr;
}

```
但是移动构造函数只能接受右值，所以如果是左值想传入，需要使用std::move()函数。

## 智能指针 

[C++智能指针](https://mp.weixin.qq.com/s?__biz=MzIwNTc4NTEwOQ==&mid=2247483809&idx=1&sn=373d64600b944be7258304119dae247e&chksm=972ad0dba05d59cd6cf08f3874db2c72f6a66f66ab8f516cc576c3e8752c3aa6c385e936696e&mpshare=1&scene=21&srcid=1007uaWOzcd3a12MYWEEsptn#wechat_redirect)

C++ STL为我们提供了四种智能指针：**auto_ptr、unique_ptr、shared_ptr和weak_ptr**；其中auto_ptr是C++98提供，在C++11中建议摒弃不用。

### 为什么不建议使用auto_ptr

```cpp
auto_ptr<int> px(new int(8));
auto_ptr<int> py;
py = px;
```
为了避免上述的代码把内存delete两次，智能指针有两种机制。
1. 建立所有权，一个对象同一时刻智能有一个智能指针可以拥有。只有拥有所有权的智能智能才能delete这个对象，unique_ptr和auto_ptr就是这种机制
2. 引用计数，share_ptr就是这么做的

当py=px，后如果再想访问px, 对于auto_ptr会在运行时报segment_fault的错误，而对于unique_ptr则会在编译期就报错。

### 自己实现share_pre

```cpp
#include <iostream>
using namespace std;

template<class T>
class SmartPtr
{
public:
    SmartPtr(T *p);
    ~SmartPtr();
    SmartPtr(const SmartPtr<T> &orig);                // 浅拷贝
    SmartPtr<T>& operator=(const SmartPtr<T> &rhs);    // 浅拷贝
private:
    T *ptr;
    // 将use_count声明成指针是为了方便对其的递增或递减操作
    int *use_count;
};

template<class T>
SmartPtr<T>::SmartPtr(T *p) : ptr(p)
{
    try
    {
        use_count = new int(1);
    }
    catch (...)
    {
        delete ptr;
        ptr = nullptr;
        use_count = nullptr;
        cout << "Allocate memory for use_count fails." << endl;
        exit(1);
    }

    cout << "Constructor is called!" << endl;
}

template<class T>
SmartPtr<T>::~SmartPtr()
{
    // 只在最后一个对象引用ptr时才释放内存
    if (--(*use_count) == 0)
    {
        delete ptr;
        delete use_count;
        ptr = nullptr;
        use_count = nullptr;
        cout << "Destructor is called!" << endl;
    }
}

template<class T>
SmartPtr<T>::SmartPtr(const SmartPtr<T> &orig)
{
    ptr = orig.ptr;
    use_count = orig.use_count;
    ++(*use_count);
    cout << "Copy constructor is called!" << endl;
}

// 重载等号函数不同于复制构造函数，即等号左边的对象可能已经指向某块内存。
// 这样，我们就得先判断左边对象指向的内存已经被引用的次数。如果次数为1，
// 表明我们可以释放这块内存；反之则不释放，由其他对象来释放。
template<class T>
SmartPtr<T>& SmartPtr<T>::operator=(const SmartPtr<T> &rhs)
{
    // 《C++ primer》：“这个赋值操作符在减少左操作数的使用计数之前使rhs的使用计数加1，
    // 从而防止自身赋值”而导致的提早释放内存
    ++(*rhs.use_count);

    // 将左操作数对象的使用计数减1，若该对象的使用计数减至0，则删除该对象
    if (--(*use_count) == 0)
    {
        delete ptr;
        delete use_count;
        cout << "Left side object is deleted!" << endl;
    }

    ptr = rhs.ptr;
    use_count = rhs.use_count;
    
    cout << "Assignment operator overloaded is called!" << endl;
    return *this;
}

```


## 一些小更新

### nullptr
可以用来替代NULL,避免因为NULL=0带来的一些混淆。

```cpp
void f(int);
void f(void *);

//下面调用
f(0);//调用f(int)
f(NULL);//调用f(int)
f(nullptr);//调用f(void *)

```
### auto
当type比较长的时候，比如iterator定义的时候可以用auto替换，也可以定义返回值类型。

或者比较复杂：比如用
```cpp
auto L = [](int x)->bool{};
```
### 统一的初始化方法-大括号

```cpp
int values[] {1,2,3};
vector<int> v{2,3,4,5};
```


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
14. 在一个独立的语句中将 new 出来的对象存入智能指针。
15. 绝不要返回一个局部栈对象的指针或引用，绝不要返回一个被分配的堆对象的引用，如果存在需要一个以上这样的对象的可能性时，绝不要返回一个局部 static 对象的指针或引用。
16. 在子类中使用与父类public成员函数同名的函数会造成覆盖，继续使用父类函数有两种方式：a.using b. 转调 [Item 33: 避免覆盖（hiding）“通过继承得到的名字”](https://wizardforcel.gitbooks.io/effective-cpp/content/35.html)
17. 

# reference
[c++11新特性](http://blog.csdn.net/wangshubo1989/article/details/50575008)

[effective-cpp](https://wizardforcel.gitbooks.io/effective-cpp/content/)


[c++单例模型](http://blog.yangyubo.com/2009/06/04/best-cpp-singleton-pattern/)


  [1]: https://www.github.com/DragonFive/CVBasicOp/raw/master/1505722824118.jpg
  [2]: https://www.github.com/DragonFive/CVBasicOp/raw/master/%E5%B0%8F%E4%B9%A6%E5%8C%A0/1500388501263.jpg