[![Build Status](https://travis-ci.org/david-grs/static_any.svg?branch=master)](https://travis-ci.org/david-grs/static_any)
[![Build status](https://ci.appveyor.com/api/projects/status/cm8wnp77xaywril0/branch/master?svg=true)](https://ci.appveyor.com/project/david-grs/static-any/branch/master)
[![Coverage Status](https://coveralls.io/repos/github/david-grs/static_any/badge.svg?branch=master)](https://coveralls.io/github/david-grs/static_any?branch=master)

static\_any\<S\>
================
A container for generic (as *general*) data type &mdash; like boost.any. However:

 - It is **~10x faster** than boost.any, mainly because there is no memory allocation
 - As it lies on the stack, it is **cache-friendly**, close to your other class attributes
 - There is a very **small space overhead**: a fixed overhead of 8 bytes


static\_any\<S\> is also **safe**:
 - operations meet the strong exception guarantee
 - compile time check during the assignment, to ensure that its buffer is big enough to store the value
 - runtime check before any conversions, to ensure that the stored type is the one's requested by the user


Example
-------

```c++
    static_any<32> a;
    static_assert(sizeof(a) == 32 + 8, "any has a fixed overhead of 8 bytes");

    a = 1234;
    ASSERT_EQ(1234, a.get<int>());

    a = std::string("foobar");
    ASSERT_EQ("foobar", a.get<std::string>());

    try {
      // This will throw as a std::string has been stored
      std::cout << any_cast<int>(a);
    }
    catch(bad_any_cast& ex) {
    }

    struct A : std::array<char, 32> {};
    a = A();

    struct B : A { char c; };  
    
    // Does not build: sizeof(B) is too big for static_any<16>
    a = B();
```


---

static\_any\_t\<S\>
===================
A container similar to static\_any\<S\>, but for trivially copyable types only. The differences:

 - **No space overhead**
 - **Faster**
 - **Unsafe**: there is no check when you try to access your data



---

Benchmarks
==========
As the main advantage of static\_any(\_t\<S\>) is speed, here is a comparison of assign/get operations on a POD/non-POD types &mdash; an integer and a std::string &mdash; between
boost.any, QVariant, static\_any\<S\> and static\_any\_t\<S\>.


**assign a double**
```
Test             Time (ns)
--------------------------
qvariant                36
boost.any               57
std::any                 4
static_any<8>            4
static_any_t<8>          0
```

**get a double**
```
Test             Time (ns)
--------------------------
qvariant                 9
boost.any               12
std::any                 9
static_any<8>            9
static_any_t<8>          6
```

**assign a small struct**
```
Test             Time (ns)
--------------------------
qvariant               294
boost.any               47
std::any                54
static_any<8>           12
static_any_t<8>          0
```

**get a small struct**
```
Test             Time (ns)
--------------------------
qvariant                12
boost.any               48
std::any                 6 
static_any<8>            4
static_any_t<8>          0
```

**assign a string**
```
Test             Time (ns)
--------------------------
qvariant               400
boost.any              143
std::any               139
static_any<32>          92
```

**get a string**
```
Test             Time (ns)
--------------------------
qvariant                31
boost.any               64
std::any                30
static_any<32>           4
```

What I used for this benchmark:
 - Boost 1.54
 - Qt5
 - GCC 5.3.0
 - CPU i7-3537U

The code is available in *benchmark.cpp*. There is a dependency on *geiger*, a benchmarking library I developed. In order to build and run the benchmark, first install geiger:

```
git clone https://github.com/david-grs/geiger
cd geiger && mkdir build && cd build
cmake .. && make install
```

Then pass *-DBUILD_BENCHMARK=1* to your cmake command.

