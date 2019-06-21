#pragma once

#define TRANSFORM_LIST_1(transform, head) transform(1, head)
#define TRANSFORM_LIST_2(transform, head, ...) transform(2, head), TRANSFORM_LIST_1(transform, __VA_ARGS__)
#define TRANSFORM_LIST_3(transform, head, ...) transform(3, head), TRANSFORM_LIST_2(transform, __VA_ARGS__)
#define TRANSFORM_LIST_4(transform, head, ...) transform(4, head), TRANSFORM_LIST_3(transform, __VA_ARGS__)
#define TRANSFORM_LIST_5(transform, head, ...) transform(5, head), TRANSFORM_LIST_4(transform, __VA_ARGS__)
#define TRANSFORM_LIST_6(transform, head, ...) transform(6, head), TRANSFORM_LIST_5(transform, __VA_ARGS__)
#define TRANSFORM_LIST_7(transform, head, ...) transform(7, head), TRANSFORM_LIST_6(transform, __VA_ARGS__)
#define TRANSFORM_LIST_8(transform, head, ...) transform(8, head), TRANSFORM_LIST_7(transform, __VA_ARGS__)
#define TRANSFORM_LIST_9(transform, head, ...) transform(9, head), TRANSFORM_LIST_8(transform, __VA_ARGS__)
#define TRANSFORM_LIST(n, transform, ...) TRANSFORM_LIST_##n(transform, __VA_ARGS__)

#define MAKE_ARG_NAME(i, type) arg##i
#define MAKE_ARG_TYPE_NAME(i, type) type MAKE_ARG_NAME(i, type)

#define ARGS_NAMES(n, ...) TRANSFORM_LIST(n, MAKE_ARG_NAME, __VA_ARGS__)
#define ARGS_TYPES_NAMES(n, ...) TRANSFORM_LIST(n, MAKE_ARG_TYPE_NAME, __VA_ARGS__)

#ifdef CUDA_FOUND

#define DISPATCH(function, n, ...)                                \
  namespace gpu {                                                 \
    void function(__VA_ARGS__);                                   \
  }                                                               \
  namespace cpu {                                                 \
    void function(__VA_ARGS__);                                   \
  }                                                               \
  static inline void function(ARGS_TYPES_NAMES(n, __VA_ARGS__)) { \
    if(arg1->getBackend()->getDeviceId().type == DeviceType::gpu) \
      gpu::function(ARGS_NAMES(n, __VA_ARGS__));                  \
    else                                                          \
      cpu::function(ARGS_NAMES(n, __VA_ARGS__));                  \
  }

#else

#define DISPATCH(function, n, ...)                                \
  namespace cpu {                                                 \
    void function(__VA_ARGS__);                                   \
  }                                                               \
  static inline void function(ARGS_TYPES_NAMES(n, __VA_ARGS__)) { \
    cpu::function(ARGS_NAMES(n, __VA_ARGS__));                    \
  }

#endif

#define DISPATCH1(function, type1) \
  DISPATCH(function, 1, type1)
#define DISPATCH2(function, type1, type2) \
  DISPATCH(function, 2, type1, type2)
#define DISPATCH3(function, type1, type2, type3) \
  DISPATCH(function, 3, type1, type2, type3)
#define DISPATCH4(function, type1, type2, type3, type4) \
  DISPATCH(function, 4, type1, type2, type3, type4)
#define DISPATCH5(function, type1, type2, type3, type4, type5) \
  DISPATCH(function, 5, type1, type2, type3, type4, type5)
#define DISPATCH6(function, type1, type2, type3, type4, type5, type6) \
  DISPATCH(function, 6, type1, type2, type3, type4, type5, type6)
#define DISPATCH7(function, type1, type2, type3, type4, type5, type6, type7) \
  DISPATCH(function, 7, type1, type2, type3, type4, type5, type6, type7)
#define DISPATCH8(function, type1, type2, type3, type4, type5, type6, type7, type8) \
  DISPATCH(function, 8, type1, type2, type3, type4, type5, type6, type7, type8)
#define DISPATCH9(function, type1, type2, type3, type4, type5, type6, type7, type8, type9) \
  DISPATCH(function, 9, type1, type2, type3, type4, type5, type6, type7, type8, type9)
