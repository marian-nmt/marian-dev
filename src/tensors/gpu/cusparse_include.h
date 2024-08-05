// header file to include cusparse.h while ignoring deprecated warnings locally

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <cusparse.h>

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif