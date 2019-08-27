# - Try to find restclient-cpp (https://github.com/mrtazz/restclient-cpp)
# Once done this will define
#  LIBRESTCLIENT_CPP_FOUND - System has restclient-cpp
#  LIBRESTCLIENT_CPP_INCLUDE_DIRS - The restclient-cpp include directories
#  LIBRESTCLIENT_CPP_LIBRARIES - The libraries needed to use restclient-cpp
#  LIBRESTCLIENT_CPP_DEFINITIONS - Compiler switches required for using restclient-cpp

find_package(PkgConfig)
pkg_check_modules(PC_RESTCLIENT_CPP restclient-cpp)
set(LIBRESTCLIENT_CPP_DEFINITIONS ${PC_RESTCLIENT_CPP_CFLAGS_OTHER})

find_path(LIBRESTCLIENT_CPP_INCLUDE_DIR
          restclient-cpp/restclient.h restclient-cpp/connection.h restclient-cpp/helpers.h restclient-cpp/version.h
          HINTS ${PC_RESTCLIENT_CPP_INCLUDEDIR} ${PC_RESTCLIENT_CPP_INCLUDE_DIRS}
          PATH_SUFFIXES restclient-cpp)
SET(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
find_library(LIBRESTCLIENT_CPP_LIBRARY restclient-cpp
             HINTS ${PC_RESTCLIENT_CPP_LIBDIR} ${PC_RESTCLIENT_CPP_LIBRARY_DIRS})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
        Restclient-cpp DEFAULT_MSG
        LIBRESTCLIENT_CPP_INCLUDE_DIR LIBRESTCLIENT_CPP_LIBRARY)
mark_as_advanced(LIBRESTCLIENT_CPP_INCLUDE_DIR LIBRESTCLIENT_CPP_LIBRARY)

set(LIBRESTCLIENT_CPP_INCLUDE_DIRS ${LIBRESTCLIENT_CPP_INCLUDE_DIR})
set(LIBRESTCLIENT_CPP_LIBRARIES ${LIBRESTCLIENT_CPP_LIBRARY})