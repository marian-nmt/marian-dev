
include_directories(.)

add_subdirectory(./yaml-cpp)

add_subdirectory(./SQLiteCpp)
include_directories(./SQLiteCpp/include)

add_subdirectory(./pathie-cpp)
include_directories(./pathie-cpp/include)

add_subdirectory(./zlib)
include_directories(./zlib)

include_directories(./CLI)

include(ExternalProject)

set(INSTALLS "")

if(CUDA_FOUND)
  if(USE_NCCL)

    # disables compilation for sm_30 to avoid ptxas warning... that's general Kepler support. But K80s are supported for instance by sm_35
    set(GENCODE "-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61")
    ExternalProject_Add(nccl_install
      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/nccl
      BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/nccl
      CONFIGURE_COMMAND ""
      BUILD_COMMAND
        $(MAKE) -f ${CMAKE_CURRENT_SOURCE_DIR}/nccl/Makefile src.build
        BUILDDIR=${CMAKE_BINARY_DIR}/local CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}
        CUDA8_GENCODE=${GENCODE} CXX=${CMAKE_CXX_COMPILER}
      INSTALL_COMMAND "")

    set_target_properties(nccl PROPERTIES IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/local/lib/libnccl_static.a)
    add_dependencies(nccl nccl_install)
    set(INSTALLS ${INSTALLS} nccl_install)

  endif(USE_NCCL)
endif(CUDA_FOUND)

if(USE_SENTENCEPIECE)

  ExternalProject_Add(protobuf_install
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/protobuf
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/protobuf
    CMAKE_CACHE_ARGS
      "-DCMAKE_BUILD_TYPE:STRING=Release"
      "-Dprotobuf_BUILD_TESTS:BOOL=OFF"
      "-Dprotobuf_BUILD_EXAMPLES:BOOL=OFF"
      "-Dprotobuf_WITH_ZLIB:BOOL=OFF"
      "-DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}"
      "-DCMAKE_INSTALL_PREFIX:STRING=${CMAKE_BINARY_DIR}/local"
    SOURCE_SUBDIR cmake
    INSTALL_COMMAND $(MAKE) install --silent)

  set_target_properties(protobuf PROPERTIES IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/local/lib/libprotobuf.a)
  add_dependencies(protobuf protobuf_install)
  set(INSTALLS ${INSTALLS} protobuf_install)

  ExternalProject_Add(sentencepiece_install
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sentencepiece
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/sentencepiece
    CMAKE_CACHE_ARGS
      "-DCMAKE_BUILD_TYPE:STRING=Release"
      "-DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}"
      "-DSPM_ENABLE_SHARED:BOOL=OFF"
      "-DSPM_ENABLE_TCMALLOC:BOOL=OFF"
      "-DSPM_TCMALLOC_STATIC:BOOL=ON"
      "-DCMAKE_INSTALL_PREFIX:STRING=${CMAKE_BINARY_DIR}/local"
    INSTALL_COMMAND $(MAKE) install --silent)

  add_dependencies(sentencepiece_install protobuf_install)

  set_target_properties(sentencepiece PROPERTIES IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/local/lib/libsentencepiece.a)
  add_dependencies(sentencepiece sentencepiece_install)

  set_target_properties(sentencepiece_train PROPERTIES IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/local/lib/libsentencepiece_train.a)
  add_dependencies(sentencepiece_train sentencepiece_install)
  set(INSTALLS ${INSTALLS} sentencepiece_install)

  set_directory_properties(PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/local)

  add_custom_target(3rd_party_installs DEPENDS ${INSTALLS})

endif(USE_SENTENCEPIECE)

