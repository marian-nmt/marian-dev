add_subdirectory(3rd_party)

include_directories(.)
include_directories(3rd_party)

if(CUDA_FOUND)
  cuda_add_library(marian
    3rd_party/cnpy/cnpy.cpp
    3rd_party/exception.cpp
    graph/expression_graph.cpp
    graph/expression_operators.cpp
    graph/node.cpp
    graph/node_operators.cpp
    tensors/tensor.cu
    tensors/tensor.cpp
    tensors/device_gpu.cu
    tensors/device_cpu.cpp
    kernels/tensor_operators.cu
    kernels/tensor_operators.cpp
    kernels/dropout.cu
    kernels/dropout.cpp
    kernels/sparse.cu
    kernels/sparse.cpp
    layers/param_initializers.cpp
    rnn/attention.cpp
    rnn/cells.cpp
    optimizers/clippers.cpp
    optimizers/optimizers.cpp
    common/utils.cpp
    common/logging.cpp
    common/config.cpp
    common/config_parser.cpp
    training/multinode.cpp
    translator/history.cpp
    translator/output_collector.cpp
    translator/nth_element.cu
    translator/nth_element.cpp
    translator/helpers.cu
    translator/helpers.cpp
    data/vocab.cpp
    data/corpus.cpp
    data/text_input.cpp
    rescorer/score_collector.cpp
    $<TARGET_OBJECTS:libyaml-cpp>
    STATIC
  )
else(CUDA_FOUND)
  add_library(marian STATIC
    3rd_party/cnpy/cnpy.cpp
    3rd_party/exception.cpp
    graph/expression_graph.cpp
    graph/expression_operators.cpp
    graph/node.cpp
    graph/node_operators.cpp
    tensors/tensor.cpp
    tensors/device_cpu.cpp
    kernels/tensor_operators.cpp
    kernels/dropout.cpp
    kernels/sparse.cpp
    layers/param_initializers.cpp
    rnn/attention.cpp
    rnn/cells.cpp
    optimizers/clippers.cpp
    optimizers/optimizers.cpp
    common/utils.cpp
    common/logging.cpp
    common/config.cpp
    common/config_parser.cpp
    training/multinode.cpp
    translator/history.cpp
    translator/output_collector.cpp
    translator/nth_element.cpp
    translator/helpers.cpp
    data/vocab.cpp
    data/corpus.cpp
    data/text_input.cpp
    rescorer/score_collector.cpp
    $<TARGET_OBJECTS:libyaml-cpp>
  )
endif(CUDA_FOUND)

set_target_properties(marian PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set_target_properties(marian PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

if(CUDA_FOUND)
  cuda_add_executable(marian_train command/marian.cpp)
else(CUDA_FOUND)
  add_executable(marian_train command/marian.cpp)
endif(CUDA_FOUND)
set_target_properties(marian_train PROPERTIES OUTPUT_NAME marian)

add_executable(marian_translate command/s2s_translator.cpp)
set_target_properties(marian_translate PROPERTIES OUTPUT_NAME s2s)

add_executable(marian_rescore command/s2s_rescorer.cpp)
set_target_properties(marian_rescore PROPERTIES OUTPUT_NAME rescorer)

set(EXECUTABLES ${EXECUTABLES} marian_train marian_translate marian_rescore)

if(COMPILE_SERVER)
  add_executable(marian_server command/s2s_server.cpp)
  set_target_properties(marian_server PROPERTIES OUTPUT_NAME server)
  set(EXECUTABLES ${EXECUTABLES} marian_server)
endif(COMPILE_SERVER)

foreach(exec ${EXECUTABLES})
  target_link_libraries(${exec} marian ${EXT_LIBS})
  if(CUDA_FOUND)
    cuda_add_cublas_to_target(${exec})
  endif(CUDA_FOUND)
  set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endforeach(exec)

#add_executable(
#  align2steps
#  tools/align2steps.cpp
#)

#set_target_properties(align2steps PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

if(PYTHONLIBS_FOUND)
  add_subdirectory(python)
endif(PYTHONLIBS_FOUND)

if(COMPILE_TESTS)
  set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party)
  add_library(Catch INTERFACE)
  target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

  add_subdirectory(tests)
endif(COMPILE_TESTS)

if(COMPILE_EXAMPLES)
  add_subdirectory(examples)
endif(COMPILE_EXAMPLES)
