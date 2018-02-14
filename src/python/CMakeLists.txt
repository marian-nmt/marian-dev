# All or part of this file was contributed by Intel under license:
#   Copyright (C) 2017-2018 Intel Corporation
#   SPDX-License-Identifier: MIT

if(CUDA_FOUND)
  cuda_add_library(pymarian SHARED
    mariannmt.cpp
    ../3rd_party/cnpy/cnpy.cpp
    ../3rd_party/exception.cpp
    ../graph/expression_graph.cpp
    ../graph/expression_operators.cpp
    ../graph/node.cpp
    ../graph/node_operators.cpp
    ../tensors/tensor.cu
    ../tensors/tensor.cpp
    ../tensors/device_gpu.cu
    ../tensors/device_cpu.cpp
    ../kernels/tensor_operators.cu
    ../kernels/tensor_operators.cpp
    ../kernels/dropout.cu
    ../kernels/dropout.cpp
    ../kernels/sparse.cu
    ../kernels/sparse.cpp
    #../layers/param_initializers.cu
    ../rnn/attention.cpp
    ../rnn/cells.cpp
    #../optimizers/clippers.cu
    #../optimizers/optimizers.cu
    #../optimizers/optimizers.cpp
    ../common/utils.cpp
    ../common/logging.cpp
    ../common/config.cpp
    ../common/config_parser.cpp
    ../translator/history.cpp
    ../translator/output_collector.cpp
    ../translator/nth_element.cu
    ../translator/nth_element.cpp
    ../translator/helpers.cu
    ../translator/helpers.cpp
    ../data/vocab.cpp
    ../data/corpus.cpp
    ../data/text_input.cpp
    #../rescorer/score_collector.cpp
    $<TARGET_OBJECTS:libyaml-cpp>
  )
else(CUDA_FOUND)
  add_library(pymarian SHARED
    mariannmt.cpp
    ../3rd_party/cnpy/cnpy.cpp
    ../3rd_party/exception.cpp
    ../graph/expression_graph.cpp
    ../graph/expression_operators.cpp
    ../graph/node.cpp
    ../graph/node_operators.cpp
    ../tensors/tensor.cpp
    ../tensors/device_cpu.cpp
    ../kernels/tensor_operators.cpp
    ../kernels/dropout.cpp
    ../kernels/sparse.cpp
    #../layers/param_initializers.cu
    ../rnn/attention.cpp
    ../rnn/cells.cpp
    #../optimizers/clippers.cu
    #../optimizers/optimizers.cu
    #../optimizers/optimizers.cpp
    ../common/utils.cpp
    ../common/logging.cpp
    ../common/config.cpp
    ../common/config_parser.cpp
    ../translator/history.cpp
    ../translator/output_collector.cpp
    ../translator/nth_element.cpp
    ../translator/helpers.cpp
    ../data/vocab.cpp
    ../data/corpus.cpp
    ../data/text_input.cpp
    #../rescorer/score_collector.cpp
    $<TARGET_OBJECTS:libyaml-cpp>
  )
endif(CUDA_FOUND)

set_target_properties(pymarian PROPERTIES EXCLUDE_FROM_ALL 1)
set_target_properties(pymarian PROPERTIES OUTPUT_NAME mariannmt)
set_target_properties(pymarian PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set_target_properties(pymarian PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

target_link_libraries(pymarian ${EXT_LIBS} marian)
if(CUDA_FOUND)
  cuda_add_cublas_to_target(pymarian)
endif(CUDA_FOUND)
