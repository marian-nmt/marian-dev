add_subdirectory(3rd_party)

include_directories(.)
include_directories(3rd_party)
include_directories(3rd_party/SQLiteCpp/include)
include_directories(3rd_party/sentencepiece)

if(USE_SENTENCEPIECE)
  include_directories(3rd_party/sentencepiece/third_party/protobuf-lite)
endif(USE_SENTENCEPIECE)
include_directories(3rd_party/fbgemm/include)
include_directories(3rd_party/intgemm)
include_directories(${CMAKE_BINARY_DIR}/src/3rd_party/intgemm) # running cmake on the intgemm submodule triggers config file generation in this directory.
include_directories(${CMAKE_BINARY_DIR}/local/include)

set(MARIAN_SOURCES
  common/aliases.cpp
  common/blas_initializer.cpp
  common/fastopt.cpp
  common/version.cpp
  common/utils.cpp
  common/logging.cpp
  common/cli_helper.cpp
  common/cli_wrapper.cpp
  common/config.cpp
  common/config_parser.cpp
  common/config_validator.cpp
  common/options.cpp
  common/binary.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/common/build_info.cpp
  common/io.cpp
  common/filesystem.cpp
  common/file_stream.cpp
  common/file_utils.cpp
  common/signal_handling.cpp
  common/types.cpp

  data/alignment.cpp
  data/vocab.cpp
  data/default_vocab.cpp
  data/sentencepiece_vocab.cpp
  data/factored_vocab.cpp
  data/corpus_base.cpp
  data/corpus.cpp
  data/corpus_sqlite.cpp
  data/corpus_nbest.cpp
  data/text_input.cpp
  data/shortlist.cpp

  3rd_party/cnpy/cnpy.cpp
  3rd_party/ExceptionWithCallStack.cpp

  3rd_party/onnx/protobuf/onnx-ml.pb-wrapper.cpp

  3rd_party/phf/phf.cc

  tensors/backend.cpp
  tensors/rand.cpp
  tensors/tensor.cpp
  tensors/cpu/cumsum.cpp
  tensors/cpu/device.cpp
  tensors/cpu/prod.cpp
  tensors/cpu/topk.cpp
  tensors/cpu/tensor_operators.cpp
  tensors/cpu/integer_common.cpp
  tensors/cpu/fbgemm/packed_gemm.cpp
  tensors/gpu/gpu_info.cpp
  tensors/cpu/mjdgemm/mjdgemm.cpp

  graph/expression_graph.cpp
  graph/expression_operators.cpp
  graph/node.cpp
  graph/node_operators.cpp
  graph/node_initializers.cpp

  onnx/expression_graph_onnx_exporter.cpp
  onnx/expression_graph_onnx_serialization.cpp

  layers/convolution.cpp
  layers/generic.cpp
  layers/loss.cpp
  layers/weight.cpp
  layers/embedding.cpp
  layers/output.cpp
  layers/logits.cpp
  layers/lsh.cpp
  layers/lsh_generic.cpp

  layers_new/neuralnet.cpp
  layers_new/alibi.cpp
  layers_new/attention.cpp

  rnn/cells.cpp
  rnn/attention.cpp

  optimizers/quantizer.cpp
  optimizers/clippers.cpp
  optimizers/optimizers.cpp
  optimizers/exponential_smoothing.cpp

  models/model_factory.cpp
  models/encoder_decoder.cpp
  models/transformer_stub.cpp
  models/costs.cpp

  rescorer/score_collector.cpp
  embedder/vector_collector.cpp

  translator/beam_search.cpp
  translator/history.cpp
  translator/output_collector.cpp
  translator/output_printer.cpp
  translator/nth_element.cpp
  translator/helpers.cpp
  translator/scorers.cpp

  training/graph_group_async.cpp
  training/graph_group_sync.cpp
  training/graph_group.cpp
  training/graph_group_singleton.cpp
  training/validator.cpp
  training/communicator.cpp

  # this is only compiled to catch build errors
  microsoft/quicksand.cpp
  microsoft/sentencepiece.cpp
  microsoft/cosmos.cpp

  # copied from quicksand to be able to read binary shortlist
  microsoft/shortlist/utils/Converter.cpp
  microsoft/shortlist/utils/StringUtils.cpp
  microsoft/shortlist/utils/ParameterTree.cpp

  $<TARGET_OBJECTS:libyaml-cpp>
  $<TARGET_OBJECTS:SQLiteCpp>
  $<TARGET_OBJECTS:pathie-cpp>
  $<TARGET_OBJECTS:zlib>
  $<TARGET_OBJECTS:faiss>
)

add_library(marian STATIC ${MARIAN_SOURCES})
target_compile_options(marian PRIVATE ${ALL_WARNINGS})

set(MJDGEMM_OBJECTS "")

# set compile options for mjdgemm instruction sets
if(ARM)
  set(MJD_INTINSICS_FLAGS_NEON -march=armv8.2-a+simd+dotprod)

  add_library(marian_neon OBJECT tensors/cpu/mjdgemm/mjdgemm_neon.cpp layers/lsh_neon.cpp)
  target_compile_options(marian_neon PRIVATE ${ALL_WARNINGS} ${MJD_INTINSICS_FLAGS_NEON})

  set(MJDGEMM_OBJECTS ${MJDGEMM_OBJECTS} $<TARGET_OBJECTS:marian_neon>)
else()
  if(MSVC)
    set(MJD_INTINSICS_FLAGS_SSE4_2 "")
    set(MJD_INTINSICS_FLAGS_AVX2 ${MJD_INTINSICS_FLAGS_SSE4_2} /arch:AVX /arch:AVX2)
    set(MJD_INTINSICS_FLAGS_AVX512 ${MJD_INTINSICS_FLAGS_AVX2} /arch:AVX512)
  else()
    set(MJD_INTINSICS_FLAGS_SSE4_2 -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mpopcnt)
    set(MJD_INTINSICS_FLAGS_AVX2 ${MJD_INTINSICS_FLAGS_SSE4_2} -mavx -mavx2 -mfma)
    set(MJD_INTINSICS_FLAGS_AVX512 ${MJD_INTINSICS_FLAGS_AVX2} -mavx512f -mavx512bw -mavx512vnni -mavx512vpopcntdq)
  endif()

  add_library(marian_sse4_2 OBJECT tensors/cpu/mjdgemm/mjdgemm_sse4_2.cpp)
  target_compile_options(marian_sse4_2 PRIVATE ${ALL_WARNINGS} ${MJD_INTINSICS_FLAGS_SSE4_2})

  add_library(marian_avx2 OBJECT tensors/cpu/mjdgemm/mjdgemm_avx2.cpp)
  target_compile_options(marian_avx2 PRIVATE ${ALL_WARNINGS} ${MJD_INTINSICS_FLAGS_AVX2})

  add_library(marian_avx512 OBJECT tensors/cpu/mjdgemm/mjdgemm_avx512.cpp layers/lsh_avx512popcnt.cpp)
  target_compile_options(marian_avx512 PRIVATE ${ALL_WARNINGS} ${MJD_INTINSICS_FLAGS_AVX512})

  set(MJDGEMM_OBJECTS ${MJDGEMM_OBJECTS} $<TARGET_OBJECTS:marian_sse4_2> $<TARGET_OBJECTS:marian_avx2> $<TARGET_OBJECTS:marian_avx512>)
endif(ARM)

# add the mjdgemm objects to the marian library if MJDGEMM_OBJECTS is not empty
if(NOT MJDGEMM_OBJECTS STREQUAL "")
  target_sources(marian PRIVATE ${MJDGEMM_OBJECTS})
endif()

# Generate git_revision.h to reflect current git revision information
# [https://stackoverflow.com/questions/1435953/how-can-i-pass-git-sha1-to-compiler-as-definition-using-cmake]
# Git updates .git/logs/HEAD file whenever you pull or commit something.

# If Marian is checked out as a submodule in another repository,
# ${CMAKE_CURRENT_SOURCE_DIR}/../.git is not a directory but a file
# that specifies the relative path from ${CMAKE_CURRENT_SOURCE_DIR}/..
# to ./git/modules/<MARIAN_ROOT_DIR> in the root of the check_out of
# the project that contains Marian as a submodule.
#
# We set MARIAN_GIT_DIR to the appropriate path, depending on whether
# ${CMAKE_CURRENT_SOURCE_DIR}/../.git is a directory or file.
set(MARIAN_GIT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../.git)
if(NOT IS_DIRECTORY ${MARIAN_GIT_DIR}) # i.e., it's a submodule
  file(READ ${MARIAN_GIT_DIR} MARIAN_GIT_DIR)
  string(REGEX REPLACE "gitdir: (.*)\n" "\\1" MARIAN_GIT_DIR ${MARIAN_GIT_DIR})
  if(NOT IS_ABSOLUTE ${MARIAN_GIT_DIR})
    get_filename_component(MARIAN_GIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../${MARIAN_GIT_DIR}" ABSOLUTE)
  endif()
endif(NOT IS_DIRECTORY ${MARIAN_GIT_DIR})

add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/common/git_revision.h
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMAND git log -1 --pretty=format:\#define\ GIT_REVISION\ \"\%h\ \%ai\" > ${CMAKE_CURRENT_SOURCE_DIR}/common/git_revision.h
  DEPENDS ${MARIAN_GIT_DIR}/logs/HEAD
  VERBATIM
)
add_custom_target(marian_version DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/common/git_revision.h)
add_dependencies(marian marian_version) # marian must depend on it so that it gets created first
# make sure all local dependencies are installed first before this is built
add_dependencies(marian 3rd_party_installs)

if(APPLE) # This is a dependency of pathie but I can't seem to link it into that CMakeLists because we're not compiling it as a library.
   set(EXT_LIBS ${EXT_LIBS} iconv)
endif()

# Add external libs to the public interface of the library
target_link_libraries(marian ${EXT_LIBS} ${CMAKE_THREAD_LIBS_INIT})

set_target_properties(marian PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set_target_properties(marian PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

if(CUDA_FOUND)
  cuda_add_library(marian_cuda
    layers_new/alibi.cu
    tensors/gpu/cumsum.cu
    tensors/gpu/device.cu
    tensors/gpu/hash.cu
    tensors/gpu/algorithm.cu
    tensors/gpu/prod.cpp
    tensors/gpu/prod.cu
    tensors/gpu/prod_sparse.cpp
    tensors/gpu/topk.cu
    tensors/gpu/element.cu
    tensors/gpu/add.cu
    tensors/gpu/add_all.cu
    tensors/gpu/tensor_operators.cu
    tensors/gpu/cudnn_wrappers.cu
    translator/nth_element.cu
    translator/helpers.cu
    STATIC)

  target_compile_options(marian_cuda PRIVATE ${ALL_WARNINGS})
  # make sure all local dependencies are installed first before this is built
  add_dependencies(marian_cuda 3rd_party_installs)

  # Add the CUDA Marian dependency to the core Marian target
  target_link_libraries(marian marian_cuda)

  if(GENERATE_MARIAN_INSTALL_TARGETS)
    install(TARGETS marian_cuda
      EXPORT marian-targets
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
  endif(GENERATE_MARIAN_INSTALL_TARGETS)
endif(CUDA_FOUND)

# If this option is ON, only build the Marian library. This is useful if this project is included
# as a sub-project of another build system that is only interested in the Marian output library.
option(COMPILE_LIBRARY_ONLY "Build only the Marian library and exclude all executables." OFF)
if (NOT COMPILE_LIBRARY_ONLY)
  add_executable(marian_train command/marian_main.cpp)
  set_target_properties(marian_train PROPERTIES OUTPUT_NAME marian)
  target_compile_options(marian_train PRIVATE ${ALL_WARNINGS})

  add_executable(marian_decoder command/marian_decoder.cpp)
  set_target_properties(marian_decoder PROPERTIES OUTPUT_NAME marian-decoder)
  target_compile_options(marian_decoder PRIVATE ${ALL_WARNINGS})

  add_executable(marian_scorer command/marian_scorer.cpp)
  set_target_properties(marian_scorer PROPERTIES OUTPUT_NAME marian-scorer)
  target_compile_options(marian_scorer PRIVATE ${ALL_WARNINGS})

  add_executable(marian_vocab command/marian_vocab.cpp)
  set_target_properties(marian_vocab PROPERTIES OUTPUT_NAME marian-vocab)
  target_compile_options(marian_vocab PRIVATE ${ALL_WARNINGS})

  add_executable(marian_conv command/marian_conv.cpp)
  set_target_properties(marian_conv PROPERTIES OUTPUT_NAME marian-conv)
  target_compile_options(marian_conv PRIVATE ${ALL_WARNINGS})

  set(EXECUTABLES ${EXECUTABLES} marian_train marian_decoder marian_scorer marian_vocab marian_conv)

  # generate the tgz file via a custom script. This will always re-create the tarball
  add_custom_target(marian_tgz
    COMMAND ${CMAKE_COMMAND} -DCWD=${CMAKE_BINARY_DIR} -P ${CMAKE_SOURCE_DIR}/cmake/Tarball.cmake
    DEPENDS marian_train marian_decoder marian_scorer marian_vocab marian_conv)

  if(COMPILE_SERVER)
    add_executable(marian_server command/marian_server.cpp)
    set_target_properties(marian_server PROPERTIES OUTPUT_NAME marian-server)
    if(MSVC)
      # Disable warnings from the SimpleWebSocketServer library needed for compilation of marian-server
      target_compile_options(marian_server PRIVATE ${ALL_WARNINGS} /wd4267 /wd4244 /wd4456 /wd4458)
    else(MSVC)
      # -Wno-suggest-override disables warnings from Boost 1.69+
      target_compile_options(marian_server PRIVATE ${ALL_WARNINGS} -Wno-suggest-override)
    endif(MSVC)
    set(EXECUTABLES ${EXECUTABLES} marian_server)
  endif(COMPILE_SERVER)

  foreach(exec ${EXECUTABLES})
  target_link_libraries(${exec} marian)
  if(CUDA_FOUND)
    target_link_libraries(${exec} marian_cuda)
  endif(CUDA_FOUND)
  set_target_properties(${exec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  endforeach(exec)
endif(NOT COMPILE_LIBRARY_ONLY)

if(COMPILE_TESTS)
  set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party)
  add_library(Catch INTERFACE)
  target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

  add_subdirectory(tests)
endif(COMPILE_TESTS)

if(COMPILE_EXAMPLES)
  add_subdirectory(examples)
endif(COMPILE_EXAMPLES)

if(GENERATE_MARIAN_INSTALL_TARGETS)
  # Install the marian library if given a "make install" target
  include(GNUInstallDirs) # This defines default values for installation directories (all platforms even if named GNU)
  install(TARGETS marian
    EXPORT marian-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif(GENERATE_MARIAN_INSTALL_TARGETS)


if(PYMARIAN)
  # this property must be set **before** including pybind11
  # otherwise pybind will intervene with our own python version detection
  set(PYBIND11_NOPYTHON On)
  include_directories(3rd_party/pybind11/include)
  add_subdirectory(3rd_party/pybind11)
  include(PythonModules)
  # print all python versions
  message(STATUS "Going to look for these Python versions: ${PYTHON_VERSIONS}")
  add_custom_target(_pymarian)
  foreach(PYTHON_VERSION ${PYTHON_VERSIONS})  # defined in PythonModules.cmake
      py_add_module(_pymarian_${PYTHON_VERSION} python/binding/bind.cpp PYTHON_VERSION ${PYTHON_VERSION} PYTHON_MODULE _pymarian)
      target_link_libraries(_pymarian_${PYTHON_VERSION} PUBLIC marian ${EXT_LIBS})
      if(CUDA_FOUND)
        target_link_libraries(_pymarian_${PYTHON_VERSION} PUBLIC marian_cuda)
      endif(CUDA_FOUND)
      add_dependencies(_pymarian _pymarian_${PYTHON_VERSION})
  endforeach()

  # ==== make .whl files  ====
  # IMPORTANT: do not parallelize the wheel builds; they conflict on a few directories (e.g. *.egg-info)
  set(LAST_PYMARIAN_TGT "")
  foreach(PYTHON_VERSION ${PYTHON_VERSIONS})
    add_custom_target(
      pymarian_${PYTHON_VERSION} ALL
      COMMAND ${PYTHON_${PYTHON_VERSION}_EXECUTABLE} -m pip install --upgrade pip build
      COMMAND ${CMAKE_COMMAND} -E env "CMAKE_BINARY_DIR=${PROJECT_BINARY_DIR}" "CMAKE_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
              ${PYTHON_${PYTHON_VERSION}_EXECUTABLE} -m build --wheel ${PROJECT_SOURCE_DIR}/src/python -o "${PROJECT_BINARY_DIR}"
      DEPENDS _pymarian_${PYTHON_VERSION} ${LAST_PYMARIAN_TGT}
      VERBATIM COMMENT "===Building pymarian wheel for python${PYTHON_VERSION}==="
    )
    set(LAST_PYMARIAN_TGT pymarian_${PYTHON_VERSION})
  endforeach(PYTHON_VERSION)

endif(PYMARIAN)
