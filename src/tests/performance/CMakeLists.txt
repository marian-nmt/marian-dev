set(PERFORMANCE_TESTS
  ssru_tests
)

foreach(test ${PERFORMANCE_TESTS})
  set(target "run_performance_${test}")
  add_executable("${target}" run_tests.cpp "${test}.cpp")

  if(CUDA_FOUND)
    target_link_libraries("${target}" marian marian_cuda ${EXT_LIBS} Catch)
  else(CUDA_FOUND)
    target_link_libraries("${target}" marian ${EXT_LIBS} Catch)
  endif(CUDA_FOUND)

  add_test(NAME ${test} COMMAND "${target}")
endforeach(test)
