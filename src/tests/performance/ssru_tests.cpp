#include "catch.hpp"

#include "marian.h"
#include "rnn/constructors.h"
#include "rnn/rnn.h"

#include "time_stats.h"

#include <chrono>

using namespace marian;

void test(const char* cell_name) {
  const unsigned SAMPLES = 1000;
  const unsigned DIM_INPUT = 256;
  const unsigned DIM_STATE = DIM_INPUT; // for SSRU both dim have to be equal
  const double DROP_EXTREAMS = 0.01;

  Config::seed = 1;

  auto graph = New<ExpressionGraph>();
  graph->setDevice({0, DeviceType::cpu});
  graph->reserveWorkspaceMB(16);

  auto rnn = rnn::rnn()
    ("prefix", "test")
    ("type", cell_name)
    ("dimInput", DIM_INPUT)
    ("dimState", DIM_STATE)
    .push_back(rnn::cell())
    .construct(graph);

  std::vector<double> times_ms;
  for (unsigned i = 0; i < SAMPLES; ++i)
  {
    auto input = graph->constant({DIM_INPUT, 1, DIM_STATE}, inits::glorot_uniform);
    auto output = rnn->transduce(input);

    auto start = std::chrono::high_resolution_clock::now();
    graph->forward();
    auto end = std::chrono::high_resolution_clock::now();

    times_ms.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.f);
    CHECK(output->shape() == Shape({DIM_INPUT, 1, DIM_STATE}));
  }

  times_ms = drop_extrems(times_ms, DROP_EXTREAMS);

  std::cout << "----- " << cell_name << " time (samples: " << SAMPLES << ", dropped: " << SAMPLES - times_ms.size() << " extrems) -----" << std::endl;
  std::cout << "Total:   " << total_time(times_ms) << " ms" << std::endl;
  std::cout << "Average: " << average_time(times_ms) << " ms" << std::endl;
  std::cout << "Stddev:  " << stddev_time(times_ms) << " ms" << std::endl;
  std::cout << "Min:     " << min_time(times_ms) << " ms" << std::endl;
  std::cout << "Max:     " << max_time(times_ms) << " ms" << std::endl;
}

TEST_CASE("SSRU", "CPU") { test("ssru"); }
TEST_CASE("SSRU INT8", "CPU") { test("ssru_int8"); }
TEST_CASE("SSRU INT16", "CPU") { test("ssru_int16"); }
