#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

static std::vector<double> drop_extrems(const std::vector<double>& times, double percentage) {
  const unsigned p = round(times.size() * percentage);

  std::vector<double> sorted_times(times);
  sort(sorted_times.begin(), sorted_times.end());

  std::vector<double> extrems;
  for (unsigned i = 0; i < p / 2; ++i)
    extrems.push_back(sorted_times[i]);
  for (unsigned i = 0; i < (p + 1) / 2; ++i)
    extrems.push_back(sorted_times[sorted_times.size() - 1 - i]);

  std::vector<double> result;
  for (unsigned i = 0; i < times.size(); ++i) {
    auto found = false;
    for (unsigned j = 0; j < extrems.size(); ++j) {
      if (times[i] == extrems[j]) {
        extrems[j] = -1;
        found = true;
        break;
      }
    }

    if (!found)
      result.push_back(times[i]);
  }

  return result;
}

static double total_time(const std::vector<double>& times) {
  return std::accumulate(times.begin(), times.end(), 0);
}

static double average_time(const std::vector<double>& times) {
  return double(total_time(times)) / times.size();
}

static double stddev_time(const std::vector<double>& times) {
  auto avg = average_time(times);
  double squared_sum = 0;
  for (const auto& t : times)
    squared_sum += t*t;
  return sqrt(squared_sum / times.size() - avg * avg);
}

static double max_time(const std::vector<double>& times) {
  return *std::max_element(times.begin(), times.end());
}

static double min_time(const std::vector<double>& times) {
  return *std::min_element(times.begin(), times.end());
}
