/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#pragma once
#include <boost/algorithm/string.hpp>
#include <string>
#include <vector>

void Trim(std::string& s);

void Split(const std::string& line,
           std::vector<std::string>& pieces,
           const std::string del = " ");

std::string Join(const std::vector<std::string>& words,
                 const std::string del = " ");

void Poison(float* data, size_t length); // not thread-safe
