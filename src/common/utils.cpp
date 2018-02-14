/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#include "utils.h"
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include "common/logging.h"

void Trim(std::string& s) {
  boost::trim_if(s, boost::is_any_of(" \t\n"));
}

void Split(const std::string& line,
           std::vector<std::string>& pieces,
           const std::string del) {
  size_t begin = 0;
  size_t pos = 0;
  std::string token;
  while((pos = line.find(del, begin)) != std::string::npos) {
    if(pos > begin) {
      token = line.substr(begin, pos - begin);
      if(token.size() > 0)
        pieces.push_back(token);
    }
    begin = pos + del.size();
  }
  if(pos > begin) {
    token = line.substr(begin, pos - begin);
  }
  if(token.size() > 0)
    pieces.push_back(token);
}

std::string Join(const std::vector<std::string>& words, const std::string del) {
  std::stringstream ss;
  if(words.empty()) {
    return "";
  }
  ss << words[0];
  for(size_t i = 1; i < words.size(); ++i) {
    ss << del << words[i];
  }
  return ss.str();
}

void Poison(float* data, size_t length) {
  static enum { NO, MAYBE, YES } state = MAYBE; // not thread-safe
  if (state == MAYBE) {
    char* sz = std::getenv("MARIAN_POISON");
    state = sz != nullptr && std::strlen(sz) > 0 ? YES : NO;
    if (state == YES) {
      LOG(info)->info("Allocations poisoned to find uninitialised reads");
    }
  }

  if (state == YES) {
    /* An aid to find uninitialised reads. We use the standard-blessed way to
     * create a particular sNaN, so we can distinguish poison NaNs from others.
     */
    float sNaN_with_payload;
    long sNaN_with_payload_bits = 0x7faabcde;
    std::memcpy(&sNaN_with_payload, &sNaN_with_payload_bits, 4);
    std::fill(data, data + length, sNaN_with_payload);
  }
}
