#pragma once

#include <cmath>
#include <functional>
#include <numeric>
#include <ostream>
#include <string>
#include "common/timer.h"
#include <iostream>
#include <mutex>
#include <vector>
#include <algorithm>
#include "common/logging.h"

struct TimeSentenceLatencies {
    marian::timer::Timer timer_;
    std::mutex mutex_;
    std::vector<int> sentenceIds_;
    std::vector<double> times_;
    std::vector<std::string> sentences_;
    std::ostream& os_;

    explicit TimeSentenceLatencies(std:: ostream& os) : os_(os) {}

    void startTimingBatch() {
        timer_.start();
    }

    void operator()(const int sentenceId, const std::string& sentence) {
        std::lock_guard<std::mutex> lock(mutex_);
        sentenceIds_.push_back(sentenceId);
        sentences_.push_back(sentence);
        times_.push_back(timer_.elapsed());
    }

    void getTimeStatistics() const {
        // Get median, average and some latency percentiles
        std::vector<double> sortedTimes(times_);
        std::sort(sortedTimes.begin(), sortedTimes.end());
        double sum = std::accumulate(sortedTimes.begin(), sortedTimes.end(), 0.0);
        LOG(info, "Average is ", sum / sortedTimes.size());
        LOG(info, "50th percentile ", getPercentile(sortedTimes, 0.5));
        LOG(info, "90th percentile ", getPercentile(sortedTimes, 0.9));
        LOG(info, "95th percentile ", getPercentile(sortedTimes, 0.95));
        LOG(info, "99th percentile ", getPercentile(sortedTimes, 0.99));
        LOG(info, "99.9th percentile ", getPercentile(sortedTimes, 0.999));
    }

    const std::vector<std::string>& getAllTranslatedSentences() const {
        return sentences_;
    }

    void writeInBatchOrder() {
        std::vector<int> ids(sentenceIds_);
        std::sort(ids.begin(), ids.end());
        for (const auto id : ids) {
            os_ << sentences_[id] << "\n";
        }
    }

private:
    double getPercentile(const std::vector<double> sortedTimes, double percentile) const {
        ABORT_IF(sortedTimes.empty(), "No times available");
        const int numTimes = (int) sortedTimes.size();
        const double floatRank = percentile * (numTimes + 1);
        const int zeroIndexRank = std::max(0, ((int) floatRank) - 1);
        if (std::floor(floatRank) == floatRank) {
            return sortedTimes[zeroIndexRank];
        }

        const double frac = floatRank - std::floor(floatRank);
        const int nextElementIndex = std::min(zeroIndexRank + 1, numTimes - 1);
        return sortedTimes[zeroIndexRank] + frac * (sortedTimes[nextElementIndex] - sortedTimes[zeroIndexRank]);
    }
};