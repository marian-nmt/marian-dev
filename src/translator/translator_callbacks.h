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
#include <thread>
#include <map>

struct TimeSentenceLatencies {
    std::ostream& os_;
    int numThreads_;
    volatile int currentIndex_;
    int batchSize_;

    // Things for thread safety with Marian
    std::map<std::thread::id, int> threadIndex_;


    // Data
    std::shared_ptr<std::vector<marian::timer::Timer>> timers_;
    std::shared_ptr<std::vector<std::vector<int>>> sentenceIds_;
    std::shared_ptr<std::vector<std::vector<double>>> times_;
    std::shared_ptr<std::vector<std::vector<std::string>>> sentences_;

    TimeSentenceLatencies(std:: ostream& os, int numThreads) : os_(os), numThreads_(numThreads), currentIndex_(0) {
        timers_ = std::make_shared<std::vector<marian::timer::Timer>>(numThreads);
        sentenceIds_ = std::make_shared<std::vector<std::vector<int>>>(numThreads);
        times_ = std::make_shared<std::vector<std::vector<double>>>(numThreads);
        sentences_ = std::make_shared<std::vector<std::vector<std::string>>>(numThreads);
    }
    
    explicit TimeSentenceLatencies(int numThreads) : TimeSentenceLatencies(std::cout, numThreads) {}

    int getThreadId(std::mutex& mutex) {
        int tid = 0;
        std::lock_guard<std::mutex> lock(mutex);
        if (threadIndex_.count(std::this_thread::get_id()) == 0) {
            threadIndex_[std::this_thread::get_id()] = currentIndex_; 
            tid = currentIndex_;
            ++currentIndex_;
        } else {
            tid = threadIndex_.at(std::this_thread::get_id());
        }
        return tid;
    }

    void resetThreadTimer(const int tid) {
        timers_->at(tid).start();
    }

    void operator()(const int sentenceId, const std::string& sentence) {
        int tid = threadIndex_.at(std::this_thread::get_id());
        
        sentenceIds_->data()[tid].push_back(sentenceId);
        sentences_->data()[tid].push_back(sentence);
        times_->data()[tid].push_back(timers_->data()[tid].elapsed());
    }

    void getTimeStatistics() const {
        // Get average and some latency percentiles
        std::vector<double> sortedTimes;
        for (size_t i = 0; i < times_->size(); ++i ) {
            sortedTimes.insert(sortedTimes.end(), times_->at(i).begin(), times_->at(i).end());
        }
        std::sort(sortedTimes.begin(), sortedTimes.end());
        double sum = std::accumulate(sortedTimes.begin(), sortedTimes.end(), 0.0);
        std::cout << "Average is " << sum / sortedTimes.size() << std::endl;
        std::cout << "50th percentile " << getPercentile(sortedTimes, 0.5) << std::endl;
        std::cout << "90th percentile " << getPercentile(sortedTimes, 0.90) << std::endl;
        std::cout << "95th percentile " << getPercentile(sortedTimes, 0.95) << std::endl;
        std::cout << "99th percentile " << getPercentile(sortedTimes, 0.99) << std::endl;
        std::cout << "99.9th percentile " << getPercentile(sortedTimes, 0.999) << std::endl;
    }

    void writeInBatchOrder() {
        // First, flatten the sentence ids and the sentences
        std::vector<int> ids;
        std::vector<std::string> sentences;
        for (size_t i = 0; i < sentences_->size(); ++i) {
            ids.insert(ids.end(), sentenceIds_->at(i).begin(), sentenceIds_->at(i).end());
            sentences.insert(sentences.end(), sentences_->at(i).begin(), sentences_->at(i).end());
        }

        // Get a vector of indices sorted by sentence ids.
        std::vector<int> indices(ids.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](const int a, const int b) -> bool {return ids[a] < ids[b];});

        // Use the sorted vector to write out the sentences in order
        for (const auto& idx : indices) {
            os_ << sentences[idx] << "\n";
        }
    }

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