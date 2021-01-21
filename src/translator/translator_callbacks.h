#include <functional>
#include <ostream>
#include <string>
#include "common/timer.h"
#include <iostream>
#include <mutex>
#include <vector>

struct WriteInBatchOrder {
    std::ostream& os_;


    explicit WriteInBatchOrder(std::ostream& os) : os_(os) {}

    void operator()(const int sentenceId, const std::string& sentence) {

    }

    void writeResult() {

    }
};

struct TimeSentenceLatencies {
    marian::timer::Timer timer_;
    std::mutex mutex_;
    std::vector<double> times_;
    std::vector<std::string> sentences_;
    bool trackSentences_;

    explicit TimeSentenceLatencies(bool trackSentences) : trackSentences_(trackSentences) {}

    void startTimingBatch() {
        timer_.start();
    }

    void operator()(const int sentenceId, const std::string& sentence) {
        std::lock_guard<std::mutex> lock(mutex_);
        sentences_.push_back(sentence);
        times_.push_back(timer_.elapsed());
    }

    void getTimeStatistics() {
        // Get median, average and some latency percentiles
    }

    std::vector<std::string> getAllTranslatedSentences() {
        return sentences_;
    }

};