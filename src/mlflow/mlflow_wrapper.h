#pragma once

#include <iostream>
#include <sstream>
#include <chrono>
#include <vector>

#include <json/json.h>

#include <restclient-cpp/connection.h>
#include <restclient-cpp/restclient.h>


Json::Value str2json(const std::string& jsonInput) {
      Json::Value root;
      Json::Reader reader;
      auto parsed = reader.parse(jsonInput.c_str(), root);
      if (!parsed) {
        std::cerr << "Could not parse json" << std::endl;
      }
      return root;
}

struct RunInfo {
  std::string runId;
  std::string experimentId;
  std::string status;
  std::string startTime;
  std::string endTime;
  std::string artifactUri;
  std::string lifecycleStage;

  explicit RunInfo() {}

  explicit RunInfo(const Json::Value& json) {
    if (json.isMember("run_id")) {
      runId = json["run_id"].asString();
    }

    if (json.isMember("run_uuid")) {
      runId = json["run_uuid"].asString();
    }

    if (json.isMember("experiment_id")) {
      experimentId = json["experiment_id"].asString();
    }

    if (json.isMember("status")) {
      status = json["status"].asString();
    }

    if (json.isMember("start_time")) {
      startTime = json["start_time"].asString();
    }

    if (json.isMember("end_time")) {
      endTime = json["end_time"].asString();
    }

    if (json.isMember("artifact_uri")) {
      artifactUri = json["artifact_uri"].asString();
    }

    if (json.isMember("lifecycle_stage")) {
      lifecycleStage = json["lifecycle_stage"].asString();
    }
  }
};


struct MLFlowMetric {
  std::string key;
  std::string value;
  int timestamp;
  int step;

};

struct MLFlowParam {
};

struct MLFlowRunData {
  std::vector<MLFlowMetric> metrics;
  std::vector<MLFlowParam> params;


};

struct RunData {
  explicit RunData(const Json::Value& json) {
  }
  explicit RunData() {}

};


struct MLFlowRun {
  MLFlowRun(RunInfo runInfo, RunData runData)
    : runInfo(runInfo), runData(runData) {}

  explicit MLFlowRun(Json::Value json) {
    if (json.isMember("info")) {
      runInfo = RunInfo(json["info"]);
    }

    if (json.isMember("data")) {
      runData = RunData(json["data"]);
    }
  }

  RunInfo runInfo;
  RunData runData;
};


class MLFlowWrapper {
public:
  MLFlowWrapper(const std::string& url, const std::string& exp_name) {
    mUrl = url;

    RestClient::Response response = RestClient::post(
        mUrl + "/api/2.0/preview/mlflow/experiments/create",
        "application/json",
        std::string("{\"name\": \"") + exp_name + "\"}");
    std::cerr << "Creating exp:" << response.code << " " << response.body << std::endl;
    if (response.code != 200) {
      RestClient::Response exps = RestClient::get(
          mUrl + "/api/2.0/preview/mlflow/experiments/list");
      std::cerr << "Getting exp list:" << exps.code << " " << exps.body << std::endl;
      Json::Value root = str2json(exps.body);

      for (auto &tt: root["experiments"]) {
        if (tt["name"] == "test-exp") {
          mExpId = tt["experiment_id"].asString();
        }
      }
    } else {
      Json::Value root = str2json(response.body);
      mExpId = root["experiment_id"].asString();
    }

    std::cerr << "ExpId: " << mExpId << std::endl;
  }

  MLFlowRun createRun() {
    std::stringstream ss;

    int timeStart = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::system_clock::now().time_since_epoch()).count();
    ss << "{";
    ss << "\"" << "experiment_id" << "\": "  << "\"" << mExpId << "\"" << ", ";
    ss << "\"" << "start_time" << "\": "  << "\"" << timeStart << "\"" << ", ";
    ss << "\"" << "tags" << "\": "  << "[]";
    ss << "}";

    RestClient::Response response = RestClient::post(
        mUrl + "/api/2.0/preview/mlflow/runs/create",
        "application/json",
        ss.str());

    std::cerr << "get run: " << response.body << std::endl;
    MLFlowRun run(str2json(response.body)["run"]);
    return run;
  }

  int logMetric(const std::string metricName, double value, int steps, const MLFlowRun& run) {
    std::stringstream ss;

    int timestamp = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::system_clock::now().time_since_epoch()).count();
    ss << "{";
    ss << "\"" << "run_id" << "\": "  << "\"" << run.runInfo.runId << "\"" << ", ";
    ss << "\"" << "key" << "\": "  << "\"" << metricName << "\"" << ", ";
    ss << "\"" << "value" << "\": "  << "\"" << value << "\"" << ", ";
    ss << "\"" << "timestamp" << "\": "  << "\"" << timestamp << "\"" << ", ";
    ss << "\"" << "step" << "\": "  << "\"" << steps << "\"";
    ss << "}";

    RestClient::Response response = RestClient::post(
        mUrl + "/api/2.0/preview/mlflow/runs/log-metric",
        "application/json",
        ss.str());

    std::cerr << "logging metrics: " << response.code << std::endl;
    std::cerr << "logging metrics: " << response.body << std::endl;

  }

protected:
  std::string getExperimentId() {
  }

private:
  std::string mUrl;
  std::string mExpId;
  std::string mRunId;
};


// int main() {
  // MLFlowWrapper wrapper("http://localhost:5000", "test-exp");
  // MLFlowRun run = wrapper.createRun();
  // for (int i  = 0; i < 10; ++i) {
    // wrapper.logMetric("BLEU", 20.00 + i, 10000 * i, run);
  // }
// }
