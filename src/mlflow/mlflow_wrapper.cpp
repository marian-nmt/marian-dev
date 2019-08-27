#include <iostream>
#include <sstream>
#include <chrono>
#include <vector>

#include <jsoncpp/json/json.h>

#include <restclient-cpp/connection.h>
#include <restclient-cpp/restclient.h>

#include "mlflow/mlflow_wrapper.h"

namespace marian {
namespace mlflow {

Json::Value str2json(const std::string& jsonInput) {
      Json::Value root;
      Json::Reader reader;
      auto parsed = reader.parse(jsonInput.c_str(), root);
      if (!parsed) {
        std::cerr << "Could not parse json" << std::endl;
      }
      return root;
}

}
}

