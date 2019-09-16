#include "marian.h"
#include "crow.h"
#include "translator/beam_search.h"
#include "translator/output_printer.h"
#include "common/timer.h"
#include "common/utils.h"
#include "api/elg.h"
#include "3rd_party/rapidjson/include/rapidjson/document.h"
#include "3rd_party/rapidjson/include/rapidjson/writer.h"
#include "3rd_party/rapidjson/include/rapidjson/stringbuffer.h"
#include "translation_service.h"
#include <sstream>
#include <cstdlib>
#include <cuda.h>
#include <driver_types.h>
#include <cuda_runtime.h>

class LogHandler : public crow::ILogHandler {
    public:
        void log(std::string msg, crow::LogLevel loglevel) override {
          if (loglevel == crow::LogLevel::DEBUG)
            LOG(debug,msg);
          else if (loglevel == crow::LogLevel::INFO)
            LOG(info,msg);
          else if (loglevel == crow::LogLevel::WARNING)
            LOG(warn,msg);
          else if (loglevel == crow::LogLevel::ERROR)
            LOG(error,msg);
          else if (loglevel == crow::LogLevel::CRITICAL)
            LOG(critical,msg);
        }
};

int main(int argc, char* argv[])
{
  using namespace marian;
  ConfigParser cp(cli::mode::translation);
  cp.addOption<int>("--port,-p","Server Options", "server port",18080);
  cp.addOption<int>("--queue-timeout","Server Options",
                    "max wait time (in ms) for new data before an underfull "
                    "batch is launched",5);
  cp.addOption<size_t>("--max-workers","Server Options",
                       "Maximum number of worker threads to deploy when using CPU.",
                       std::thread::hardware_concurrency());
  cp.addOption<std::string>("--server-root","Server Options",
                            "server's document root directory","./rest");

  auto options = cp.parseOptions(argc, argv, true);
  auto service = New<server::TranslationService<BeamSearch>>(options);
  service->start();

  crow::SimpleApp app;
  std::string doc_root = options->get<std::string>("server-root");
  if (doc_root.back() == '/') doc_root.pop_back();
  crow::mustache::set_base(doc_root+"/ui");

  // route for serving actual translations via the ELG API
  CROW_ROUTE(app, "/api/elg/v1")
    .methods("POST"_method)
    ([service](const crow::request& req){
      rapidjson::Document D;
      D.Parse(req.body.c_str());
      if (!D.IsObject()){
        return crow::response(200,"Invalid Json");
      }

      LOG(debug, "REQUEST: {}", server::serialize(D));
      auto R = server::elg::translate_v1(*service,D);
      std::string response = server::serialize(*R);
      LOG(debug,"RESPONSE: {}", response);
      if (R->HasMember("failure")){
        auto res = crow::response(500,response);
        res.set_header("Content-Type","application/json");
        return res;
      }
      else{
        auto res = crow::response(200,response);
        res.set_header("Content-Type","application/json");
        return res;
      }
    });

  CROW_ROUTE(app, "/api/bergamot/v1")
    .methods("POST"_method)
    ([service](const crow::request& req){
      rapidjson::Document D;
      auto payload_field = req.url_params.get("payload");
      std::string payload = payload_field ? payload_field : "text";
      std::cerr << "MESSAGE BODY IS " << req.body << std::endl;
      std::cerr << "PAYLOAD FIELD IS " << payload << std::endl;
      D.Parse(req.body.c_str());
      if (!D.IsObject()) {
        return crow::response(500,"Invalid Json");
      }
      std::cerr << "PARSED: " << server::serialize(D) << std::endl;
      server::NodeTranslation<> job(&D,*service,payload);
      job.finish(D.GetAllocator());
      std::string response = server::serialize(D);
      std::cerr << response << std::endl;
      return crow::response(response.c_str());
    });


  // route for serving the UI (templated)
  CROW_ROUTE(app, "/api/elg/v1") // GET requests
    ([](const crow::request& req){
      crow::mustache::context ctx;
      ctx["URL"] = req.get_header_value("Host");
      return crow::mustache::load("demo.html").render(ctx);
    });

  CROW_ROUTE(app, "/")
    ([](const crow::request& req){
      crow::mustache::context ctx;
      ctx["URL"] = req.get_header_value("Host");
      return crow::mustache::load("demo.html").render(ctx);
    });

  // CROW_ROUTE(app, "/<string>")
  //   ([](const crow::request& req, std::string path){
  //     crow::mustache::context ctx;
  //     ctx["URL"] = req.get_header_value("Host");
  //     std::string url = dump(ctx["URL"]);
  //     LOG(debug, "URL {}", url);
  //     LOG(debug, "PATH {}", path);
  //     return crow::mustache::load("demo.html").render(ctx);
  //   });

  app.loglevel(crow::LogLevel::WARNING);

  LogHandler logger;
  crow::logger::setHandler(&logger);
  app.port(options->get<int>("port"))
    .multithreaded()
    .run();
}
