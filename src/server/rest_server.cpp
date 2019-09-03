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
                    "max wait for new data before batch is launched",10);
  cp.addOption<std::string>("--server-root","Server Options",
                            "server's document root directory","./rest");

  auto options = cp.parseOptions(argc, argv, true);
  auto service = New<server::TranslationService<BeamSearch>>(options);
  service->start();

  // crow::App<ExampleMiddleware> app;
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
      auto R = server::elg::translate_v1(*service,D);
      return crow::response(500,"Invalid Json");
      // auto payload = crow::json::load(req.body);
      // if (!payload){ // parsing failed
      //   return crow::response(500,"Invalid Json");
      // }
      // LOG(debug, "REQUEST BODY IS {}", payload);
      // auto foo = marian::server::elg::translate_v1(*service, payload);
      // LOG(debug, "RESPONSE IS {}", crow::json::dump(foo));
      // if (foo.has("response")){
      //   return crow::response(200, crow::json::dump(foo));
      // }
      // return crow::response(500, crow::json::dump(foo));
      // rapidjson::Document D;
      // std::cerr << "MESSAGE BODY IS " << req.body << std::endl;
      // D.Parse(req.body.c_str());
      // if (!D.IsObject()) {
      //   return crow::response(500,"Invalid Json");
      // }
      // std::cerr << "PARSED: " << server::serialize(D) << std::endl;
      // if (!D.HasMember("request") || !D["request"].IsObject() ||
      //     !D["request"].HasMember("content")){
      //   return crow::response(500,"Invalid Request Structure");
      // }
      // std::string input = D["request"]["content"].GetString();
      // std::string translation = service->translate(input);
      // return crow::response(200,translation);
    });

  CROW_ROUTE(app, "/api/ug/v1")
    .methods("POST"_method)
    ([service](const crow::request& req){
      rapidjson::Document D;
      std::cerr << "MESSAGE BODY IS " << req.body << std::endl;
      D.Parse(req.body.c_str());
      if (!D.IsObject()) {
        return crow::response(500,"Invalid Json");
      }
      std::cerr << "PARSED: " << server::serialize(D) << std::endl;
      server::NodeTranslation<> job(&D,*service);
      job.finish(D.GetAllocator());
      std::string response = server::serialize(D);
      std::cerr << response << std::endl;
      return crow::response(response.c_str());
    });


  // route for serving the UI (templated)
  CROW_ROUTE(app, "/")
    ([](const crow::request& req){
      crow::mustache::context ctx;
      ctx["URL"] = req.get_header_value("Host");
      return crow::mustache::load("demo.html").render(ctx);
    });

  // // route for serving files
  // CROW_ROUTE(app, "/<string>")
  //   ([](const crow::request& req, std::string path){
  //     marian::filesystem::Path p(doc_root+path);
  //     if (!p.exists())
  //       return crow::response(404, path+" not found");
  //     ifstream(doc_root+path);


  // //     crow::mustache::context ctx;
  // //     ctx["url"] = req.raw_url;
  // //     std::cout << "URL " << req.get_header_value("Host") << req.url << std::endl;
  // //     auto page = crow::mustache::load("demo.html");
  // //     return page.render(ctx);

  //     if (marian::filesystem::exists(path))
  //       return crow::mustache::load(path).render();
  //     return page.render();
  //   });

  // enables all log
  app.loglevel(crow::LogLevel::WARNING);
  //crow::logger::setHandler(std::make_shared<ExampleLogHandler>());

//            cerr << "ExampleLogHandler -> " << message;
  LogHandler logger;
  crow::logger::setHandler(&logger);
  app.port(options->get<int>("port"))
    .multithreaded()
    .run();
}
