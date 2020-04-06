#include "3rd_party/rapidjson/include/rapidjson/document.h"
#include "3rd_party/rapidjson/include/rapidjson/stringbuffer.h"
#include "3rd_party/rapidjson/include/rapidjson/writer.h"
#include "common/timer.h"
#include "common/utils.h"
#include "marian.h"
#include "service/api/json_request_handler.h"
#include "service/common/translation_service.h"
#include "translator/beam_search.h"
#include "translator/output_printer.h"
#include <cstdlib>
#include <sstream>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#include "crow.h"
#pragma GCC diagnostic pop


// Wrapper class for CROW (HTTP server) logging
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

typedef marian::server::TranslationService<marian::BeamSearch> tservice_t;

// Base class for handling requests. API-specific handlers (e.g. for Bergamot,
// ELG) are derived from this class.
class RequestHandler{
  const std::string gui_file_;
  const std::string src_lang_;
  const std::string trg_lang_;

  std::string
  get(const crow::request& req) const {
    crow::mustache::context ctx_;
    ctx_["URL"] = req.get_header_value("Host");
    ctx_["SOURCE_LANGUAGE"] = src_lang_;
    ctx_["TARGET_LANGUAGE"] = trg_lang_;
    return crow::mustache::load(gui_file_).render(ctx_);
  }

  virtual
  std::string
  post(const crow::request& req) const = 0;

protected:
  RequestHandler(const std::string gui_file,
                 const std::string src_lang,
                 const std::string trg_lang)
    : gui_file_(gui_file),
      src_lang_(src_lang),
      trg_lang_(trg_lang) {
  }

public:

  crow::response
  operator()(const crow::request& req) const{
    std::string method=crow::method_name(req.method);
    LOG(debug, "{} REQUEST: {}", method, req.url);
    std::string body, http_content_type;
    auto res = crow::response(200);
    res.add_header("Access-Control-Allow-Origin", "*");
    res.add_header("Access-Control-Allow-Headers", "Content-Type");

    if (method == "GET"){
      res.write(body=get(req));
      res.add_header("Content-Type", "text/html; charset=UTF-8");
    }

    else if (method == "POST"){
      res.write(body=post(req));
      res.add_header("Content-Type", "application/json; charset=UTF-8");
      LOG(debug, "RESPONSE: {}", body);
    }

    else if (method == "OPTIONS"){
      // The list of response headers below is take from this article:
      // https://developer.mozilla.org/en-US/docs/Web/HTTP/Server-Side_Access_Control

      // res.add_header("Access-Control-Allow-Origin","*");
      res.add_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
      // res.add_header("Access-Control-Allow-Headers", "X-PINGARUNER");
      // res.add_header("Access-Control-Max-Age","1728000"); // 20 days
      res.add_header("Content-Type", "text/plain; charset=UTF-8");
      res.add_header("Content-Length", "0");
    }
    else{
      return crow::response(501);
    }
    res.add_header("Content-Type", http_content_type);
    return res;
  }

};

class BergamotRequestHandler : public RequestHandler {

  marian::server::BergamotJsonRequestHandlerV1<tservice_t> process_;

  std::string
  post(const crow::request& req) const override {
    auto payload_field = req.url_params.get("payload");
    auto options_field = req.url_params.get("options");
    // to be used later, with multi-model engines
    // auto srclang = req.url_params.get("src");
    // auto trglang = req.url_params.get("trg");
    std::string payload = payload_field ? payload_field : "text";
    std::string t_opts = options_field ? options_field : "options";

    marian::Ptr<rapidjson::Document> D = process_(req.body, payload, t_opts);
    std::string response = marian::server::serialize(*D);
    return response;
  }
public:
  BergamotRequestHandler(tservice_t& service,
                         const std::string gui_file,
                         const std::string src_lang,
                         const std::string trg_lang)
    : RequestHandler(gui_file, src_lang, trg_lang), process_(service){}
};

class ElgRequestHandler : public RequestHandler {
  marian::server::ElgJsonRequestHandlerV1<tservice_t> process_;
  std::string
  post(const crow::request& req) const override {
    marian::Ptr<rapidjson::Document> D = process_(req.body.c_str());
    return marian::server::serialize(*D);
  }
public:
  ElgRequestHandler(tservice_t& service,
                    const std::string gui_file,
                    const std::string src_lang,
                    const std::string trg_lang)
    : RequestHandler(gui_file, src_lang, trg_lang), process_(service){}
};


int main(int argc, char* argv[])
{
  using namespace marian;
  using namespace marian::server;
  ConfigParser cp(cli::mode::translation);
  cp.addOption<int>("--port,-p","Server Options", "server port",18080);
  cp.addOption<int>("--queue-timeout","Server Options",
                    "max wait time (in ms) for new data before an underfull "
                    "batch is launched",100);
  cp.addOption<size_t>("--max-workers","Server Options",
                       "Maximum number of worker threads to deploy when using CPU.",
                       std::thread::hardware_concurrency());
  cp.addOption<std::string>("--server-root","Server Options",
                            "server's document root directory","./rest");
  cp.addOption<std::string>("--ssplit-prefix-file","Server Options",
                            "File with nonbreaking prefixes for sentence splitting.");
  cp.addOption<std::string>("--source-language","Server Options",
                            "source language of translation service");
  cp.addOption<std::string>("--target-language","Server Options",
                            "target language of translation service");

  auto options = cp.parseOptions(argc, argv, true);
  auto service = New<tservice_t>(options);
  service->start();

  crow::SimpleApp app;
  std::string doc_root = options->get<std::string>("server-root");
  if (doc_root.back() == '/') doc_root.pop_back();
  crow::mustache::set_base(doc_root+"/ui");

  const std::string src = options->get<std::string>("source-language","");
  const std::string trg = options->get<std::string>("target-language","");
  BergamotRequestHandler bmot_handler(*service,"bergamot_api_v1.html", src, trg);
  ElgRequestHandler elg_handler(*service,"elg_api_v1.html", src, trg);

  // For some odd reason (probably a bug in crow), a GET
  // on a path not ending in a parameter specification
  // or slash results in a 404 (not found) error.
  // This is a hack to prevent that. Unfortunately
  CROW_ROUTE(app, "/api/bergamot/v1")
    .methods("POST"_method)(bmot_handler);

  // Google Chrome automatically appends a slash to the path
  // ending in /v1 above.
  CROW_ROUTE(app, "/api/bergamot/v1/") // legacy path, deprecated ...
    .methods("GET"_method)(bmot_handler);
  CROW_ROUTE(app, "/api/bergamot/demo.html")
    .methods("GET"_method)(bmot_handler);


  CROW_ROUTE(app, "/api/elg/v1")
    .methods("POST"_method)(elg_handler);
  CROW_ROUTE(app, "/api/elg/v1/")
    .methods("GET"_method)(elg_handler);
  CROW_ROUTE(app, "/api/elg/demo.html")
    .methods("GET"_method)(elg_handler);


  app.loglevel(crow::LogLevel::WARNING);
  // app.loglevel(crow::LogLevel::DEBUG);

  LogHandler logger;
  crow::logger::setHandler(&logger);
  app.port(options->get<int>("port")).multithreaded().run();
}
