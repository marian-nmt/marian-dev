// -*- mode: c++; indent-tabs-mode: nil; tab-width: 2 -*-
#include "marian.h"
#include "translator/beam_search.h"
#include "translator/output_printer.h"
#include "common/timer.h"
#include "common/utils.h"

#include "translation_service.h"
#include <sstream>
#include <mutex>
#include <future>

#include "3rd_party/simple-websocket-server/server_ws.hpp"
#include "3rd_party/threadpool.h"
#include "3rd_party/rapidjson/include/rapidjson/document.h"
#include "3rd_party/rapidjson/include/rapidjson/writer.h"
#include "3rd_party/rapidjson/include/rapidjson/stringbuffer.h"

typedef SimpleWeb::SocketServer<SimpleWeb::WS> WSS;

using namespace marian;

void handle_request2(Ptr<server::TranslationService<BeamSearch>> service,
                     Ptr<WSS::Connection> conn, std::string const& msg)
{
  // @TODO: In the long run, we should be able to handle translation
  // requests in various formats (json, xml, plain text).
  // @TOOD: Offer single response at end of translation vs. ASAP responses
  // in and out of order.
  // For the time being, we only handle plain text, one sentence per line with
  // prepended line number, e.g.
  // 1 This is sentence one.
  // 2 This is sentence two.

  LOG(info, "Handling Translation Request {}", msg);
  using namespace rapidjson;
  Document D;
  D.Parse(msg.c_str());
  auto sendStream = std::make_shared<WSS::SendStream>();
  if (!D.IsObject()) {
    *sendStream << "{\"error\": \"Invalid Json!\"}";
    LOG(error, "Invalid Json: {}", msg);
  } else if (D.HasMember("text")) {
    auto input = D["text"].GetString();
    LOG(info, "Input is '{}'", input);
    auto t = service->translate(D["text"].GetString())->await();
    D["text"].SetString(t.c_str(),t.size());
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    D.Accept(writer);
    std::string response = buffer.GetString();
    LOG(info, "Response: {}", response);
    *sendStream << response;
  } else {
    *sendStream << "{\"error\": \"Input format error: no field 'text'!\"}";
    LOG(error, "No input filed text in json request.");
  }

  auto error_handler = [](const SimpleWeb::error_code &ec) {
    if(ec) {
      LOG(error, "Error sending message: ({}) {}", ec.value(), ec.message());
    }
  };
  conn->send(sendStream, error_handler);
}

int main(int argc, char* argv[])
{

  // Start the service
  ConfigParser cp(cli::mode::server);
  cp.addOption<int>("--queue-timeout","Server Options",
                    "timeout for queue in ms",100);
  auto options = cp.parseOptions(argc, argv, true);
  auto service = New<server::TranslationService<BeamSearch>>(options);
  service->start();

  auto pool = New<ThreadPool>(std::thread::hardware_concurrency(),
                              std::thread::hardware_concurrency());



  // Start the server
  WSS server;
  server.config.port = (short)options->get<size_t>("port", 8079);
  auto &translate = server.endpoint["^/translate/?$"];

  // Error Codes for error code meanings
  // http://www.boost.org/doc/libs/1_55_0/doc/html/boost_asio/reference.html
  translate.on_error
    = [](Ptr<WSS::Connection> connection, const SimpleWeb::error_code &ec) {
    LOG(error, "Connection error: ({}) {}", ec.value(), ec.message());
  };

  translate.on_message = [service, pool](Ptr<WSS::Connection> conn,
                                         Ptr<WSS::Message> msg) {
    // msg->string() consumes the buffer! A second call would return
    // an empty string, so we need to store the msg locally.
    std::string msg_text = msg->string();
    pool->enqueue(handle_request2, service, conn, msg_text);
  };

  // @TODO: signal handling for graceful shutdown
  LOG(info, "Starting Marian web socket server on port {}",
      server.config.port);
  server.start();
}
