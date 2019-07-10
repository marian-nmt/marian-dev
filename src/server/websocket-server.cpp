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

typedef SimpleWeb::SocketServer<SimpleWeb::WS> WSS;

using namespace marian;

// class Callback {
//   // callback function for finished translations
//   Ptr<std::promise<std::string>> prom_;
//   Ptr<server::TranslationService<BeamSearch>> service_;
// public:
//   Callback(Ptr<std::promise<std::string>> prom,
//            Ptr<server::TranslationService<BeamSearch>> service)
//     : prom_(prom), service_(service) { }

//   void operator()(uint64_t jid, Ptr<History const> h) {
//     auto snt = std::get<0>(h->NBest(1)[0]);
//     for (auto& w: snt) std::cerr << w << " "; std::cerr << std::endl;
//     if (snt.size() == 0) // skip empty translation
//       snt = std::get<0>(h->NBest(2).back());
//     if (service_->isRight2LeftDecoder())
//       std::reverse(snt.begin(),snt.end());
//     auto response = service_->vocab(-1)->decode(snt);
//     LOG(info, "Translation of job {} is {}", jid, response);
//     prom_->set_value(response);
//   }
// };


// void handle_request(Ptr<server::TranslationService<BeamSearch>> service,
//                     Ptr<WSS::Connection> conn, std::string const& msg)
// {
//   // @TODO: In the long run, we should be able to handle translation
//   // requests in various formats (json, xml, plain text).
//   // @TOOD: Offer single response at end of translation vs. ASAP responses
//   // in and out of order.
//   // For the time being, we only handle plain text, one sentence per line with
//   // prepended line number, e.g.
//   // 1 This is sentence one.
//   // 2 This is sentence two.

//   LOG(info, "Handling Translation Request {}", msg);

//   std::istringstream buf(msg);
//   std::string line;
//   std::vector<std::string> src; // source sentences
//   std::vector<size_t> jid; // job ids to go with them
//   // std::vector<std::promise<std::string>> promises(src.size());

//   while (getline(buf,line)) {
//     if(line.size()==0) continue;
//     LOG(info,line);
//     uint64_t rjid; // remote job id
//     int scanned;
//     sscanf(line.c_str(), "%zu %n", &rjid, &scanned);

//     std::cerr << scanned << " character scanned" << std::endl;
//     src.push_back(line.substr(scanned));
//     jid.push_back(rjid);
//   }

//   // create promises and futures for each input sentence
//   std::vector<std::future<std::string>> fut(src.size());
//   for(size_t i = 0; i < src.size(); ++i) {
//     Ptr<std::promise<std::string> > prom(new std::promise<std::string>);
//     fut[i] = prom->get_future();
//     service->push(i, src[i], Callback(prom, service));
//   }

//   // collect translations ...
//   auto sendStream = std::make_shared<WSS::SendStream>();
//   for (size_t i = 0; i < fut.size(); ++i) {
//     std::string response = fut[i].get();
//     LOG(info, "[{}] {}", jid[i], response);
//     *sendStream << jid[i] << " " << response << std::endl;
//   }

//   // ... and ship them
//   auto error_handler = [](const SimpleWeb::error_code &ec) {
//     if(ec) {
//       LOG(error, "Error sending message: ({}) {}", ec.value(), ec.message());
//     }
//   };
//   conn->send(sendStream, error_handler);
// }

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
  auto t = service->translate(msg);
  auto sendStream = std::make_shared<WSS::SendStream>();
  *sendStream << t;

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
  auto options = parseOptions(argc, argv, cli::mode::server, true);
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
