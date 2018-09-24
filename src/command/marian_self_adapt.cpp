#include "marian.h"

#include "3rd_party/simple-websocket-server/server_ws.hpp"
#include "common/file_stream.h"
#include "common/utils.h"
#include "training/self_adaptive.h"
#include "training/training.h"

using namespace marian;

typedef SimpleWeb::SocketServer<SimpleWeb::WS> WSServer;

static void InitializeServer(WSServer &, Ptr<ModelServiceTask>, size_t);

int main(int argc, char **argv) {
  auto options = New<Config>(argc, argv, cli::mode::selfadaptive);
  auto task = New<TrainSelfAdaptive>(options);

  if(!options->has("port")) {
    boost::timer::cpu_timer timer;
    task->run();
    LOG(info, "Total time: {}", timer.format());
  } else {
    WSServer server;
    InitializeServer(server, task, options->get<size_t>("port"));

    // start server
    std::thread server_thread([&server]() {
      LOG(info, "Server is listening on port {}", server.config.port);
      server.start();
    });
    server_thread.join();
  }

  return 0;
}

void InitializeServer(WSServer &server,
                      Ptr<ModelServiceTask> task,
                      size_t port) {
  server.config.port = port;
  auto &translate = server.endpoint["^/translate/?$"];

  translate.on_message = [&task](Ptr<WSServer::Connection> connection,
                                 Ptr<WSServer::Message> message) {
    auto message_str = message->string();

    auto message_short = message_str;
    boost::algorithm::trim_right(message_short);
    LOG(error, "Message received: {}", message_short);

    auto send_stream = std::make_shared<WSServer::SendStream>();
    boost::timer::cpu_timer timer;
    for(auto &transl : task->run({message_str})) {
      LOG(info, "Best translation: {}", transl);
      *send_stream << transl << std::endl;
    }
    LOG(info, "Translation took: {}", timer.format(5, "%ws"));

    connection->send(send_stream, [](const SimpleWeb::error_code &ec) {
      if(ec) {
        auto ec_str = std::to_string(ec.value());
        LOG(error, "Error sending message: ({}) {}", ec_str, ec.message());
      }
    });
  };

  // Error Codes for error code meanings
  // http://www.boost.org/doc/libs/1_55_0/doc/html/boost_asio/reference.html
  translate.on_error = [](Ptr<WSServer::Connection> connection,
                          const SimpleWeb::error_code &ec) {
    auto ec_str = std::to_string(ec.value());
    LOG(error, "Connection error: ({}) {}", ec_str, ec.message());
  };
}
