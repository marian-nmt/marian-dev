#include "marian.h"

#include "3rd_party/simple-websocket-server/server_ws.hpp"
#include "common/file_stream.h"
#include "common/timer.h"
#include "common/utils.h"
#include "training/training.h"
#include "translator/self_adaptive.h"

using namespace marian;

typedef SimpleWeb::SocketServer<SimpleWeb::WS> WSServer;

int main(int argc, char **argv) {
  auto options = parseOptions(argc, argv, cli::mode::selfadaptiveServer);
  auto task = New<TrainSelfAdaptive>(options);

  // Initialize web server
  WSServer server;
  server.config.port = options->get<size_t>("port", 8080);

  auto &translate = server.endpoint["^/translate/?$"];

  translate.on_message = [&task](Ptr<WSServer::Connection> connection,
                                  Ptr<WSServer::InMessage> message) {
    auto sendStream = std::make_shared<WSServer::OutMessage>();

    // Get input text
    auto inputText = message->string();

    // Translate
    timer::Timer timer;
    auto outputText = task->run(inputText);
    LOG(info, "Best translation: {}", outputText);
    *sendStream << outputText << std::endl;
    LOG(info, "Translation took: {:.5f}s", timer.elapsed());

    // Send translation back
    connection->send(sendStream, [](const SimpleWeb::error_code &ec) {
      if(ec)
        LOG(error, "Error sending message: ({}) {}", ec.value(), ec.message());
    });
  };

  // Error Codes for error code meanings
  // http://www.boost.org/doc/libs/1_55_0/doc/html/boost_asio/reference.html
  translate.on_error = [](Ptr<WSServer::Connection> connection, const SimpleWeb::error_code &ec) {
    LOG(error, "Connection error: ({}) {}", ec.value(), ec.message());
  };

  // Start server thread
  std::thread serverThread([&server]() {
    LOG(info, "Server is listening on port {}", server.config.port);
    server.start();
  });

  serverThread.join();

  return 0;
}
