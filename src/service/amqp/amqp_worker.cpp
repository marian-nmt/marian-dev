// -*- mode: c++; indent-tabs-mode: nil; tab-width: 2 -*-
// Author: Ulrich Germann (ulrich.germann@gmail.com)
/**
 *  Based on examples in the AMQP-CPP library code repository.
 *  (libuv.cpp and libev.cpp).
 *
 */

/**
 *  Dependencies
 */
#include <ev.h>
#include <ev++.h>
#include <amqpcpp.h>
#include <amqpcpp/libev.h>
#include "marian.h"
#include "3rd_party/threadpool.h"
#include "service/common/queue.h"
#include "service/common/translation_service.h"
#include "service/common/translation_job.h"
#include "translator/beam_search.h"
#include "3rd_party/rapidjson/include/rapidjson/document.h"
#include "3rd_party/rapidjson/include/rapidjson/writer.h"
#include "3rd_party/rapidjson/include/rapidjson/stringbuffer.h"
#include "service/amqp/event_handler.h"
#include "service/amqp/request.h"

using namespace marian;
using namespace marian::amqp;
Ptr<Options>
interpret_args(int argc, char** argv){
  auto cp = ConfigParser(cli::mode::translation);
  cp.addOption<std::string>("--amqp-broker", "AMQP options", // option, group
                            "AMQP message broker URI",  // option description
                            "amqp://guest:guest@localhost/"); // default value
  cp.addOption<std::string>("--amqp-queue","AMQP options",
                            "Message queue to listen on",
                            "MT-tasks");
  cp.addOption<std::string>("--api","AMQP options",
                            "Which API to use (bergamot|elg)",
                            "elg");
  return cp.parseOptions(argc, argv, /*validate_options=*/ true);
}

// This class is woken up when a Request has been processed
template<class Search>
class Responder {
  Ptr<marian::server::Queue<Request<Search>>> q_;
  AMQP::TcpChannel& channel_;
public:
  Responder(Ptr<marian::server::Queue<Request<Search>>> q,
            AMQP::TcpChannel& channel)
    : q_(q), channel_(channel) { }

  void operator()(ev::async& a, int revents) {
    Request<Search> R;
    std::chrono::milliseconds timeout(10);
    typename marian::server::Queue<Request<Search>>::STATUS_CODE success;
    do {
      success = q_->pop(R, timeout);
      if (success == marian::server::Queue<Request<Search>>::SUCCESS) {
        auto response = server::serialize(R.doc());
        LOG(debug, "RESPONSE: {}", response);
        AMQP::Envelope reply(response.c_str(), response.size());
        if (R.correlationID() != "")
          reply.setCorrelationID(R.correlationID());
        reply.setContentType("application/json");
        channel_.publish(R.exchange(), R.replyTo(), reply);
        channel_.ack(R.deliveryTag());
      }
    } while (success == marian::server::Queue<Request<Search>>::SUCCESS);
  }
};

int main(int argc, char** argv)
{
  using namespace marian;
  auto opts = interpret_args(argc,argv);

  // start the translation service
  auto service = New<server::TranslationService<BeamSearch>>(opts);
  service->start();

  typedef server::TranslationService<BeamSearch> tservice_t;
  Ptr<server::JsonRequestHandlerBaseClass<tservice_t> const> request_handler;
  if (opts->get<std::string>("api") == "elg"){
    request_handler.reset(new server::ElgJsonRequestHandlerV1<tservice_t>(*service));
  }
  else{
    request_handler.reset(new server::BergamotJsonRequestHandlerV1<tservice_t>(*service));
  }

  // connect to AMQP broker and establish a communication channel

  // // init the SSL library; currently not used
  // #if OPENSSL_VERSION_NUMBER < 0x10100000L
  //     SSL_library_init();
  // #else
  //     OPENSSL_init_ssl(0, NULL);
  // #endif

  auto *loop = EV_DEFAULT;
  EventHandler event_handler(loop);
  std::string broker = opts->get<std::string>("amqp-broker");
  std::string listen_queue = opts->get<std::string>("amqp-queue");
  AMQP::Address address(broker);
  AMQP::TcpConnection connection(&event_handler, address);
  AMQP::TcpChannel channel(&connection);
  // @TODO: add event handlers to reconnect if connection / channel breaks down

  // we need a thread pool to deal with requests in the background
  auto nproc = std::thread::hardware_concurrency();
  auto pool = New<ThreadPool>(nproc,nproc);

  // The Responder is a callback function that is called when a
  // request has been processed. Connections and channels cannot be
  // shared between threads, so it needs to run in the same thread
  // that listens on the AMQP input queue. Jobs finished by the
  // translation workers will be pushed onto the Responder queue /Q/,
  // from which the Responder delivers them back to the AMQP broker.
  auto Q = New<marian::server::Queue<Request<BeamSearch>>>();
  Responder<BeamSearch> responder(Q, channel);

  // qbell is a bell rung by translation workers when a finished job
  // has been pushed onto Q, to wake up the Responder.  For
  // convenience, we use the C++ LibEv interface, which always uses
  // the default event loop.
  ev::async qbell;
  qbell.set(&responder); // trigger responder when bell is rung
  qbell.start(); // start waiting for qbell events

  // listen on the input queue
  auto& queue = channel.declareQueue(listen_queue);
  queue.onSuccess([](const std::string &name,
                     uint32_t messagecount,
                     uint32_t consumercount) {
                    LOG(info, "Declared queue {}", name);
                  });

  auto on_message = [&channel,&pool,&request_handler,&Q, &qbell]
    (const AMQP::Message &message, uint64_t deliveryTag, bool redelivered)
    {

      auto R = New<Request<BeamSearch>>(deliveryTag, message, request_handler);
      auto task = [R, &Q, &qbell] () {
        R->process();
        Q->push(std::move(*R));
        qbell.send();
      };
      pool->enqueue(task);
      // // channel.ack(deliveryTag); // ack here or upon completion?
    };

  // callback function that is called when the consume operation starts
  auto on_start = [&listen_queue](const std::string &consumertag)
    {
      LOG(info, "Consume operation started on queue {}", listen_queue);
    };

  // callback function that is called when the consume operation failed
  auto on_error = [&listen_queue](const char *message)
    {
      LOG(warn, "Consume operation failed on queue {}: {}",
          listen_queue, message);
    };

  // start consuming from the queue, and install the callbacks
  channel.consume(listen_queue)
    .onReceived(on_message)
    .onSuccess(on_start)
    .onError(on_error);

  ev_run(loop, 0);
  return 0;
}
