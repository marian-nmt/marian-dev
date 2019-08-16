// -*- mode: c++; indent-tabs-mode: nil; tab-width: 2 -*-
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
#include "server/queue.h"
#include "server/translation_service.h"
#include "server/translation_job.h"
#include "translator/beam_search.h"
#include "3rd_party/rapidjson/include/rapidjson/document.h"
#include "3rd_party/rapidjson/include/rapidjson/writer.h"
#include "3rd_party/rapidjson/include/rapidjson/stringbuffer.h"
#include "server/amqp/event_handler.h"
#include "server/amqp/request.h"

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
  return New<Options>(cp, argc, argv, /*validate_options=*/ true);
}

// This class is woken up when a Request has been processed
class Responder {
  Ptr<marian::server::Queue<Request<>>> q_;
  AMQP::TcpChannel& channel_;
public:
  Responder(Ptr<marian::server::Queue<Request<>>> q,
            AMQP::TcpChannel& channel)
    : q_(q), channel_(channel) { }

  void operator()(ev::async& a, int revents) {
    Request<> R;
    std::chrono::milliseconds timeout(10);
    marian::server::Queue<Request<>>::STATUS_CODE success;
    do {
      success = q_->pop(R, timeout);
      if (success == marian::server::Queue<Request<>>::SUCCESS) {
        auto response = server::serialize(R.doc());
        std::cout << response << std::endl;
        AMQP::Envelope reply(response.c_str(), response.size());
        if (R.correlationID() != "")
          reply.setCorrelationID(R.correlationID());
        reply.setContentType("application/json");
        channel_.publish(R.exchange(), R.replyTo(), reply);
        channel_.ack(R.deliveryTag());
      }
    } while (success == marian::server::Queue<Request<>>::SUCCESS);
  }
};

/**
 *  Main program
 *  @return int
 */
int main(int argc, char** argv)
{
  using namespace marian;
  auto opts = interpret_args(argc,argv);

  // start the translation service
  auto service = New<server::TranslationService<BeamSearch>>(opts);
  service->start();

  // connect to AMQP broker and establish a communication channel

  // // init the SSL library; currently not used
  // #if OPENSSL_VERSION_NUMBER < 0x10100000L
  //     SSL_library_init();
  // #else
  //     OPENSSL_init_ssl(0, NULL);
  // #endif

  auto *loop = EV_DEFAULT;
  EventHandler handler(loop);
  std::string broker = opts->get<std::string>("amqp-broker");
  std::string listen_queue = opts->get<std::string>("amqp-queue");
  AMQP::Address address(broker);
  AMQP::TcpConnection connection(&handler, address);
  AMQP::TcpChannel channel(&connection);
  // @TODO: add event handlers if connection / channel breaks down


  // Set up a queue for finished jobs that the Responder function
  // will read from.
  auto Q = New<marian::server::Queue<Request<>>>();

  // we need a thread pool to deal with requests in the background
  auto nproc = std::thread::hardware_concurrency();
  auto pool = New<ThreadPool>(nproc,nproc);


  // The responder is a callback function that is called when a
  // request has been processed. Connections and channels cannot be
  // shared between threads.
  Responder responder(Q, channel);

  ev::async qbell; // a bell to ring when a job is done
  // Note that the C++ LibEv interface always uses the default event
  // loop.
  qbell.set(&responder); // trigger responder when bell is rung
  qbell.start(); // start waiting for qbell events

  // listen on the input queue
  auto& queue = channel.declareQueue(listen_queue);
  queue.onSuccess([](const std::string &name, uint32_t messagecount,
                     uint32_t consumercount) {
                    std::cout << "declared queue " << name << std::endl;
                  });

  auto on_message = [&channel,&pool,&service,&Q, &qbell]
    (const AMQP::Message &message, uint64_t deliveryTag, bool redelivered)
    {
      auto R = New<Request<>>(deliveryTag, message, service);
      auto task = [R, &Q, &qbell] () {
        R->process();
        Q->push(std::move(*R));
        qbell.send();
      };
      pool->enqueue(task);
      // // channel.ack(deliveryTag); // ack here or upon completion?
    };

  // callback function that is called when the consume operation starts
  auto on_start = [](const std::string &consumertag)
    {
      std::cout << "consume operation started" << std::endl;
    };

  // callback function that is called when the consume operation failed
  auto on_error = [](const char *message)
    {
      std::cout << "consume operation failed" << std::endl;
    };

  // start consuming from the queue, and install the callbacks
  channel.consume(listen_queue)
    .onReceived(on_message)
    .onSuccess(on_start)
    .onError(on_error);

  ev_run(loop, 0);
  return 0;
}
