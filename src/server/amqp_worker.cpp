/**
 *  Based on examples in the AMQP-CPP library code repository.
 *  (libuv.cpp and libev.cpp).
 *
 */

/**
 *  Dependencies
 */
#include <ev.h>
#include <amqpcpp.h>
#include <amqpcpp/libev.h>
#include "marian.h"

class MyHandler;

class amqp_receiver
{
private:
  AMQP::TcpChannel& channel_;
  std::string queue_name_;
public:
  amqp_receiver(AMQP::TcpChannel& channel) : channel_(channel) {};

  void
  listen(std::string const& queue_name)
  {
    queue_name_ = queue_name;

  }
};

class amqp_publisher
{

private:
  AMQP::TcpChannel& channel_;

public:
  amqp_publisher(AMQP::TcpChannel& channel)
    : channel_(channel)
  { }

};

/**
 *  Custom handler
 */
class MyHandler : public AMQP::LibEvHandler
{
private:
    /**
     *  Method that is called when a connection error occurs
     *  @param  connection
     *  @param  message
     */
    virtual void onError(AMQP::TcpConnection *connection, const char *message) override
    {
        std::cout << "error: " << message << std::endl;
    }

    /**
     *  Method that is called when the TCP connection ends up in a connected state
     *  @param  connection  The TCP connection
     */
    virtual void onConnected(AMQP::TcpConnection *connection) override
    {
        std::cout << "connected" << std::endl;
    }

    /**
     *  Method that is called when the TCP connection ends up in a ready
     *  @param  connection  The TCP connection
     */
    virtual void onReady(AMQP::TcpConnection *connection) override
    {
        std::cout << "ready" << std::endl;
    }

    /**
     *  Method that is called when the TCP connection is closed
     *  @param  connection  The TCP connection
     */
    virtual void onClosed(AMQP::TcpConnection *connection) override
    {
        std::cout << "closed" << std::endl;
    }

    /**
     *  Method that is called when the TCP connection is detached
     *  @param  connection  The TCP connection
     */
    virtual void onDetached(AMQP::TcpConnection *connection) override
    {
        std::cout << "detached" << std::endl;
    }


public:
    /**
     *  Constructor
     *  @param  ev_loop
     */
    MyHandler(struct ev_loop *loop) : AMQP::LibEvHandler(loop) {}

    /**
     *  Destructor
     */
    virtual ~MyHandler() = default;
};

/**
 *  Class that runs a timer
 */
class MyTimer
{
private:
    /**
     *  The actual watcher structure
     *  @var struct ev_io
     */
    struct ev_timer _timer;

    /**
     *  Pointer towards the AMQP channel
     *  @var AMQP::TcpChannel
     */
    AMQP::TcpChannel *_channel;

    /**
     *  Name of the queue
     *  @var std::string
     */
    std::string _queue;


    /**
     *  Callback method that is called by libev when the timer expires
     *  @param  loop        The loop in which the event was triggered
     *  @param  timer       Internal timer object
     *  @param  revents     The events that triggered this call
     */
    static void callback(struct ev_loop *loop, struct ev_timer *timer, int revents)
    {
        // retrieve the this pointer
        MyTimer *self = static_cast<MyTimer*>(timer->data);

        // publish a message
        self->_channel->publish("", self->_queue, "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ");
    }

public:
    /**
     *  Constructor
     *  @param  loop
     *  @param  channel
     *  @param  queue
     */
    MyTimer(struct ev_loop *loop, AMQP::TcpChannel *channel, std::string queue) :
        _channel(channel), _queue(std::move(queue))
    {
        // initialize the libev structure
        ev_timer_init(&_timer, callback, 0.005, 1.005);

        // this object is the data
        _timer.data = this;

        // and start it
        ev_timer_start(loop, &_timer);
    }

    /**
     *  Destructor
     */
    virtual ~MyTimer()
    {
        // @todo to be implemented
    }
};

using namespace marian;
Ptr<Options>
interpret_args(int argc, char** argv){
  auto cp = ConfigParser(cli::mode::translation);
  cp.addOption<std::string>("--amqp-broker", "AMQP options", // option, group
                            "AMQP message broker URI",  // option description
                            "amqp://guest:guest@localhost/"); // default value
  cp.addOption<std::string>("--amqp-queue","AMQP options",
                            "Message queue to listen on",
                            "MT-tasks");
  bool validate_options = true;
  cp.parseOptions(argc, argv, validate_options);
  auto options = New<Options>();
  options->merge(Config(cp).get());
  return options;
}

// class Publisher {
// public:
//   Publisher(std::string broker);
// };

/**
 *  Main program
 *  @return int
 */
int main(int argc, char** argv)
{
  auto opts = interpret_args(argc,argv);
  std::string broker = opts->get<std::string>("amqp-broker");
  std::string listen_queue = opts->get<std::string>("amqp-queue");


  // access to the event loop
  auto *loop = EV_DEFAULT;

  // handler for libev
  MyHandler handler(loop);

//     // init the SSL library
// #if OPENSSL_VERSION_NUMBER < 0x10100000L
//     SSL_library_init();
// #else
//     OPENSSL_init_ssl(0, NULL);
// #endif

  // make a connection
  AMQP::Address address("amqp://guest:guest@localhost/");
  //    AMQP::Address address("amqps://guest:guest@localhost/");
  AMQP::TcpConnection connection(&handler, address);

  // we need a channel too
  AMQP::TcpChannel channel(&connection);

  // create a temporary queue
  std::string qname;
  auto& queue = channel.declareQueue(listen_queue);
  queue.onSuccess([&connection, &channel, &qname, loop]
                  (const std::string &name, uint32_t messagecount,
                   uint32_t consumercount) {
                    // report the name of the temporary queue
                    std::cout << "declared queue " << name << std::endl;
                    qname = name;
                  });

  auto on_message = [&channel](const AMQP::Message &message,
                               uint64_t deliveryTag,
                               bool redelivered)
    {
      std::cout << "received message with "
      << message.bodySize() << " bytes."
      << std::endl;

      // acknowledge the message
      // channel.ack(deliveryTag);
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



  channel.publish("", qname, "Hello");

  std::cout << "published message" << std::endl;

  // run the loop
  ev_run(loop, 0);

  // done
  return 0;
}

// /**
//  *  Main program
//  *  @return int
//  */
// int main()
// {
//     // access to the event loop
//     auto *loop = uv_default_loop();

//     // handler for libev
//     MyHandler handler(loop);

//     // make a connection
//     AMQP::TcpConnection connection(&handler, AMQP::Address("amqp://guest:guest@localhost/"));

//     // we need a channel too
//     AMQP::TcpChannel channel(&connection);

//     // create a temporary queue
//     auto& queue = channel.declareQueue(AMQP::autodelete);
//     std::string queue_name;
//     queue.onSuccess([&connection,&queue_name](const std::string &name,
//                                   uint32_t messagecount,
//                                   uint32_t consumercount)
//                     {
//                       // report the name of the temporary queue
//                       queue_name = name;
//                       std::cout << "declared queue " << name << std::endl;
//                     });


//     channel.publish("", queue_name, "Hello");

//     std::cout << "published message" << std::endl;


//     // run the loop
//     uv_run(loop, UV_RUN_DEFAULT);

//     // done
//     return 0;
// }
// x
