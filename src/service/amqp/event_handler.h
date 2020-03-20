// -*- mode: c++; indent-tabs-mode: nil; tab-width: 2 -*-
#pragma once
#include <amqpcpp.h>
#include <amqpcpp/libev.h>

// @TODO: put implementation into a .cpp file

namespace marian {
namespace amqp {
class EventHandler : public AMQP::LibEvHandler
{
private:
  virtual void
  onError(AMQP::TcpConnection *connection, const char *message) override {
    std::cout << "error: " << message << std::endl;
  }

  virtual void
  onConnected(AMQP::TcpConnection *connection) override {
    std::cout << "connected" << std::endl;
  }

  virtual void
  onReady(AMQP::TcpConnection *connection) override {
    std::cout << "ready" << std::endl;
  }

  virtual void
  onClosed(AMQP::TcpConnection *connection) override {
    std::cout << "closed" << std::endl;
  }

  virtual void
  onDetached(AMQP::TcpConnection *connection) override {
    std::cout << "detached" << std::endl;
  }
public:
  EventHandler(struct ev_loop *loop) : AMQP::LibEvHandler(loop) {}
  virtual ~EventHandler() = default;
};

}} // end of namespace marian::amqp
